// mps_bn_fp16.mm
//
// Custom PyTorch kernel: BatchNorm backward for fp16 tensors on Apple MPS.
//
// PyTorch's built-in MPS native_batch_norm_backward rejects fp16 inputs with
// an explicit dtype check.  This kernel bypasses that and implements the
// standard BN backward formula using basic MPSGraph ops (no specialised BN
// gradient API) — every op used here (multiply, subtract, sum, reshape,
// divide) is well-supported for fp32 on MPS.  fp16 inputs are cast to fp32
// at the graph boundary; outputs are cast back to fp16.
//
// Registered as:  canopyai::mps_bn_backward_fp16
//
// BatchNorm backward formula (training mode):
//   m        = N * H * W               (elements per channel)
//   x_hat    = (x - mean) * invstd     (saved normalised input)
//   dy       = grad_out * gamma
//   sum_dy   = sum(dy, [0,2,3])
//   sum_dyx  = sum(dy * x_hat, [0,2,3])
//   grad_x   = invstd * (dy - sum_dy/m - x_hat * sum_dyx/m)
//   grad_g   = sum(grad_out * x_hat, [0,2,3])
//   grad_b   = sum(grad_out, [0,2,3])

#include <ATen/Tensor.h>
#include <torch/library.h>
#include <c10/util/Optional.h>
#include <ATen/native/mps/OperationUtils.h>

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

using namespace at;
using namespace at::mps;
using namespace at::native::mps;

namespace canopyai {

// ---------------------------------------------------------------------------
// Graph cache entry
// ---------------------------------------------------------------------------

struct BNBackwardGraph : MPSCachedGraph {
    BNBackwardGraph(MPSGraph* g) : MPSCachedGraph(g) {}

    // Placeholders (feeds)
    MPSGraphTensor* gradOut    = nil;
    MPSGraphTensor* input      = nil;
    MPSGraphTensor* weight     = nil;
    MPSGraphTensor* saveMean   = nil;
    MPSGraphTensor* saveInvstd = nil;

    // Results
    MPSGraphTensor* gradInput  = nil;
    MPSGraphTensor* gradWeight = nil;
    MPSGraphTensor* gradBias   = nil;
};

// ---------------------------------------------------------------------------
// Build the MPSGraph for BN backward (manual formula, fp32 internally)
// ---------------------------------------------------------------------------

static void build_bn_backward_graph(
    MPSGraph* graph,
    BNBackwardGraph* g,
    MPSShape* xShape,      // [N, C, H, W]
    MPSShape* cShape,      // [C]
    MPSDataType xType,     // caller dtype (fp16 or fp32)
    int64_t N, int64_t C, int64_t H, int64_t W)
{
    auto fp32 = MPSDataTypeFloat32;

    // Input placeholders in caller dtype
    g->gradOut    = mpsGraphRankedPlaceHolder(graph, xType, xShape);
    g->input      = mpsGraphRankedPlaceHolder(graph, xType, xShape);
    g->weight     = mpsGraphRankedPlaceHolder(graph, xType, cShape);
    g->saveMean   = mpsGraphRankedPlaceHolder(graph, fp32,  cShape);
    g->saveInvstd = mpsGraphRankedPlaceHolder(graph, fp32,  cShape);

    // Cast all fp16 inputs to fp32 for stable computation
    auto gradF  = [graph castTensor:g->gradOut    toType:fp32 name:@"gO_f32"];
    auto inputF = [graph castTensor:g->input      toType:fp32 name:@"in_f32"];
    auto wF     = [graph castTensor:g->weight     toType:fp32 name:@"w_f32"];

    // Reshape [C] → [1, C, 1, 1] for broadcasting with [N, C, H, W]
    NSArray<NSNumber*>* bcShape = @[@1, @(C), @1, @1];

    auto meanBC   = [graph reshapeTensor:g->saveMean    withShape:bcShape name:@"mean_bc"];
    auto invstdBC = [graph reshapeTensor:g->saveInvstd  withShape:bcShape name:@"invstd_bc"];
    auto wBC      = [graph reshapeTensor:wF              withShape:bcShape name:@"w_bc"];

    // Reduction axes: [0, 2, 3] — reduce over batch, height, width
    NSArray<NSNumber*>* redAxes = @[@0, @2, @3];

    // 1-D [C] shape for output scalars-per-channel
    NSArray<NSNumber*>* cShape1d = @[@((NSInteger)C)];

    // m = N*H*W  (scalar constant, fp32)
    float m_val   = (float)(N * H * W);
    auto m_tensor = [graph constantWithScalar:m_val dataType:fp32];

    // x_hat = (x - mean) * invstd  -- [N, C, H, W]
    auto x_mu  = [graph subtractionWithPrimaryTensor:inputF
                                     secondaryTensor:meanBC
                                                name:@"x_mu"];
    auto x_hat = [graph multiplicationWithPrimaryTensor:x_mu
                                        secondaryTensor:invstdBC
                                                   name:@"x_hat"];

    // dy = grad_out * gamma  -- [N, C, H, W]
    auto dy = [graph multiplicationWithPrimaryTensor:gradF
                                     secondaryTensor:wBC
                                                name:@"dy"];

    // Reduce dy and dy*x_hat over [0,2,3].
    // Reshape immediately to [1,C,1,1] for use in elementwise operations.
    // This explicit reshape handles any keepDims ambiguity across MPSGraph versions.
    auto sum_dy_raw  = [graph reductionSumWithTensor:dy    axes:redAxes name:@"sum_dy_raw"];
    auto sum_dy_bc   = [graph reshapeTensor:sum_dy_raw  withShape:bcShape name:@"sum_dy_bc"];

    auto dy_xhat     = [graph multiplicationWithPrimaryTensor:dy
                                              secondaryTensor:x_hat
                                                         name:@"dy_xhat"];
    auto sum_dyx_raw = [graph reductionSumWithTensor:dy_xhat axes:redAxes name:@"sum_dyx_raw"];
    auto sum_dyx_bc  = [graph reshapeTensor:sum_dyx_raw withShape:bcShape name:@"sum_dyx_bc"];

    // grad_x = invstd * (dy  -  sum_dy/m  -  x_hat * sum_dyx/m)
    auto sdy_m   = [graph divisionWithPrimaryTensor:sum_dy_bc  secondaryTensor:m_tensor name:@"sdy_m"];
    auto sdyx_m  = [graph divisionWithPrimaryTensor:sum_dyx_bc secondaryTensor:m_tensor name:@"sdyx_m"];
    auto term1   = [graph subtractionWithPrimaryTensor:dy      secondaryTensor:sdy_m   name:@"t1"];
    auto xh_sdyx = [graph multiplicationWithPrimaryTensor:x_hat secondaryTensor:sdyx_m name:@"xh_sdyx"];
    auto term2   = [graph subtractionWithPrimaryTensor:term1    secondaryTensor:xh_sdyx name:@"t2"];
    auto gradInputF  = [graph multiplicationWithPrimaryTensor:term2 secondaryTensor:invstdBC name:@"gi_f32"];

    // grad_gamma = sum(grad_out * x_hat, [0,2,3]) -> reshape to [C]
    auto go_xhat    = [graph multiplicationWithPrimaryTensor:gradF secondaryTensor:x_hat name:@"go_xhat"];
    auto gradGam_raw = [graph reductionSumWithTensor:go_xhat axes:redAxes name:@"gg_raw"];
    auto gradGamF    = [graph reshapeTensor:gradGam_raw withShape:cShape1d name:@"gg_f32"];

    // grad_beta = sum(grad_out, [0,2,3]) -> reshape to [C]
    auto gradBet_raw = [graph reductionSumWithTensor:gradF axes:redAxes name:@"gb_raw"];
    auto gradBetF    = [graph reshapeTensor:gradBet_raw withShape:cShape1d name:@"gb_f32"];

    // Cast results back to caller dtype
    g->gradInput  = [graph castTensor:gradInputF toType:xType name:@"gi_out"];
    g->gradWeight = [graph castTensor:gradGamF   toType:xType name:@"gw_out"];
    g->gradBias   = [graph castTensor:gradBetF   toType:xType name:@"gb_out"];
}

// ---------------------------------------------------------------------------
// Main op implementation
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor, Tensor>
mps_bn_backward_fp16(
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& save_mean,    // fp32 [C]
    const Tensor& save_invstd,  // fp32 [C]  (= 1/sqrt(var+eps))
    double /*eps*/)
{
    TORCH_CHECK(input.device().type() == kMPS,
                "mps_bn_backward_fp16: input must be on MPS device");
    TORCH_CHECK(input.is_contiguous(), "mps_bn_backward_fp16: input must be contiguous");
    TORCH_CHECK(grad_out.is_contiguous(), "mps_bn_backward_fp16: grad_out must be contiguous");
    TORCH_CHECK(input.dim() == 4, "mps_bn_backward_fp16: only NCHW tensors supported");

    auto grad_input  = at::empty_like(input);
    auto grad_weight = at::empty_like(weight);
    auto grad_bias   = at::empty_like(weight);

    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t H = input.size(2);
    const int64_t W = input.size(3);

    auto xShape = getMPSShape(input);
    auto cShape = getMPSShape(save_mean);
    auto xType  = getMPSDataType(input.scalar_type());

    @autoreleasepool {
        auto key = std::string("canopyai_bn_bwd_")
                   + getMPSShapeString(xShape) + "_"
                   + getMPSTypeString(input.scalar_type());

        auto* cachedGraph = at::native::mps::LookUpOrCreateCachedGraph<BNBackwardGraph>(
            key,
            [&](MPSGraph* graph, BNBackwardGraph* g) {
                build_bn_backward_graph(graph, g, xShape, cShape, xType, N, C, H, W);
            }
        );

        MPSStream* stream = getCurrentMPSStream();

        Placeholder goP(cachedGraph->gradOut,    grad_out);
        Placeholder inP(cachedGraph->input,      input);
        Placeholder wP (cachedGraph->weight,     weight);
        Placeholder mP (cachedGraph->saveMean,   save_mean);
        Placeholder iP (cachedGraph->saveInvstd, save_invstd);
        Placeholder giP(cachedGraph->gradInput,  grad_input);
        Placeholder gwP(cachedGraph->gradWeight, grad_weight);
        Placeholder gbP(cachedGraph->gradBias,   grad_bias);

        NSDictionary* feeds = @{
            goP.getMPSGraphTensor(): goP.getMPSGraphTensorData(),
            inP.getMPSGraphTensor(): inP.getMPSGraphTensorData(),
            wP.getMPSGraphTensor():  wP.getMPSGraphTensorData(),
            mP.getMPSGraphTensor():  mP.getMPSGraphTensorData(),
            iP.getMPSGraphTensor():  iP.getMPSGraphTensorData(),
        };

        NSDictionary* results = @{
            giP.getMPSGraphTensor(): giP.getMPSGraphTensorData(),
            gwP.getMPSGraphTensor(): gwP.getMPSGraphTensorData(),
            gbP.getMPSGraphTensor(): gbP.getMPSGraphTensorData(),
        };

        runMPSGraph(stream, cachedGraph->graph(), feeds, results);
    }

    return {grad_input, grad_weight, grad_bias};
}

// ---------------------------------------------------------------------------
// Op registration
// ---------------------------------------------------------------------------

TORCH_LIBRARY(canopyai, m) {
    m.def("mps_bn_backward_fp16("
          "Tensor grad_out, Tensor input, Tensor weight, "
          "Tensor save_mean, Tensor save_invstd, float eps"
          ") -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(canopyai, MPS, m) {
    m.impl("mps_bn_backward_fp16", &mps_bn_backward_fp16);
}

} // namespace canopyai
