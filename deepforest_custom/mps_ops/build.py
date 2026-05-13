"""
Compile the MPS fp16 BatchNorm backward extension.

torch.utils.cpp_extension.load() doesn't quote paths with spaces in ninja
build files, which breaks on paths like 'dphil icloud'.  We invoke clang
directly with subprocess so every path is properly shell-quoted.
"""
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.utils.cpp_extension as _ext

_HERE  = Path(__file__).resolve().parent
_OUT   = Path("/tmp/canopyai_mps_ops")
_SO    = _OUT / "canopyai_mps_ops.so"
_OBJ   = _OUT / "mps_bn_fp16.o"
_SRC   = _HERE / "mps_bn_fp16.mm"


def _needs_rebuild() -> bool:
    if not _SO.exists():
        return True
    return _SRC.stat().st_mtime > _SO.stat().st_mtime


def build():
    if not _needs_rebuild():
        # Already up to date — just load the .so
        torch.ops.load_library(str(_SO))
        return

    _OUT.mkdir(parents=True, exist_ok=True)

    torch_dir   = Path(torch.__file__).parent
    torch_inc   = _ext.include_paths()          # list of include dirs
    sdk_path    = subprocess.check_output(["xcrun", "--show-sdk-path"]).decode().strip()
    python_inc  = subprocess.check_output([sys.executable, "-c",
                      "import sysconfig; print(sysconfig.get_path('include'))"]).decode().strip()

    # ── Compile ─────────────────────────────────────────────────────────
    compile_cmd = [
        "clang++",
        "-std=c++17", "-ObjC++",
        "-fPIC", "-O2", "-w",
        "-DAT_PER_OPERATOR_HEADERS",
        "-DTORCH_EXTENSION_NAME=canopyai_mps_ops",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        f"-I{sdk_path}/System/Library/Frameworks",
        f"-I{python_inc}",
    ]
    for inc in torch_inc:
        compile_cmd += [f"-I{inc}"]
    compile_cmd += ["-c", str(_SRC), "-o", str(_OBJ)]

    print(f"[mps_ops] Compiling {_SRC.name} ...")
    subprocess.run(compile_cmd, check=True)

    # ── Link ─────────────────────────────────────────────────────────────
    link_cmd = [
        "clang++",
        "-shared", "-undefined", "dynamic_lookup",
        str(_OBJ),
        f"-L{torch_dir / 'lib'}",
        "-lc10", "-ltorch_cpu", "-ltorch",
        "-framework", "Metal",
        "-framework", "MetalPerformanceShaders",
        "-framework", "MetalPerformanceShadersGraph",
        "-framework", "Foundation",
        "-o", str(_SO),
    ]

    print(f"[mps_ops] Linking -> {_SO} ...")
    subprocess.run(link_cmd, check=True)

    # ── Load ─────────────────────────────────────────────────────────────
    torch.ops.load_library(str(_SO))
    print("[mps_ops] canopyai_mps_ops compiled and loaded.")


if __name__ == "__main__":
    build()
