# Solar-Gated Detection Pipeline Architecture

```mermaid
graph TD
    subgraph "Stage 0: Global Context Inference"
        Ortho[Full Orthomosaic] -->|"Sample N random 500x500 crops"| Crops[Random Crops]
        Crops -->|ResNet18 backbone| GCE["GlobalContextEncoder"]
        GCE -->|"projection_head → normalize"| Predictions["Per-Crop Sun Vectors (N,2)"]
        Predictions -->|Circular Mean| SunVec["sun_vector (B,2)"]
    end

    subgraph "Stage 1: SolarRetinaNet"
        Tile[High-Res 500x500 Tile] -->|"ResNet50 (oscar50 weights)"| Backbone[backbone]
        Backbone -->|C3, C4, C5| FPN["FPN features (B,256,H,W)"]
        
        subgraph SolarGate["SolarAttentionBlock (per FPN level)"]
            SunVec -.->|"FiLM: (B,2)"| MLP["sun_mlp: 2→64→256"]
            MLP -->|"sun_embedding (B,256,1,1)"| Add(("+"))
            FPNIn["x: FPN features (B,256,H,W)"] -->|"broadcast add"| Add
            Add -->|"conditioned_features"| Conv3x3["gate_conv: 3×3 → 1ch"]
            Conv3x3 --> Sigmoid["sigmoid"]
            Sigmoid -->|"attn_map (B,1,H,W)"| Gate(("×"))
            FPNIn -.->|"x (original)"| Gate
            Gate --> GatedOut["out (B,256,H,W)"]
        end
        
        FPN --> FPNIn
        GatedOut --> BoxHead["Box Regression Head (B, A, 4)"]
        GatedOut --> ClassHead["Classification Head (B, A, 1)"]
        
        BoxHead --> NMS["NMS + Score Filtering"]
        ClassHead --> NMS
        NMS --> Detections["Final Detections (boxes + scores)"]
    end

    subgraph "Stage 2: Instance Segmentation"
        Detections --> SAM["SAM (Refined)"]
        SAM --> Masks[Final Canopy Masks]
    end


```

## Description of Components

### 1. Global Context Encoder (Stage 0)
*   **Input:** N random 500x500 crops from the orthomosaic (using existing pre-tiled training data).
*   **Model:** Lightweight CNN (ResNet18).
*   **Inference:** Sample ~30 crops, predict sun vector for each, compute **circular mean** for consensus.
*   **Training Objective:** **Self-Supervised Rotation Consistency**.
    *   We rotate the input image $I$ by angle $\theta$.
    *   The output vector $\vec{z}$ must rotate by $\theta$.
    *   This forces $\vec{z}$ to lock onto the strongest directional signal (shadows).

### 2. Solar Attention Gate (The Core Innovation)
*   **Location:** Inserted after the FPN, before the RetinaNet prediction heads.
*   **Mechanism:** Projects `SunVector` via MLP to create a per-channel bias (FiLM conditioning), adds this to FPN features, then applies a **3×3 conv** to generate a spatial attention map.
*   **Effect:** Learns to detect shadow-consistent patterns: *"If Sun is North and I see dark-to-the-south-of-bright, boost attention here."*

