# Solar-Gated Detection Pipeline Architecture

```mermaid
graph TD
    subgraph "Stage 0: Global Context Inference"
        Ortho[Full Orthomosaic] -->|"Sample N random 500x500 crops"| Crops[Random Crops]
        Crops -->|ResNet18| ContextEncoder[Context Encoder]
        ContextEncoder -->|Regress| Predictions["Per-Crop Sun Vectors (N,2)"]
        Predictions -->|Circular Mean| SunVec["Global Sun Vector (1,2)"]
    end

    subgraph "Stage 1: Solar-Gated Detection"
        Tile[High-Res 500x500 Tile] -->|Fine-tuned ResNet| Backbone[ResNet Backbone]
        Backbone -->|C3, C4, C5| FPN["Feature Pyramid Network (B,256,H,W)"]
        
        subgraph SolarGate["Solar Gate Layer (per FPN level)"]
            SunVec -.->|"FiLM: (B,2)"| MLP["MLP: 2→256"]
            MLP -->|"Channel Bias (B,256,1,1)"| Add(("+"))
            FPNIn["FPN Features (B,256,H,W)"] -->|"broadcast add"| Add
            Add -->|"Conditioned (B,256,H,W)"| Conv3x3["3×3 Conv → 1ch"]
            Conv3x3 --> Sigmoid["σ: Sigmoid"]
            Sigmoid -->|"Attention (B,1,H,W)"| Gate(("×"))
            FPNIn -.->|"Original features"| Gate
            Gate --> GatedOut["Gated Features (B,256,H,W)"]
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

