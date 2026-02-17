#!/usr/bin/env python3
"""Inspect PyTorch checkpoint contents and sizes."""

import torch
import sys
from pathlib import Path


def inspect_checkpoint(checkpoint_path):
    """Load and inspect a PyTorch checkpoint file."""
    print(f"\n{'=' * 60}")
    print(f"Inspecting: {checkpoint_path}")
    print(f"File size: {Path(checkpoint_path).stat().st_size / (1024 * 1024):.1f} MB")
    print(f"{'=' * 60}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    print(f"\nTop-level keys:")
    for key in ckpt.keys():
        print(f"  - {key}")

    print(f"\nDetailed breakdown:")
    total_params = 0

    for key, value in ckpt.items():
        if isinstance(value, dict):
            # Count parameters in nested dict (like 'model' or 'optimizer')
            num_tensors = sum(1 for v in value.values() if torch.is_tensor(v))
            num_params = sum(v.numel() for v in value.values() if torch.is_tensor(v))
            total_params += num_params
            print(f"\n  {key}:")
            print(f"    Type: dict with {len(value)} items")
            print(f"    Tensors: {num_tensors}")
            print(f"    Parameters: ~{num_params / 1e6:.2f}M")
        elif torch.is_tensor(value):
            total_params += value.numel()
            print(f"\n  {key}:")
            print(f"    Type: tensor")
            print(f"    Shape: {value.shape}")
            print(f"    Parameters: {value.numel()}")
        else:
            print(f"\n  {key}:")
            print(f"    Type: {type(value).__name__}")
            print(f"    Value: {value}")

    print(f"\nTotal parameters stored: ~{total_params / 1e6:.2f}M")
    print(f"Estimated size (FP32): ~{total_params * 4 / (1024 * 1024):.1f} MB")

    # Specific check for FiLM layers
    print("\n🔍 Checking for FiLM layers...")
    film_keys = []

    # Handle state_dict vs direct dict
    state_dict = ckpt
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]

    for key in state_dict.keys():
        if "film_blocks" in key:
            film_keys.append(key)

    if film_keys:
        print(f"✅ Found {len(film_keys)} FiLM parameter keys.")
        print("Sample keys:")
        for k in film_keys[:5]:
            print(f"  - {k}")
        if len(film_keys) > 5:
            print(f"  ... and {len(film_keys) - 5} more")
    else:
        print(
            "❌ No 'film_blocks' keys found. This model likely does not have FiLM layers trained."
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py <checkpoint.pth>")
        sys.exit(1)

    for checkpoint_path in sys.argv[1:]:
        inspect_checkpoint(checkpoint_path)
