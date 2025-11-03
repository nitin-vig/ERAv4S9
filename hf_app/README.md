# Hugging Face CPU App

A CPU-optimized Gradio app that loads ResNet50 weights from `checkpoints/checkpoint_best.pth` and serves top-5 ImageNet predictions.

## Features
- Pure CPU inference (`map_location='cpu'`)
- Dataset normalization from `checkpoints/dataset_stats.json` when available
- Robust checkpoint loader (handles `model_state_dict`, plain dicts, and `module.` prefixes)
- Minimal resource usage: limited CPU threads, MKL-DNN enabled when present

## Folder Expectations
- Checkpoint: `checkpoints/checkpoint_best.pth` 
  - The app also falls back to `checkpoints/checkpoints/checkpoint_best.pth`
- Optional stats: `checkpoints/dataset_stats.json` for per-channel mean/std
- The repo must include `models.py` with `get_model`

## Run Locally
1) Install: