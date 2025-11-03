# Module imports
import os
import json
import logging
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
from PIL import Image
import gradio as gr
import torchvision.transforms as T
from pathlib import Path  # NEW

from models import get_model

# ----------------------------
# CPU optimization and logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hf_app_cpu")

# Tune threads for CPU
try:
    cpu_count = os.cpu_count() or 1
    torch.set_num_threads(max(1, min(4, cpu_count)))
    torch.set_num_interop_threads(max(1, min(4, cpu_count)))
except Exception as e:
    logger.warning(f"Thread tuning failed: {e}")

# Enable MKL-DNN for conv performance on CPU if available
try:
    torch.backends.mkldnn.enabled = True
except Exception as e:
    logger.warning(f"MKL-DNN enable failed: {e}")

# ----------------------------
# Paths and dataset stats
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent  # NEW

CANDIDATE_CHECKPOINTS = [
    str(BASE_DIR / "checkpoint_best.pth"),  # prefer local hf_app/
    os.path.join("checkpoints", "checkpoint_best.pth"),
    os.path.join("checkpoints", "checkpoints", "checkpoint_best.pth"),
]

CANDIDATE_STATS = [
    str(BASE_DIR / "dataset_stats.json"),  # prefer local hf_app/
    os.path.join("checkpoints", "dataset_stats.json"),
    os.path.join("checkpoints", "checkpoints", "dataset_stats.json"),
]

DEFAULT_IMAGENET_MEAN = [0.485, 0.456, 0.406]
DEFAULT_IMAGENET_STD = [0.229, 0.224, 0.225]

# ----------------------------
# Utility: load labels (optional)
# ----------------------------
# function load_imagenet_labels()
def load_imagenet_labels() -> Optional[List[str]]:
    """
    Attempt to load ImageNet class names. 
    Priority: local json → local text → remote fetch → None.
    """
    # Prefer local hf_app JSON mapping first
    local_json = str(BASE_DIR / "imagenet_class_index.json")
    if os.path.exists(local_json):
        try:
            with open(local_json, "r") as f:
                idx_map = json.load(f)
            labels = [idx_map[str(i)][1] for i in range(len(idx_map))]
            return labels
        except Exception as e:
            logger.warning(f"Failed reading {local_json}: {e}")
    # Fallback to local text
    local_txt = str(BASE_DIR / "imagenet_classes.txt")
    if os.path.exists(local_txt):
        try:
            with open(local_txt, "r") as f:
                labels = [line.strip() for line in f if line.strip()]
            return labels
        except Exception as e:
            logger.warning(f"Failed reading {local_txt}: {e}")
    # Remote fallback

    return None

# ----------------------------
# Utility: dataset mean/std
# ----------------------------
def load_dataset_mean_std() -> Tuple[List[float], List[float]]:
    for stats_path in CANDIDATE_STATS:
        if os.path.exists(stats_path):
            try:
                with open(stats_path, "r") as f:
                    stats = json.load(f)
                mean = stats.get("mean", DEFAULT_IMAGENET_MEAN)
                std = stats.get("std", DEFAULT_IMAGENET_STD)
                logger.info(f"Using dataset stats from {stats_path}: mean={mean}, std={std}")
                return mean, std
            except Exception as e:
                logger.warning(f"Failed to read stats {stats_path}: {e}")
    logger.info("Using default ImageNet mean/std")
    return DEFAULT_IMAGENET_MEAN, DEFAULT_IMAGENET_STD

# ----------------------------
# Model loading
# ----------------------------
def load_model_cpu() -> Tuple[Optional[nn.Module], str]:
    """
    Load the ResNet50 model and weights on CPU.
    Returns (model, status_message).
    """
    try:
        model = get_model(model_name="resnet50", dataset_name="imagenet1k", num_classes=1000, pretrained=False)
        model.eval()
        model.to("cpu")
    except Exception as e:
        msg = f"Model instantiation failed: {e}"
        logger.error(msg, exc_info=True)
        return None, msg

    ckpt_path = None
    for p in CANDIDATE_CHECKPOINTS:
        if os.path.exists(p):
            ckpt_path = p
            break
    if ckpt_path is None:
        msg = "checkpoint_best.pth not found in checkpoints/ or checkpoints/checkpoints/"
        logger.error(msg)
        return None, msg

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        msg = f"Failed to load checkpoint file: {e}"
        logger.error(msg, exc_info=True)
        return None, msg

    try:
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        # Strip possible 'module.' prefixes
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[new_key] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing or unexpected:
            logger.warning(f"State dict loaded with missing={missing} unexpected={unexpected}")

        msg = f"Model loaded from {ckpt_path}"
        logger.info(msg)
        return model, msg
    except Exception as e:
        msg = f"Error loading model state: {e}"
        logger.error(msg, exc_info=True)
        return None, msg

# ----------------------------
# Preprocessing and Postprocessing
# ----------------------------
_mean, _std = load_dataset_mean_std()

preprocess = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=_mean, std=_std),
])

labels = load_imagenet_labels()

def postprocess_logits(logits: torch.Tensor, topk: int = 5) -> Dict[str, float]:
    """Return top-k predictions as label -> probability dict."""
    probs = torch.softmax(logits, dim=1)
    top_probs, top_idxs = torch.topk(probs, k=topk, dim=1)
    top_probs = top_probs[0].tolist()
    top_idxs = top_idxs[0].tolist()

    result = {}
    for i, p in zip(top_idxs, top_probs):
        name = labels[i] if labels and i < len(labels) else f"class_{i}"
        result[name] = float(p)
    return result

# ----------------------------
# Inference function for Gradio
# ----------------------------
MODEL, LOAD_STATUS = load_model_cpu()

# function predict(image: Image.Image)
def predict(image: Image.Image) -> Tuple[Dict[str, float], str]:
    """
    Run inference on a PIL image.
    Returns (top-5 label->prob dict, status message).
    """
    if MODEL is None:
        return {"error": 1.0}, f"Model not loaded: {LOAD_STATUS}"

    try:
        if image is None:
            return {"error": 1.0}, "No image provided"

        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = preprocess(image).unsqueeze(0)

        # Test-time augmentation (horizontal flip) + temperature scaling
        with torch.inference_mode():
            logits_main = MODEL(tensor)
            flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
            tensor_flip = preprocess(flipped).unsqueeze(0)
            logits_flip = MODEL(tensor_flip)
            logits = (logits_main + logits_flip) / 2
            logits = logits / TEMPERATURE

        result = postprocess_logits(logits, topk=5)
        return result, "ok"
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return {"error": 1.0}, f"Inference failed: {e}"

# ----------------------------
# Gradio Interface
# ----------------------------
description = (
    "CPU-optimized ResNet50 ImageNet inference using the best checkpoint. "
    "Uses dataset mean/std if available; falls back to ImageNet defaults. "
    "Returns top-5 probabilities with robust error handling."
)

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Input Image"),
    outputs=[
        gr.Label(num_top_classes=5, label="Top-5 Predictions"),
        gr.Textbox(label="Status")
    ],
    title="ImageNet ResNet50 (CPU)",
    description=description,
    allow_flagging="never"
)

if __name__ == "__main__":
    TEMPERATURE = float(os.environ.get("IMAGENET_TEMPERATURE", "1.0"))
    iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))

# Top-level constants (add near other globals)
