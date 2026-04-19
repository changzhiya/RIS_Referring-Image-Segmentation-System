"""
推理入口：单图 + 文本 → 分割掩码。

- 与 train/eval 共用 CLIP 图像归一化与同一套模型构造逻辑（见 load_model_bundle）。
- 依赖项目根下 models/、utils/；Streamlit 通过本模块加载权重并调用 run_segmentation。
- 默认权重：下方 _DEFAULT_CKPT_FIXED 或 result/ 下自动搜索，可按部署修改。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import clip
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "缺少 OpenAI CLIP Python 包（`import clip`）。在项目根目录执行:\n"
        "  pip install openai-clip\n"
        "或: pip install -r system/requirements-demo.txt\n"
        "（需已安装 torch / torchvision。）"
    ) from e

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# system/ 的上一级为项目根（含 models、train.py、result/）
_SYSTEM_DIR = Path(__file__).resolve().parent
_RIS_ROOT = _SYSTEM_DIR.parent
if str(_RIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_RIS_ROOT))

import torch.nn as nn  # noqa: E402

from models.clip_ris import ClipTextGuidedRIS  # noqa: E402
from utils.clip_finetune import unfreeze_clip_text_last_blocks  # noqa: E402

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# 部署默认权重（存在则优先；否则 result/checkpoint_v2、再否则 result 下自动搜索）
_DEFAULT_CKPT_FIXED = Path(
    r"D:\NVIDIA GPU Computing Toolkit\CUDA\test\ris_mvp\RIS_Referring-Image-Segmentation-System\result\checkpoint_v2\best.pt"
)


def default_checkpoint_path() -> str:
    """默认权重路径：固定部署路径 → 项目根 result/checkpoint_v2 → result/**/best.pt（优先 v33）。"""
    if _DEFAULT_CKPT_FIXED.is_file():
        return str(_DEFAULT_CKPT_FIXED.resolve())
    rel_v2 = _RIS_ROOT / "result" / "checkpoint_v2" / "best.pt"
    if rel_v2.is_file():
        return str(rel_v2.resolve())
    result = _RIS_ROOT / "result"
    if not result.is_dir():
        return ""
    paths = list(result.glob("**/best.pt"))
    if not paths:
        return ""
    v33 = [p for p in paths if "v33" in p.as_posix().lower()]
    pool = v33 if v33 else paths
    best = max(pool, key=lambda p: p.stat().st_mtime)
    return str(best.resolve())


def _image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ]
    )


def _build_ris_from_ckpt_args(clip_model, args_dict: Dict[str, Any], device: str) -> nn.Module:
    """据 ckpt['args']['ris_arch'] 构建 ClipTextGuidedRIS（baseline）或 ClipRISV33（v33）。"""
    arch = str(args_dict.get("ris_arch", "baseline")).lower()
    unfreeze_last = int(args_dict.get("clip_unfreeze_last", 0))
    clip_trainable = unfreeze_last > 0
    if arch in ("v33", "33", "doc33"):
        from models.clip_ris_v33 import ClipRISV33  # noqa: WPS433

        return ClipRISV33(
            clip_model=clip_model,
            clip_text_trainable=clip_trainable,
        ).to(device)
    return ClipTextGuidedRIS(
        clip_model=clip_model,
        clip_text_trainable=clip_trainable,
    ).to(device)


def load_model_bundle(
    checkpoint_path: str,
    device: Optional[str] = None,
) -> Tuple[nn.Module, str, int, Dict[str, Any]]:
    """加载权重与 CLIP，返回 (model, device, image_size, ckpt_meta)。"""
    path = (checkpoint_path or "").strip()
    if not path:
        raise FileNotFoundError("未指定 checkpoint 路径（请填写 best.pt / last.pt）")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到权重文件: {path}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args_dict: Dict[str, Any] = dict(ckpt.get("args") or {})
    clip_name = str(args_dict.get("clip_model", "ViT-B/32"))
    unfreeze_last = int(args_dict.get("clip_unfreeze_last", 0))
    image_size = int(args_dict.get("image_size", 256))
    ris_arch = str(args_dict.get("ris_arch", "baseline")).lower()

    clip_model, _ = clip.load(clip_name, device=device)
    clip_model = clip_model.float()
    if unfreeze_last > 0:
        unfreeze_clip_text_last_blocks(clip_model, unfreeze_last)
    else:
        for p in clip_model.parameters():
            p.requires_grad = False

    model = _build_ris_from_ckpt_args(clip_model, args_dict, device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    meta = {
        "checkpoint": os.path.abspath(path),
        "epoch": ckpt.get("epoch"),
        "best_val_miou": ckpt.get("best_val_miou", ckpt.get("best_val_iou")),
        "image_size": image_size,
        "clip_model": clip_name,
        "clip_unfreeze_last": unfreeze_last,
        "ris_arch": ris_arch,
    }
    return model, device, image_size, meta


@torch.no_grad()
def predict_mask_logits(
    model: nn.Module,
    device: str,
    image_size: int,
    pil_image: Image.Image,
    text: str,
) -> torch.Tensor:
    """返回 logits [1,1,H,W]，H=W=image_size。"""
    tfm = _image_transform(image_size)
    x = tfm(pil_image.convert("RGB")).unsqueeze(0).to(device)
    tokens = clip.tokenize([text.strip() or " "], truncate=True).to(device)
    return model(x, tokens)


def logits_to_mask_array(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """sigmoid 后按阈值二值化；返回 (概率图 HW, uint8 掩码 0/255)。"""
    prob = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
    m = (prob >= float(threshold)).astype(np.uint8) * 255
    return prob, m


def overlay_mask(
    pil_rgb: Image.Image,
    mask_hw: np.ndarray,
    color: Tuple[int, int, int] = (255, 64, 64),
    alpha: float = 0.42,
) -> Image.Image:
    h, w = mask_hw.shape[:2]
    base = pil_rgb.convert("RGB").resize((w, h), Image.BILINEAR)
    arr = np.asarray(base, dtype=np.float32)
    m = (mask_hw.astype(np.float32) / 255.0)[..., None]
    c = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out = arr * (1.0 - alpha * m) + c * (alpha * m)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def mask_to_pil(mask_uint8: np.ndarray) -> Image.Image:
    return Image.fromarray(mask_uint8, mode="L")


def run_segmentation(
    model: nn.Module,
    device: str,
    image_size: int,
    pil_image: Image.Image,
    text: str,
    threshold: float = 0.5,
) -> Tuple[Image.Image, Image.Image, str]:
    """单尺度推理：sigmoid 阈值化得到掩码与叠图。"""
    logits = predict_mask_logits(model, device, image_size, pil_image, text)
    _prob, mask_u8 = logits_to_mask_array(logits, threshold)
    overlay = overlay_mask(pil_image, mask_u8)
    info = f"image_size={image_size}, thr={threshold:.2f}, fg_ratio={(mask_u8 > 0).mean():.4f}"
    return overlay, mask_to_pil(mask_u8), info
