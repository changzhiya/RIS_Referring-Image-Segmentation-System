"""
单图 + 文本推理：与训练/评估一致的 CLIP 归一化与模型构造。
与训练代码分离，仅依赖 ris_mvp 根目录下的 models/utils；权重默认在 result/ 下。

修改阈值、叠图颜色、默认权重路径等：编辑本文件即可；Gradio/Streamlit 在同级目录。
"""
from __future__ import annotations

import os
import sys
from collections import deque
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


def default_checkpoint_path() -> str:
    """在 result/**/best.pt 中自动选一个（优先名称含 v33、否则取最近修改）；无权重时返回空串。"""
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
    """根据 checkpoint 内 args 选择 baseline ClipTextGuidedRIS 或 one.docx 3.3 的 ClipRISV33。"""
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
    prob = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
    m = (prob >= float(threshold)).astype(np.uint8) * 255
    return prob, m


def _apply_logit_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """one.docx 推理增强：T<1 时放大 logits，使 sigmoid 更尖锐（不改网络权重）。"""
    t = float(temperature)
    if t >= 1.0 - 1e-9:
        return logits
    t = max(t, 1e-3)
    return logits / t


def _keep_largest_connected_component(mask_u8: np.ndarray) -> np.ndarray:
    """二值 mask 只保留最大 4-连通域，抑制小碎片与部分误激活（纯 numpy）。"""
    m = (mask_u8 >= 128).astype(np.uint8)
    if int(m.sum()) == 0:
        return mask_u8
    h, w = m.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    best: list[tuple[int, int]] = []
    best_n = 0
    for y in range(h):
        for x in range(w):
            if m[y, x] == 0 or visited[y, x]:
                continue
            q: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = 1
            comp: list[tuple[int, int]] = []
            while q:
                cy, cx = q.popleft()
                comp.append((cy, cx))
                for ny, nx in (cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1):
                    if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = 1
                        q.append((ny, nx))
            if len(comp) > best_n:
                best_n = len(comp)
                best = comp
    out = np.zeros((h, w), dtype=np.uint8)
    for cy, cx in best:
        out[cy, cx] = 255
    return out


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
    *,
    logit_temperature: Optional[float] = None,
    keep_largest_cc: Optional[bool] = None,
) -> Tuple[Image.Image, Image.Image, str]:
    """logit_temperature / keep_largest_cc：one.docx 推理阶段增强；未传则从环境变量读取。"""
    lt = (
        float(logit_temperature)
        if logit_temperature is not None
        else float(os.environ.get("RIS_LOGIT_TEMP", "1.0"))
    )
    klcc = (
        bool(keep_largest_cc)
        if keep_largest_cc is not None
        else (os.environ.get("RIS_KEEP_LARGEST_CC", "").strip().lower() in ("1", "true", "yes"))
    )

    logits = predict_mask_logits(model, device, image_size, pil_image, text)
    logits = _apply_logit_temperature(logits, lt)
    prob, mask_u8 = logits_to_mask_array(logits, threshold)
    if klcc:
        mask_u8 = _keep_largest_connected_component(mask_u8)
    overlay = overlay_mask(pil_image, mask_u8)
    info = (
        f"image_size={image_size}, thr={threshold:.2f}, logit_temp={lt:.2f}, "
        f"largest_cc={int(klcc)}, fg_ratio={(mask_u8 > 0).mean():.4f}"
    )
    return overlay, mask_to_pil(mask_u8), info
