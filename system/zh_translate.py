"""
中文指代表达 -> 英文（CLIP）离线翻译层（second.docx 方案）。
- 默认 Helsinki-NLP/opus-mt-zh-en，CPU 推理，不占 GPU。
- 本地快照按顺序查找（须含 config.json + pytorch_model.bin 或 model.safetensors）：
  1) ris_mvp/models/opus-mt-zh-en
  2) ris_mvp/train
  3) 与 ris_mvp 同级的 train/（例如 CUDA/test/train）
- 可通过环境变量 ZH_EN_MT_MODEL 指向任意完整快照目录（优先级最高）。
- 若上述「约定目录」存在但不完整，将报错提示补全权重，避免误走外网下载。
- 可在 CUSTOM_TRANSLATION 中覆盖固定短语（答辩演示更稳）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_RIS_ROOT = Path(__file__).resolve().parent.parent

# 候选目录（按优先级）；与 system/zh_translate.py 相对：ris_mvp 下及上一级工作区
_LOCAL_MT_CANDIDATES: List[Path] = [
    _RIS_ROOT / "models" / "opus-mt-zh-en",
    _RIS_ROOT / "train",
    _RIS_ROOT.parent / "train",
]


def _mt_snapshot_complete(d: Path) -> bool:
    if not d.is_dir() or not (d / "config.json").is_file():
        return False
    return (d / "pytorch_model.bin").is_file() or (d / "model.safetensors").is_file()


def first_complete_local_mt_dir() -> Optional[str]:
    """第一个 Marian 完整快照目录的绝对路径字符串。"""
    for d in _LOCAL_MT_CANDIDATES:
        if _mt_snapshot_complete(d):
            return str(d.resolve())
    return None


def first_incomplete_local_mt_dir() -> Optional[Path]:
    """存在 config 但缺权重的约定目录（用于阻止误用 Hub）。"""
    for d in _LOCAL_MT_CANDIDATES:
        if d.is_dir() and (d / "config.json").is_file() and not _mt_snapshot_complete(d):
            return d.resolve()
    return None


def resolve_mt_model_id() -> str:
    """ZH_EN_MT_MODEL > 约定本地目录（完整快照）> Hub 模型名。"""
    env = (os.environ.get("ZH_EN_MT_MODEL") or "").strip()
    if env:
        return env
    local = first_complete_local_mt_dir()
    if local:
        return local
    return "Helsinki-NLP/opus-mt-zh-en"


def describe_mt_resolution() -> str:
    """侧栏展示：当前会从哪里加载 Marian（不触发下载）。"""
    env = (os.environ.get("ZH_EN_MT_MODEL") or "").strip()
    if env:
        return f"Marian 来源：环境变量 ZH_EN_MT_MODEL → {env}"
    done = first_complete_local_mt_dir()
    if done:
        return f"Marian 来源：本地快照 → {done}"
    bad = first_incomplete_local_mt_dir()
    if bad:
        return (
            f"Marian：检测到不完整本地目录（缺 pytorch_model.bin / model.safetensors）→ {bad}。"
            " 补全权重后即可离线机翻；否则请勿使用非词典中文。"
        )
    return "Marian 来源：尚未配置完整本地目录，首次机翻将从镜像拉取 Helsinki-NLP/opus-mt-zh-en。"

# 文档 4.2 / one.docx 3.1.3：自定义映射优先；对左右、性别等加显式英文约束，减轻机翻歧义。
CUSTOM_TRANSLATION: Dict[str, str] = {
    "穿黑色衣服的人": "a person in black clothes",
    "戴眼镜的女人": "a woman wearing glasses",
    "左边的男人": "only the man on the LEFT side of the image, ignore the right and other people",
    "右边的女人": "only the woman on the RIGHT side of the image, ignore men and other people",
    "穿黑色衣服的男人": "the man wearing BLACK clothes, distinct from other people",
    "穿白色衣服的女人": "the woman wearing WHITE clothes, not the man",
    "中间的男人": "the man in the MIDDLE of the image, ignore people on the left and right",
    "右边的背包": "the backpack on the RIGHT side of the image, not the left",
    "黑色的背包": "the black backpack",
    "穿蓝色衬衫的女士": "the lady with the blue shirt",
    "站着的人": "person standing u",
}

_MT: Optional[Tuple[Any, Any]] = None


def warmup_mt() -> None:
    """预先加载 Marian（首次可能下载权重）；供 Streamlit 等界面启动时调用。"""
    _load_mt()


def is_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _load_mt():
    global _MT
    if _MT is not None:
        return _MT
    # 国内访问 huggingface.co 易超时；未手动设置时使用 hf-mirror（可用环境变量覆盖）。
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # 大文件经镜像仍可能较慢，适当延长单次下载超时（秒）。
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "中文翻译需要安装: pip install -r system/requirements-demo.txt"
        ) from e

    model_name = resolve_mt_model_id()
    mp = Path(model_name)
    if not mp.is_dir():
        incomplete = first_incomplete_local_mt_dir()
        if incomplete is not None:
            raise FileNotFoundError(
                "本地 Marian 目录不完整：缺少 pytorch_model.bin 或 model.safetensors。\n"
                "请将 Helsinki-NLP/opus-mt-zh-en 的权重放入该目录，或删除该目录后改用网络下载。\n"
                f"不完整路径: {incomplete}"
            )
    elif not _mt_snapshot_complete(mp):
        raise FileNotFoundError(
            "ZH_EN_MT_MODEL 指向的目录不完整：需要 config.json 与 pytorch_model.bin（或 model.safetensors）。\n"
            f"路径: {mp.resolve()}"
        )

    _local_only = mp.is_dir()
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=_local_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=_local_only)
    model.eval()
    model.to("cpu")
    _MT = (tok, model)
    return _MT


def translate_zh_to_en(text: str) -> str:
    """单句中译英；空串原样返回。"""
    text = (text or "").strip()
    if not text:
        return ""
    if text in CUSTOM_TRANSLATION:
        return CUSTOM_TRANSLATION[text]
    tok, model = _load_mt()
    import torch

    # CPU 上 beam 搜索很慢；默认 greedy（num_beams=1）显著减少「拖动阈值时卡顿」。
    # 需要更高质量可设环境变量 ZH_EN_MT_NUM_BEAMS=3
    num_beams = max(1, int(os.environ.get("ZH_EN_MT_NUM_BEAMS", "1")))
    inputs = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_length=64,
            num_beams=num_beams,
            early_stopping=num_beams > 1,
            do_sample=False,
        )
    return tok.decode(out[0], skip_special_tokens=True).strip()


def resolve_for_clip(raw: str, lang_mode: str) -> Tuple[str, Optional[str]]:
    """
    根据界面「语言模式」与是否含汉字，得到送入 CLIP 的英文及提示文案。
    lang_mode: 含「中文」则对含汉字输入强制走翻译；否则仅在检测到汉字时翻译。
    返回 (english_for_clip, note_or_none)
    """
    raw = (raw or "").strip()
    if not raw:
        return "", None
    if raw in CUSTOM_TRANSLATION:
        en = CUSTOM_TRANSLATION[raw]
        return en, f"词典映射: {raw} -> {en}"

    zh_priority = lang_mode and ("中文" in lang_mode)
    if zh_priority and is_chinese(raw):
        en = translate_zh_to_en(raw)
        return en, f"离线翻译: {raw} -> {en}"
    if (not zh_priority) and is_chinese(raw):
        en = translate_zh_to_en(raw)
        return en, f"检测到中文，已自动翻译: {raw} -> {en}"
    return raw, None
