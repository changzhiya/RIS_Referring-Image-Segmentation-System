"""
将 Helsinki-NLP/opus-mt-zh-en 下载到项目内 models/opus-mt-zh-en（仅 PyTorch + 分词所需文件）。
在 ris_mvp 根目录执行:
  conda activate ris_env
  python scripts/download_opus_mt_zh_en.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_RIS_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_DIR = _RIS_ROOT / "models" / "opus-mt-zh-en"


def main() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit("请先安装: pip install huggingface_hub") from e

    _LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to: {_LOCAL_DIR}")
    snapshot_download(
        repo_id="Helsinki-NLP/opus-mt-zh-en",
        local_dir=str(_LOCAL_DIR),
        local_dir_use_symlinks=False,
        ignore_patterns=["tf_model.h5", "rust_model.ot", "*.h5", "*.ot"],
    )
    print("Done. Set ZH_EN_MT_MODEL or rely on zh_translate default local path.")


if __name__ == "__main__":
    main()
