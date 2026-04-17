"""
文本引导图像语义分割 — Gradio 最小 Demo（位于 system/，与训练代码分离）。

在项目根目录执行:
  pip install -r system/requirements-demo.txt
  python system/gradio_app.py

权重默认读取 ../result/.../best.pt；可在界面填写任意 .pt 路径。
答辩向界面优化（the_one.docx）已整合在 `streamlit_app.py`，建议答辩用 Streamlit。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_SYSTEM_DIR = Path(__file__).resolve().parent
_RIS_ROOT = _SYSTEM_DIR.parent
_DEFAULT_ZH_MT_DIR = _RIS_ROOT.parent / "train"
if _DEFAULT_ZH_MT_DIR.is_dir() and (
    (_DEFAULT_ZH_MT_DIR / "pytorch_model.bin").is_file()
    or (_DEFAULT_ZH_MT_DIR / "model.safetensors").is_file()
):
    os.environ.setdefault("ZH_EN_MT_MODEL", str(_DEFAULT_ZH_MT_DIR.resolve()))

if str(_RIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_RIS_ROOT))
if str(_SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(_SYSTEM_DIR))

try:
    import gradio as gr
except ImportError as e:
    raise SystemExit("请先安装: pip install -r system/requirements-demo.txt") from e

import predict  # noqa: E402
import zh_translate  # noqa: E402

_bundle_cache: dict[str, tuple] = {}


def _get_bundle(ckpt_path: str):
    path = os.path.abspath(ckpt_path.strip() or predict.default_checkpoint_path())
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到权重: {path}")
    if path not in _bundle_cache:
        model, device, image_size, meta = predict.load_model_bundle(path)
        _bundle_cache[path] = (model, device, image_size, meta)
    return _bundle_cache[path]


def on_run(
    pil_image,
    text: str,
    threshold: float,
    ckpt_path: str,
    lang_mode: str,
    logit_temp: float,
    keep_largest_cc: bool,
):
    if pil_image is None:
        return None, None, "请先上传图像。"
    ckpt = ckpt_path.strip() or predict.default_checkpoint_path()
    raw = (text or "").strip()
    try:
        en, note = zh_translate.resolve_for_clip(raw, lang_mode or "中文（离线翻译）")
    except Exception as e:
        return None, None, f"翻译失败（需 pip install transformers sentencepiece）: {e}"
    try:
        model, device, image_size, meta = _get_bundle(ckpt)
    except Exception as e:
        return None, None, f"加载模型失败: {e}"
    try:
        overlay, mask, info = predict.run_segmentation(
            model,
            device,
            image_size,
            pil_image,
            en,
            float(threshold),
            logit_temperature=float(logit_temp),
            keep_largest_cc=bool(keep_largest_cc),
        )
        head = f"ckpt={os.path.basename(ckpt)} | ep={meta.get('epoch')} | best_mIoU={meta.get('best_val_miou')}"
        lines = [head, info]
        if note:
            lines.append(note)
        if en != raw:
            lines.append(f"CLIP文本(英): {en}")
        return overlay, mask, "\n".join(lines)
    except Exception as e:
        return None, None, f"推理失败: {e}"


def on_reload_ckpt(ckpt_path: str):
    path = os.path.abspath(ckpt_path.strip() or predict.default_checkpoint_path())
    _bundle_cache.pop(path, None)
    try:
        _get_bundle(path)
        return f"已重新加载: {path}"
    except Exception as e:
        return f"加载失败: {e}"


def build_ui():
    default_ckpt = predict.default_checkpoint_path()
    with gr.Blocks(title="RIS Demo") as demo:
        gr.Markdown(
            "## 文本引导图像语义分割（最小 Demo）\n"
            "系统代码位于 `system/`；模型与训练在仓库根目录 `train.py`、`models/`、`result/`。"
        )
        with gr.Row():
            ckpt = gr.Textbox(
                value=default_ckpt,
                label="Checkpoint（best.pt / last.pt）",
                lines=1,
            )
            reload_btn = gr.Button("重新加载权重", variant="secondary")
        reload_msg = gr.Textbox(label="加载状态", lines=1, interactive=False)
        with gr.Row():
            img_in = gr.Image(type="pil", label="输入图像", sources=["upload", "clipboard"])
            text_in = gr.Textbox(
                value="the object on the left",
                label="指代表达（中文模式含汉字时离线译英再送 CLIP）",
                lines=2,
            )
        lang = gr.Radio(
            ["英文（CLIP原生）", "中文（离线翻译）"],
            value="中文（离线翻译）",
            label="指代表达语言",
        )
        thr = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="前景阈值（sigmoid）")
        logit_t = gr.Slider(
            0.45,
            1.0,
            value=1.0,
            step=0.05,
            label="Logit 锐化温度（one.docx，<1 更尖锐）",
        )
        klcc = gr.Checkbox(label="仅保留最大连通域", value=False)
        run_btn = gr.Button("运行分割", variant="primary")
        with gr.Row():
            out_overlay = gr.Image(type="pil", label="叠加预览")
            out_mask = gr.Image(type="pil", label="二值掩码")
        log = gr.Textbox(label="信息", lines=3, interactive=False)

        run_btn.click(
            on_run,
            [img_in, text_in, thr, ckpt, lang, logit_t, klcc],
            [out_overlay, out_mask, log],
        )
        reload_btn.click(on_reload_ckpt, [ckpt], [reload_msg])
    return demo


if __name__ == "__main__":
    import os

    demo = build_ui()
    demo.queue()
    # 7860 被占用时传 None 由 Gradio 自找空闲端口；也可用环境变量 GRADIO_SERVER_PORT=7861
    port = os.environ.get("GRADIO_SERVER_PORT")
    server_port = int(port) if port else None
    demo.launch(server_name="127.0.0.1", server_port=server_port, share=False)
