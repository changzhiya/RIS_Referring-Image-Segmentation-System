"""
文本引导指代图像分割 — Streamlit 交互界面（含中文指代与离线翻译）。

在项目根目录执行:
  pip install -r system/requirements-demo.txt
  streamlit run system/streamlit_app.py

可修改处: 下方 CSS、示例图路径、系统说明文案、默认阈值等。
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

from PIL import Image

_SYSTEM_DIR = Path(__file__).resolve().parent
_RIS_ROOT = _SYSTEM_DIR.parent
# 与项目根同级的 train/（例如 .../CUDA/test/train/pytorch_model.bin）；未设置 ZH_EN_MT_MODEL 时默认使用该目录。
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
    import streamlit as st
except ImportError as e:
    raise SystemExit("请先安装: pip install -r system/requirements-demo.txt") from e

import predict  # noqa: E402
import zh_translate  # noqa: E402


def _pil_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _load_demo_presets() -> List[Tuple[str, Path, str, str]]:
    """(标签, 路径, 中文示例, 英文示例)。优先内置 demo_refcoco；有完整 refcoco_ready 时用示例图。"""
    demo_img = _RIS_ROOT / "demo_refcoco" / "images" / "sample.jpg"
    root = _RIS_ROOT / "refcoco_ready"
    cands: List[Tuple[str, Path, str, str]] = [
        (
            "内置示例(demo_refcoco)",
            demo_img,
            "画面中心的物体",
            "the object in the center",
        ),
        (
            "示例1-581857",
            root / "images" / "COCO_train2014_000000581857.jpg",
            "穿蓝色衬衫的女士",
            "the lady with the blue shirt",
        ),
        (
            "示例2-581839",
            root / "images" / "COCO_train2014_000000581839.jpg",
            "站着的人",
            "person standing u",
        ),
    ]
    return [(a, p, z, e) for a, p, z, e in cands if p.is_file()]


@st.cache_resource
def cached_bundle(checkpoint_path: str):
    path = os.path.abspath(checkpoint_path.strip())
    return predict.load_model_bundle(path)


@st.cache_resource(show_spinner="正在准备中译英模型（进程内仅一次，首次可能数十秒）...")
def _ensure_zh_mt_cached() -> int:
    """与 zh_translate 共用全局权重；避免每次拖动阈值都重复 from_pretrained。"""
    zh_translate.warmup_mt()
    return 1


def _run_infer(*, thr_only: bool = False) -> None:
    """使用当前 session_state 的 ckpt_path / work_pil / expr / thr / lang_mode；中文经 zh_translate 再送 CLIP。

    thr_only=True（阈值实时刷新）：若指代表达未改，复用上次的英文译文，避免每帧都跑 CPU 翻译。
    """
    ckpt = str(st.session_state.get("ckpt_path") or predict.default_checkpoint_path())
    pil = st.session_state.get("work_pil")
    raw = (st.session_state.get("expr") or "").strip()
    if pil is None or not raw:
        return
    thr = float(st.session_state.get("thr", 0.55))
    lang = str(st.session_state.get("lang_mode") or "中文（离线翻译）")
    # 阈值实时刷新时复用同一句话的英译，避免重复跑 CPU 翻译
    try:
        reuse = (
            thr_only
            and st.session_state.get("_last_infer_raw") == raw
            and st.session_state.get("_last_infer_en")
        )
        if reuse:
            en = str(st.session_state.get("_last_infer_en"))
            note = st.session_state.get("_last_infer_note")
        else:
            en, note = zh_translate.resolve_for_clip(raw, lang)
    except Exception as e:
        st.session_state["clip_note"] = f"翻译失败（可检查是否已 pip install transformers）：{e}"
        raise
    st.session_state["clip_note"] = note
    model, device, image_size, meta = cached_bundle(ckpt)
    t0 = time.perf_counter()
    overlay, mask, info = predict.run_segmentation(
        model,
        device,
        image_size,
        pil,
        en,
        thr,
    )
    st.session_state["overlay"] = overlay
    st.session_state["mask"] = mask
    st.session_state["infer_ms"] = (time.perf_counter() - t0) * 1000
    st.session_state["info"] = info + (f" | CLIP文本(英): {en}" if en != raw else "")
    st.session_state["meta"] = meta
    st.session_state["_last_infer_raw"] = raw
    st.session_state["_last_infer_en"] = en
    st.session_state["_last_infer_note"] = note


def _on_thr_change() -> None:
    if st.session_state.get("live_thr", True) and st.session_state.get("work_pil") is not None:
        try:
            _run_infer(thr_only=True)
        except Exception:
            pass


def _apply_demo_preset(index: int) -> None:
    """在按钮 on_click 中执行：先于侧栏 text_input 绑定，可安全写入 session_state.expr。"""
    presets = _load_demo_presets()
    if len(presets) < 2:
        st.session_state["_demo_flash"] = ("error", "未找到 demo_refcoco 或 refcoco_ready 示例图")
        return
    _, p, zh, en = presets[index]
    st.session_state.work_pil = Image.open(p).convert("RGB")
    use_zh = "中文" in str(st.session_state.get("lang_mode", ""))
    st.session_state.expr = zh if use_zh else en
    try:
        _run_infer()
    except Exception as e:
        st.session_state["_demo_flash"] = ("error", str(e))
    else:
        st.session_state["_demo_flash"] = ("ok", f"示例{index + 1} 完成")


def _warm_zh_mt_click() -> None:
    """按需预加载 Marian；不在页面打开时自动执行，避免长时间卡在首屏。"""
    try:
        _ensure_zh_mt_cached()
    except Exception as e:
        st.session_state["_warm_mt_flash"] = ("error", str(e))
    else:
        st.session_state["_warm_mt_flash"] = ("ok", "中译英模型已就绪（本进程内已缓存，之后无需再点）。")


def _apply_full_demo() -> None:
    presets = _load_demo_presets()
    if len(presets) < 2:
        st.session_state["_demo_flash"] = ("error", "缺少示例数据，请先准备 refcoco_ready。")
        return
    use_zh = "中文" in str(st.session_state.get("lang_mode", ""))
    try:
        for _, p, zh, en in presets[:2]:
            st.session_state.work_pil = Image.open(p).convert("RGB")
            st.session_state.expr = zh if use_zh else en
            _run_infer()
            time.sleep(0.15)
    except Exception as e:
        st.session_state["_demo_flash"] = ("error", str(e))
    else:
        st.session_state["_demo_flash"] = ("ok", "完整演示完成（已切换至示例2结果）")


def _inject_ui_styles() -> None:
    hide = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stToolbar"] {visibility: hidden;}
    </style>
    """
    st.markdown(hide, unsafe_allow_html=True)
    custom = """
    <style>
    html { font-size: 16px; }
    h1, h2, h3 { color: #1E293B !important; font-weight: 600 !important; }
    div[data-testid="stSidebar"] { background-color: #F8FAFC; padding-top: 0.5rem; }
    div[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
    div[data-testid="stPrimaryButton"] button {
        background-color: #165DFF !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: none !important;
    }
    div[data-testid="stPrimaryButton"] button:hover {
        background-color: #0E42D2 !important;
        box-shadow: 0 4px 12px rgba(22, 93, 255, 0.25);
    }
    hr { border-color: #E2E8F0; margin: 1rem 0; }
    </style>
    """
    st.markdown(custom, unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="文本引导指代图像分割系统",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_ui_styles()

    st.session_state.setdefault("thr", 0.55)
    st.session_state.setdefault("expr", "the lady with the blue shirt")
    _def_ckpt = predict.default_checkpoint_path()
    if _def_ckpt and os.path.isfile(_def_ckpt):
        st.session_state.setdefault("ckpt_path", _def_ckpt)
    else:
        st.session_state.setdefault("ckpt_path", "")
    if "work_pil" not in st.session_state:
        st.session_state.work_pil = None
    if "overlay" not in st.session_state:
        st.session_state.overlay = None
    if "mask" not in st.session_state:
        st.session_state.mask = None
    if "live_thr" not in st.session_state:
        st.session_state.live_thr = True
    st.session_state.setdefault("lang_mode", "中文（离线翻译）")

    st.title("文本引导指代图像分割系统")
    _ckpt_warn = str(st.session_state.get("ckpt_path") or "").strip()
    if not _ckpt_warn or not os.path.isfile(_ckpt_warn):
        st.warning(
            "未找到有效权重文件。请在侧栏「模型权重路径」填写 `best.pt` / `last.pt`，"
            "或在 `system/predict.py` 中配置 `default_checkpoint_path()` 后重开页面。"
        )
    st.divider()

    # —— 左侧控制面板 ——
    with st.sidebar:
        st.header("控制面板")

        st.subheader("1. 模型加载")
        st.text_input(
            "模型权重路径",
            help="best.pt / last.pt，须与训练时结构一致",
            key="ckpt_path",
        )

        c_clear, c_reload = st.columns(2)
        with c_clear:
            if st.button("清除模型缓存", width="stretch"):
                cached_bundle.clear()
                st.success("已清除缓存")
        with c_reload:
            if st.button("重新加载权重", type="primary", width="stretch"):
                with st.spinner("正在加载模型权重..."):
                    try:
                        cached_bundle.clear()
                        cached_bundle(str(st.session_state.ckpt_path))
                    except Exception as e:
                        st.error(f"加载失败: {e}")
                    else:
                        st.success("模型加载成功")
        st.divider()

        st.subheader("2. 输入设置")
        up = st.file_uploader(
            "上传图像",
            type=["jpg", "jpeg", "png", "webp"],
            help="支持常见格式；与训练分辨率无关，推理时会按 checkpoint 内 image_size 缩放",
        )
        if up is not None:
            st.session_state.work_pil = Image.open(io.BytesIO(up.getvalue())).convert("RGB")

        st.selectbox(
            "指代表达语言",
            ["英文（CLIP原生）", "中文（离线翻译）"],
            index=1,
            key="lang_mode",
            help="中文模式：非词典中文会走 opus-mt-zh-en（CPU）。模型不在打开页面时自动下载，可侧栏先「预加载」或在首次分割时等待下载完成。",
        )
        lm = st.session_state.get("lang_mode", "")
        if "中文" in str(lm):
            st.caption("可输入中文或英文；含汉字时将离线翻译后再送 CLIP。")
            st.caption(zh_translate.describe_mt_resolution())
            st.button(
                "预加载中译英模型（可选，首次约数百 MB）",
                width="stretch",
                help="从镜像下载 Marian 权重到本机缓存；网络慢时请耐心等待或改用环境变量 ZH_EN_MT_MODEL 指向已下载目录。",
                on_click=_warm_zh_mt_click,
            )
            warm_flash = st.session_state.pop("_warm_mt_flash", None)
            if warm_flash:
                k, msg = warm_flash
                (st.error if k == "error" else st.success)(msg)
        else:
            st.caption("请输入英文指代表达（与 CLIP 训练分布一致）。")

        st.text_input(
            "指代表达",
            placeholder="中文例：左边的男人 / 英文例：the man on the left",
            help="中文：词典优先，其次 Helsinki-NLP 中译英再送 CLIP",
            key="expr",
        )

        st.divider()
        st.subheader("3. 分割参数")
        st.session_state.live_thr = st.checkbox(
            "阈值拖动时实时刷新",
            value=st.session_state.live_thr,
            help="开启后拖动阈值会反复推理。中文指代会走 CPU 翻译：已缓存「同一句话」的译文，仅阈值变化时不再重复翻译；若仍卡顿可关闭此项。",
        )
        st.slider(
            "分割阈值（Sigmoid）",
            min_value=0.05,
            max_value=0.95,
            step=0.05,
            key="thr",
            help="越高越严格（漏检少、误检可能增）；越低越宽松",
            on_change=_on_thr_change if st.session_state.live_thr else None,
        )

        st.divider()
        st.subheader("快速演示")
        presets = _load_demo_presets()
        if len(presets) >= 2:
            dc1, dc2 = st.columns(2)
            with dc1:
                st.button("示例1", width="stretch", on_click=_apply_demo_preset, args=(0,))
            with dc2:
                st.button("示例2", width="stretch", on_click=_apply_demo_preset, args=(1,))
        else:
            st.caption("未找到 demo_refcoco / refcoco_ready 示例图；请上传本地图片演示。")

        st.button(
            "一键完整演示",
            type="secondary",
            width="stretch",
            on_click=_apply_full_demo,
            disabled=len(presets) < 2,
        )

        st.divider()
        run_btn = st.button(
            "运行分割",
            type="primary",
            width="stretch",
        )

        flash = st.session_state.pop("_demo_flash", None)
        if flash:
            kind, msg = flash
            if kind == "error":
                st.error(msg)
            else:
                st.success(msg)

    # —— 主区：预加载 + 三列结果 ——
    with st.spinner("系统初始化：加载模型..."):
        try:
            model, device, image_size, meta = cached_bundle(str(st.session_state.ckpt_path))
        except Exception as e:
            st.error(f"无法加载模型: {e}")
            st.stop()

    if run_btn:
        if st.session_state.work_pil is None:
            st.sidebar.error("请先上传图像或使用示例。")
        elif not (st.session_state.expr or "").strip():
            st.sidebar.error("请输入指代表达。")
        else:
            with st.spinner("正在进行图像分割..."):
                try:
                    _run_infer()
                except Exception as e:
                    st.error(f"推理失败: {e}")
                else:
                    st.success("分割完成")

    st.header("分割结果")
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        st.markdown("### 原图")
        st.caption("输入原始图像")
        if st.session_state.work_pil is not None:
            st.image(st.session_state.work_pil, width="stretch", caption="原始图像")
        else:
            st.info("请上传图像或点侧边栏「示例」。")
    with col2:
        st.markdown("### 分割叠加图")
        st.caption("红色区域为模型预测前景")
        if st.session_state.overlay is not None:
            st.image(st.session_state.overlay, width="stretch", caption="叠加效果")
    with col3:
        st.markdown("### 分割掩码")
        st.caption("白=前景，黑=背景")
        if st.session_state.mask is not None:
            st.image(st.session_state.mask, width="stretch", caption="二值掩码")

    if st.session_state.overlay is not None and st.session_state.mask is not None:
        st.divider()
        st.subheader("性能指标")
        m = st.session_state.get("meta") or {}
        ms = st.session_state.get("infer_ms")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("单次推理耗时", f"{ms:.0f} ms" if ms is not None else "-")
        with c2:
            miou = m.get("best_val_miou")
            st.metric("checkpoint 记录 best mIoU", f"{float(miou):.4f}" if miou is not None else "-")
        with c3:
            st.metric("当前阈值", f"{float(st.session_state.thr):.2f}")
        if st.session_state.get("info"):
            st.caption(st.session_state["info"])
        if st.session_state.get("clip_note"):
            st.info(st.session_state["clip_note"])

        st.divider()
        st.subheader("结果下载")
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                label="下载分割叠加图",
                data=_pil_png_bytes(st.session_state.overlay),
                file_name="segmentation_overlay.png",
                mime="image/png",
                width="stretch",
            )
        with d2:
            st.download_button(
                label="下载分割掩码",
                data=_pil_png_bytes(st.session_state.mask),
                file_name="segmentation_mask.png",
                mime="image/png",
                width="stretch",
            )

    st.divider()
    with st.expander("系统说明", expanded=False):
        st.write(
            """
本系统为文本引导指代图像分割（RIS）：CLIP 文本编码与可配置分割头，推理与训练共用同一归一化与权重格式。
默认权重路径由 `system/predict.py` 中配置决定（可在侧栏修改 `best.pt` / `last.pt` 路径）。
验证指标以训练生成的 `val_metrics.jsonl` 为准；界面「best mIoU」为保存 checkpoint 时的记录值。

中文指代：在「中文（离线翻译）」模式下，含汉字的句子会先经词典 / Helsinki-NLP opus-mt-zh-en
译为英文再送入 CLIP，翻译在 CPU 运行。首次使用需 `pip install -r system/requirements-demo.txt` 并联网下载模型。
完全离线部署：若存在与「项目根」同级的 `train/` 且内含 `pytorch_model.bin`，应用启动时会自动
`setdefault(ZH_EN_MT_MODEL, …)` 指向该目录。其它路径约定见 `zh_translate` 模块说明。
拉取翻译模型时默认使用国内镜像 HF_ENDPOINT=https://hf-mirror.com（若已自行设置 HF_ENDPOINT 则不会覆盖）。
打开页面时不再自动下载 Marian；需要中文机翻时请侧栏点「预加载中译英模型」，或在首次「运行分割」时等待下载完成（默认单次下载超时 300 秒，见 zh_translate）。

卡顿说明：Marian 在 CPU 上比英文 tokenize 慢得多；若开启「阈值拖动时实时刷新」，同一句话下仅会翻译一次，后续拖动复用译文。
默认贪心解码（较快）；要更高翻译质量可设环境变量 ZH_EN_MT_NUM_BEAMS=3（会更慢）。

自定义短语映射可在 `zh_translate.CUSTOM_TRANSLATION` 中维护；网络结构见 `models/clip_ris.py`、`models/clip_ris_v33.py`。
            """.strip()
        )


if __name__ == "__main__":
    main()
