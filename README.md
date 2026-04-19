# 文本引导指代图像分割系统（RIS）

基于 **PyTorch** 与 **OpenAI CLIP** 的指代表达图像分割（Referring Image Segmentation）工程：支持 **RefCOCO 索引数据**、**训练 / 评估 / 可视化**，并提供 **Streamlit** 交互推理界面。

---

## 系统与模型

| 组件 | 说明 |
|------|------|
| **CLIP** | 文本编码；训练时可按 `--clip-unfreeze-last` 解冻文本 Transformer 末若干层。 |
| **分割头（二选一）** | **`--ris-arch baseline`**：`ClipTextGuidedRIS`，CLIP 句向量 + 轻量图像编码 + 解码。**`--ris-arch v33`**：`ClipRISV33`（文档 §3.3）：空间位置残差、**token 级**文本特征、像素–token 跨模态注意力 + 解码；与 baseline **权重不通用**，须单独训练。 |
| **权重** | 训练写入 `--save-dir`（默认在 `result/` 下子目录）。`ckpt["args"]` 含 `ris_arch`、`image_size` 等；`eval.py` / `visualize.py` / `system/predict.py` 按 checkpoint 自动构造对应网络。 |
| **推理** | `python -m streamlit run system/streamlit_app.py`。默认权重路径在 `system/predict.py` 的 `default_checkpoint_path()` 中配置（可按部署修改）。 |

---

## 目录结构（摘要）

```text
├── train.py / eval.py / visualize.py   # 训练、评估、批量可视化
├── models/                              # clip_ris.py（baseline）、clip_ris_v33.py（§3.3）
├── data/                                # RefCOCO 数据集加载
├── utils/                               # 指标、CLIP 解冻等
├── system/                              # Streamlit、推理 predict、中文翻译
├── tools/                               # 索引构建、Word 报告等
├── demo_refcoco/                        # 极小样本，用于环境自检
├── refcoco_ready/                       # 通常仅提交 splits/*.json；images、masks 见下文
└── result/                              # 训练产出（默认 .gitignore，勿提交大 .pt）
```

---

## 环境

```powershell
conda activate ris_env   # 或你的 Python 3.10+ 环境

pip install torch torchvision tqdm pillow pycocotools numpy
pip install git+https://github.com/openai/CLIP.git
pip install -r system/requirements-demo.txt   # Streamlit、翻译等
```

---

## 数据

- **格式**：`splits/train.json` 等，每条含 `image`、`mask`（相对 `data-root`）、`text`。示例见下文 JSON。
- **本仓库**：可包含 `refcoco_ready/splits/*.json`；**大图与掩码**体积大，请本地放入 `refcoco_ready/images/`、`refcoco_ready/masks/`（说明见 [refcoco_ready/README_DATA.md](refcoco_ready/README_DATA.md)）。
- **自检**：`python tools/verify_refcoco_setup.py`；小样本可用 `python tools/verify_refcoco_setup.py --data-root demo_refcoco`。

---

## 训练

默认 **256×256**、`batch_size=2`、`grad_accum_steps=8` 等见 `train.py --help`。

**基线示例：**

```powershell
python train.py `
  --data-root ".\refcoco_ready" `
  --train-index ".\refcoco_ready\splits\train.json" `
  --val-index ".\refcoco_ready\splits\val.json" `
  --epochs 10
```

**§3.3 结构（须单独训练，与旧 .pt 不兼容）：**

```powershell
python train.py `
  --data-root ".\refcoco_ready" `
  --train-index ".\refcoco_ready\splits\train.json" `
  --val-index ".\refcoco_ready\splits\val.json" `
  --ris-arch v33 `
  --epochs 15
```

- **续训**：`--resume path\to\last.pt`，并显式指定 **`--save-dir`** 与首次训练一致。
- **指标**：每轮验证 **mIoU**、**cIoU** 写入 `<save-dir>/val_metrics.jsonl`；验证 mIoU 提升时更新 **`best.pt`**，每轮结束写 **`last.pt`**。

---

## 评估与可视化

```powershell
python eval.py `
  --data-root ".\refcoco_ready" `
  --checkpoint ".\result\checkpoint_v2\best.pt" `
  --split val=splits/val.json `
  --save-json ".\reports\eval_results.json"
```

```powershell
pip install python-docx   # 若需 Word 报告

python visualize.py `
  --data-root ".\refcoco_ready" `
  --index "splits\testA.json" `
  --checkpoint ".\result\checkpoint_v2\best.pt" `
  --out-dir ".\reports\vis_testA" `
  --num-samples 16

python tools/build_report_docx.py `
  --eval-json ".\reports\eval_results.json" `
  --vis-dir ".\reports\vis_testA" `
  --out-docx ".\reports\RIS实验成果说明.docx" `
  --embed-images 6
```

---

## 交互推理（Streamlit）

```powershell
cd <本仓库根目录>
streamlit run system/streamlit_app.py
```

浏览器访问终端提示的 **Local URL**（一般为 `http://localhost:8501`）。侧栏可更换 **模型权重路径**、上传图像、输入指代表达（支持中文离线翻译流程，见 `system/zh_translate.py`）。

---

## 从官方 RefCOCO 标注生成索引

若标注为 REFER 格式（`instances.json` + `refs(unc).p`）：

```powershell
python tools/build_refcoco_index.py `
  --dataset-dir "D:\path\to\refer\data\refcoco" `
  --refs-file "refs(unc).p" `
  --images-dir "D:\path\to\COCO\train2014" `
  --output-root "D:\path\to\your_refcoco_root" `
  --splits "train,val,testA,testB" `
  --copy-images
```

---

## 克隆与推送（GitHub）

首次克隆、数据与推送体积说明见 **[GITHUB_SETUP.md](GITHUB_SETUP.md)**。

---

## 索引 JSON 示例

```json
{
  "items": [
    {
      "image": "images/COCO_train2014_000000000009.jpg",
      "mask": "masks/train/000000000009_0.png",
      "text": "the man with a red shirt"
    }
  ]
}
```

---

## English summary

**RIS** pipeline: CLIP text + selectable segmentation head (**baseline** or **v33** cross-modal design), RefCOCO-style JSON indices, training/eval/visualization scripts, and a **Streamlit** UI. Checkpoints record `ris_arch` and `image_size` for correct loading. Large weights and image trees stay under `result/` and `refcoco_ready/{images,masks}/` locally; see **GITHUB_SETUP.md** for Git constraints.
