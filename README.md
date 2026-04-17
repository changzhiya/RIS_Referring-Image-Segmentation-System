# RIS MVP (PyTorch + CLIP + RefCOCO)

This is a minimal runnable baseline for text-guided image segmentation.

> **对外发布 / GitHub**：以子目录 `RIS_Referring-Image-Segmentation-System/` 为仓库根时，请阅读其中 [`GITHUB_SETUP.md`](RIS_Referring-Image-Segmentation-System/GITHUB_SETUP.md)。

## 1) Environment

Use your existing conda env:

```powershell
conda activate ris_env
```

If needed, install dependencies:

```powershell
pip install torch torchvision tqdm pillow
pip install git+https://github.com/openai/CLIP.git
pip install pycocotools numpy
```

## 2) Dataset layout

This MVP uses an index JSON format for RefCOCO samples.

```text
your_refcoco_root/
  images/
    COCO_train2014_*.jpg
  masks/
    train/*.png
    val/*.png
  splits/
    train.json
    val.json
```

`train.json` and `val.json` format:

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

## 3) Train

```powershell
python .\train.py `
  --data-root "D:\path\to\your_refcoco_root" `
  --train-index "D:\path\to\your_refcoco_root\splits\train.json" `
  --val-index "D:\path\to\your_refcoco_root\splits\val.json" `
  --epochs 5 `
  --batch-size 8 `
  --image-size 352
```

Checkpoint will be saved to `.\checkpoints\best.pt`.

Each epoch, validation **mIoU** (mean per-sample IoU) and **cIoU** (cumulative IoU: total intersection / total union over the val set) are printed and appended to `.\checkpoints\val_metrics.jsonl` (override with `--metrics-log`). **best.pt** is kept when val **mIoU** improves; checkpoint fields include `best_val_miou` and `best_val_ciou`.

**Low VRAM (e.g. 4GB):** defaults use **image_size=224**, **batch_size=2**, **grad_accum_steps=8** (effective batch ~16), and **CLIP text** last block + `ln_final` fine-tune (`--lr-clip`, `--clip-unfreeze-last`). **AMP** is off by default (BN + grad accum can break GradScaler); pass `--amp` to try mixed precision if your run is stable.

## 4) Evaluate, visualize, and Word report

Dependencies for the report:

```powershell
pip install python-docx
```

**Evaluate** (mean IoU / loss on one or more splits):

```powershell
python .\eval.py `
  --data-root "D:\path\to\refcoco_ready" `
  --checkpoint ".\checkpoints\best.pt" `
  --split val=splits/val.json `
  --split testA=splits/testA.json `
  --split testB=splits/testB.json `
  --save-json ".\reports\eval_results.json"
```

**Visualize** (random samples; green overlay = prediction, red = GT):

```powershell
python .\visualize.py `
  --data-root "D:\path\to\refcoco_ready" `
  --index "splits\testA.json" `
  --checkpoint ".\checkpoints\best.pt" `
  --out-dir ".\reports\vis_testA" `
  --num-samples 16
```

**Build Word report** (Chinese, embeds figures from `--vis-dir`):

```powershell
python .\tools\build_report_docx.py `
  --eval-json ".\reports\eval_results.json" `
  --vis-dir ".\reports\vis_testA" `
  --out-docx ".\reports\RIS实验成果说明.docx" `
  --embed-images 6
```

## 5) Auto-convert official RefCOCO annotations

If your RefCOCO annotations come from REFER format (`instances.json` + `refs(unc).p`), use:

```powershell
python .\tools\build_refcoco_index.py `
  --dataset-dir "D:\path\to\refer\data\refcoco" `
  --refs-file "refs(unc).p" `
  --images-dir "D:\path\to\COCO\train2014" `
  --output-root "D:\path\to\your_refcoco_root" `
  --splits "train,val,testA,testB" `
  --copy-images
```

After conversion:

- `output-root\splits\train.json`, `val.json`, `testA.json`, `testB.json` are generated.
- `output-root\masks\<split>\*.png` masks are generated.
- With `--copy-images`, used COCO images are copied to `output-root\images\`.

Then train with `--train-index ...\train.json` and `--val-index ...\val.json`.

## 6) Notes

- This is intentionally minimal for thesis prototyping.
- CLIP is used as frozen text encoder.
- You can improve by swapping image encoder/decoder and adding stronger fusion modules.

---

# 中文说明（翻译版）

这是一个可运行的最小基线，用于文本引导图像分割（RIS）。

## 1）环境准备

使用你现有的 conda 环境：

```powershell
conda activate ris_env
```

如有需要，安装依赖：

```powershell
pip install torch torchvision tqdm pillow
pip install git+https://github.com/openai/CLIP.git
pip install pycocotools numpy
```

## 2）数据集目录结构

本项目使用 RefCOCO 的索引 JSON 格式。

```text
your_refcoco_root/
  images/
    COCO_train2014_*.jpg
  masks/
    train/*.png
    val/*.png
  splits/
    train.json
    val.json
```

`train.json` 与 `val.json` 格式如下：

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

## 3）训练

```powershell
python .\train.py `
  --data-root "D:\path\to\your_refcoco_root" `
  --train-index "D:\path\to\your_refcoco_root\splits\train.json" `
  --val-index "D:\path\to\your_refcoco_root\splits\val.json" `
  --epochs 5 `
  --batch-size 8 `
  --image-size 352
```

模型权重会保存到 `.\checkpoints\best.pt`。

每轮验证会打印并写入 **mIoU**（逐样本 IoU 算术平均）与 **cIoU**（验证集全体像素总交/总比），默认追加到 `.\checkpoints\val_metrics.jsonl`（可用 `--metrics-log` 指定路径）。**best.pt** 在验证 **mIoU** 提升时更新；权重内包含 `best_val_miou`、`best_val_ciou`（与最佳 mIoU 同一轮）。

**低显存（如 4GB）**：默认 **224 分辨率**、**batch_size=2**、**grad_accum_steps=8**（等效 batch≈16），并微调 CLIP **文本塔最后 N 层**（`--clip-unfreeze-last`，配合 `--lr-clip`）。**AMP 默认关闭**（BN+梯度累积易与 GradScaler 冲突）；可试 `--amp`。

## 4）评估、可视化与 Word 报告

安装报告依赖：`pip install python-docx`

**评估**（可多划分，输出 JSON）：

```powershell
python .\eval.py `
  --data-root "D:\path\to\refcoco_ready" `
  --checkpoint ".\checkpoints\best.pt" `
  --split val=splits/val.json `
  --split testA=splits/testA.json `
  --split testB=splits/testB.json `
  --save-json ".\reports\eval_results.json"
```

**可视化**（随机抽样；绿色为预测叠加，红色为真值叠加）：

```powershell
python .\visualize.py `
  --data-root "D:\path\to\refcoco_ready" `
  --index "splits\testA.json" `
  --checkpoint ".\checkpoints\best.pt" `
  --out-dir ".\reports\vis_testA" `
  --num-samples 16
```

**生成 Word 说明**：

```powershell
python .\tools\build_report_docx.py `
  --eval-json ".\reports\eval_results.json" `
  --vis-dir ".\reports\vis_testA" `
  --out-docx ".\reports\RIS实验成果说明.docx" `
  --embed-images 6
```

## 5）自动转换官方 RefCOCO 标注

如果你的 RefCOCO 标注来自 REFER 格式（`instances.json` + `refs(unc).p`），可以使用：

```powershell
python .\tools\build_refcoco_index.py `
  --dataset-dir "D:\path\to\refer\data\refcoco" `
  --refs-file "refs(unc).p" `
  --images-dir "D:\path\to\COCO\train2014" `
  --output-root "D:\path\to\your_refcoco_root" `
  --splits "train,val,testA,testB" `
  --copy-images
```

转换完成后：

- 会生成 `output-root\splits\train.json`、`val.json`、`testA.json`、`testB.json`
- 会生成 `output-root\masks\<split>\*.png` 掩码文件
- 若使用 `--copy-images`，会把用到的 COCO 图片复制到 `output-root\images\`

之后即可用 `--train-index ...\train.json` 和 `--val-index ...\val.json` 启动训练。

## 6）说明

- 该版本是面向毕设原型验证的最小实现。
- CLIP 在此作为冻结的文本编码器使用。
- 你可以替换图像编码器/解码器，并加入更强的融合模块来提升效果。
