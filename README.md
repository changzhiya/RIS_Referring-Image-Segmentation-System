# 文本引导指代图像分割系统（RIS）

基于 **PyTorch** 与 **OpenAI CLIP** 的指代表达图像分割（Referring Image Segmentation）工程：支持 **RefCOCO 索引数据**、**训练 / 评估 / 可视化**，并提供 **Streamlit** 交互推理界面。

---

## 系统与模型

| 组件 | 说明 |
|------|------|
| **CLIP** | 文本编码；训练时可按 `--clip-unfreeze-last` 解冻文本 Transformer 末若干层。 |
| **分割头（二选一）** | **`--ris-arch baseline`**：`ClipTextGuidedRIS`，CLIP 句向量 + 轻量图像编码 + 解码。**`--ris-arch v33`**：`ClipRISV33`（文档 §3.3）：空间位置残差、**token 级**文本特征、像素–token 跨模态注意力 + 解码；与 baseline **权重不通用**，须单独训练。 |
| **权重** | 训练写入 `--save-dir`（默认在 `result/` 下子目录）。**保存策略与验证 mIoU / cIoU 相关**（见下文「验证指标」）。`ckpt["args"]` 含 `ris_arch`、`image_size` 等；`eval.py` / `visualize.py` / `system/predict.py` 按 checkpoint 自动构造对应网络。 |
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

### 验证指标 mIoU、cIoU 与权重文件

训练过程中，每个 epoch 结束会在**验证集**上计算两类 IoU（实现见 `utils/metrics.py`，与 `eval.py` 口径一致）：

| 指标 | 含义（验证集） |
|------|----------------|
| **mIoU**（`val_mIoU`） | **逐样本 IoU** 的算术平均：先对每张图算预测与真值的 IoU，再对所有样本取平均。更能反映「每张图是否都分得好」。 |
| **cIoU**（`val_cIoU`） | **累积 IoU**：把整个验证集上所有样本的前景 **交集像素总数 / 并集像素总数**，等价于「全集当作一张大图」的 IoU，大目标权重更高。 |

**与权重保存的关系：**

- **`<save-dir>/val_metrics.jsonl`**：每轮追加一行 JSON，含该轮的 `val_mIoU`、`val_cIoU`、`val_loss` 以及训练侧 `train_loss`、`train_iou_batch_mean` 等，便于画曲线或对比实验。
- **`best.pt`**：当本轮 **`val_mIoU` 高于历史最佳** 时覆盖写入（**不以 cIoU 单独作为选优标准**）；checkpoint 内同时记录当时的 **`best_val_miou`**、**`best_val_ciou`**（便于对照「mIoU 最佳那一轮」上的 cIoU），以及 `epoch`、`args` 等。
- **`last.pt`**：每个 epoch 结束都会更新，用于中断后从上一轮整轮结束处 **`--resume`** 续训。

推理或部署时一般选用 **`best.pt`**（验证 mIoU 最优）；若更关心全集像素级重合，可结合 `val_metrics.jsonl` 查看各轮 cIoU 再人工选轮次对应的 `last.pt`（需自行保留或从日志推断）。

- **续训**：`--resume path\to\last.pt`，并显式指定 **`--save-dir`** 与首次训练一致。

### 多次训练对比（验证集 val_mIoU / val_cIoU）

下表为**同一验证集**（RefCOCO `val`）上，多次不同配置训练的汇总：在各自 `result/<save-dir>/val_metrics.jsonl` 中，取 **`val_mIoU` 的全局最大值**，并列出**该轮**对应的 **`val_cIoU`** 与 **epoch**（与 `best.pt` 选优规则一致）。  

**「配置要点」对比维度**：分割头 **baseline / v33**、输入 **224² / 256²**、是否 **AMP**、**梯度累积**步数、**损失权重**（BCE / Dice / soft-IoU）、**学习率**（头 / CLIP）、**CLIP 文本解冻层数**、以及**计划 epoch 数**（与表中「最佳出现轮」不同）。要点来自各目录 `best.pt` 内 `args` 与 `train.py` 预设说明。  

*说明：数值随数据路径、随机种子与硬件略有波动；换机复现时请以你本地的 `val_metrics.jsonl` 为准，下表仅作配置对比参考。*

| 保存目录 `result/...` | 配置要点（差异一览） | 验证最佳 val_mIoU | 该轮 val_cIoU | epoch |
|----------------------|----------------------|-------------------:|---------------:|------:|
| `checkpoints_active` | **baseline**；**256²**；无 AMP；`grad_accum=8`；头 LR **3e-4**、CLIP **5e-6**；解冻文本末 **1** 层；**默认** BCE+Dice（无 soft-IoU）；默认 `save-dir`，**计划 15 epoch**（偏「长训 + 稳显存」） | **0.2632** | 0.2818 | 14 |
| `checkpoints_doc31` | **baseline**；**224²**（更小输入省显存）；**AMP 开**；`accum=4`；头/CLIP 均为 **5e-5**；解冻末 **2** 层；**`--doc-stage1`**：损失 **0.5·BCE+0.3·Dice+0.2·softIoU** + 余弦等文档预设；**计划 10 epoch**（偏「文档对齐 + 双端同 LR」） | **0.2405** | 0.2655 | 8 |
| `checkpoints_miou10_20260414_215059` | **baseline**；**256²**；**AMP 开**；`accum=4`；头 **8e-5**、CLIP **2e-5**；解冻 **1** 层；**`--preset-early-val-miou`**：损失 **0.15·BCE+0.45·Dice+0.40·softIoU** + 更强裁剪（如 scale≥0.88）；时间戳独立目录（偏「抬早期 val mIoU」） | **0.2629** | 0.2733 | 7 |
| `checkpoints_retrain` | **baseline**；**224²**（与 active 的 256 不同，与 doc31 同分辨率）；无 AMP；`accum=8`；LR 与 **active 同档**（3e-4 / 5e-6）；解冻 **1** 层；**仅计划 5 epoch** 的短训/重训实验（总监督步数明显少于 active） | **0.2479** | 0.2758 | 5 |
| `checkpoint_v2` | **ClipRISV33（`--ris-arch v33`）**：在 baseline 之外增加 **位置编码模块**（图像特征上的可学习空间位置残差）与 **细粒度文本特征提取模块**（CLIP 文本 **token 级** 表征 + 像素–token 跨模态注意力，而非仅用句向量）；与上表其余 **baseline** 权重不通用；**256²**；无 AMP；`accum=8`；LR 与 active 同档；解冻 **1** 层；**默认** BCE+Dice；**主实验**（与下文 eval 示例 `best.pt` 一致，`val_metrics.jsonl` 见该目录） | **0.4112** | 0.4098 | 9 |

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

浏览器访问终端提示的 **Local URL**（一般为 `http://localhost:8501`）。侧栏可更换 **模型权重路径**、上传图像、输入指代表达。

### 中文指代与离线翻译（实现说明）

CLIP 文本塔面向**英文**分布训练，界面若输入**中文指代**，需先变成英文再 `clip.tokenize` 送入分割模型。本系统在 **`system/zh_translate.py`** 中实现离线翻译层，并由 **`system/streamlit_app.py`** 在推理前调用。  

**具体机翻模型**：采用 **Helsinki-NLP** 在 Hugging Face 上发布的 **中英翻译模型** **[Helsinki-NLP/opus-mt-zh-en](https://huggingface.co/Helsinki-NLP/opus-mt-zh-en)**（Opus-MT 系列，**中文 → 英文** 方向）；实现上为 Marian 序列到序列结构，经 `transformers` 加载，默认在 **CPU** 推理。

**1. 总体数据流**

- 用户在侧栏选择 **「语言模式」**（`lang_mode`）：**英文（CLIP 原生）** 或 **中文（离线翻译）**。
- 推理函数根据当前文本调用 **`zh_translate.resolve_for_clip(原文, lang_mode)`**，得到 **`(送入 CLIP 的英文, 可选说明文案)`**；分割与可视化均基于该英文句。

**2. 译文从哪里来（优先级从高到低）**

| 步骤 | 行为 |
|------|------|
| ① 词典 | 若原文整句命中 **`CUSTOM_TRANSLATION`**（`zh_translate.py` 内字典），**直接使用固定英文**，不调用神经网络（答辩演示更稳、无延迟）。 |
| ② Marian 机翻 | 否则调用上述 **Helsinki-NLP 中英（zh→en）翻译模型** `opus-mt-zh-en`（Marian），经 **`transformers`** 在 **CPU** 上 `generate` 得到英文；**不占 GPU**，避免与分割抢显存。 |
| ③ 何时机翻 | **「中文（离线翻译）」**：输入中含汉字则走 ①/②；**「英文（CLIP 原生）」**：仅当检测到汉字时才自动机翻（混合输入场景）。纯英文则原样送入 CLIP。 |

**3. Helsinki-NLP 中英翻译模型如何加载**

- **环境变量 `ZH_EN_MT_MODEL`**：若设为某目录路径，则**优先**从该目录加载快照（须含 `config.json` 与 `pytorch_model.bin` 或 `model.safetensors`）。
- **否则**按顺序查找项目内约定目录（**第一个完整快照即用**）：`<项目根>/models/opus-mt-zh-en/` → `<项目根>/train/` → **上一级目录**的 `train/`（便于与历史权重目录并存）。
- 若某约定目录**只有 config、缺权重**，模块会**报错提示补全**，避免误连外网；无本地完整快照时，回退为从 Hub 拉取同一 **`Helsinki-NLP/opus-mt-zh-en`** 权重（代码中默认设置 **`HF_ENDPOINT`** 镜像与下载超时，见 `zh_translate._load_mt`）。
- **预加载**：Streamlit 侧栏可调用 **`zh_translate.warmup_mt()`**，把 **Helsinki-NLP opus-mt-zh-en** 提前载入内存；界面也会展示 **`describe_mt_resolution()`** 的文案，说明当前将从何处加载。

**4. 机翻质量与性能**

- 默认 **`ZH_EN_MT_NUM_BEAMS=1`**（贪心解码），减轻拖动阈值时重复推理的卡顿；需要更高质量可在环境中设为 **`3`** 等开启 beam。
- 同一句话、仅改分割阈值时，应用层可**缓存译文**，避免重复跑 **opus-mt-zh-en**（具体见 `streamlit_app.py` 中推理与 `thr_only` 相关逻辑）。

**5. 依赖与离线部署**

- 需安装 **`system/requirements-demo.txt`** 中的 **`transformers`**、**`sentencepiece`**、**`sacremoses`** 等。
- 可将模型快照下载到 **`models/opus-mt-zh-en/`**（仓库 **`.gitignore` 已忽略** 该目录，避免大文件进 Git）；或使用仓库内 **`scripts/download_opus_mt_zh_en.py`** 拉取到该路径。

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

**RIS** pipeline: CLIP text + selectable segmentation head (**baseline** or **v33** cross-modal design), RefCOCO-style JSON indices, training/eval/visualization scripts, and a **Streamlit** UI. Checkpoints record `ris_arch` and `image_size` for correct loading. Training logs per-epoch **mIoU** (mean of per-image IoU) and **cIoU** (cumulative intersection-over-union); **`best.pt` is chosen by validation mIoU**, with `best_val_ciou` stored for reference. A **multi-run comparison table** (best val mIoU / matching cIoU per configuration) is included in the Chinese section above. **Chinese referring expressions** are handled offline before CLIP: a **phrase dictionary** then the **Helsinki-NLP Chinese-to-English translation model** **`opus-mt-zh-en`** (Opus-MT / Marian via `transformers`, CPU) in `system/zh_translate.py`, wired from Streamlit through `resolve_for_clip`; details are in the Chinese subsection **「中文指代与离线翻译（实现说明）」** above. Large weights and image trees stay under `result/` and `refcoco_ready/{images,masks}/` locally; see **GITHUB_SETUP.md** for Git constraints.
