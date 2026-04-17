# 作为 GitHub 独立仓库使用

本文件夹 **即项目根目录**（含 `train.py`、`models/`、`data/`、`system/` 等）。克隆后请按下述检查，避免路径或缺权重报错。

## 1. 环境与依赖

```powershell
conda create -n ris_env python=3.10 -y
conda activate ris_env
pip install torch torchvision tqdm pillow pycocotools numpy
pip install git+https://github.com/openai/CLIP.git
pip install -r system/requirements-demo.txt
```

## 2. 数据与 Git 提交范围

| 路径 | 是否适合 `git push` | 说明 |
|------|---------------------|------|
| `refcoco_ready/splits/*.json` | 是 | 已纳入版本控制 |
| `refcoco_ready/images/`、`masks/` | **否**（`.gitignore`） | 需本地或网盘/COCO 官方渠道补全 |
| `demo_refcoco/` | 是 | 最小可运行示例 |
| `result/` | **否** | 训练产出与 `.pt`，体积大 |

单文件 **&gt;100MB** 无法普通推送 GitHub；大权重请用 [Git LFS](https://git-lfs.com/) 或 Release 附件分发。

## 3. 克隆后自检

```powershell
cd RIS_Referring-Image-Segmentation-System
python tools\verify_refcoco_setup.py
python tools\verify_refcoco_setup.py --data-root demo_refcoco
```

## 4. 训练命令示例（数据已补全时）

```powershell
python train.py `
  --data-root .\refcoco_ready `
  --train-index .\refcoco_ready\splits\train.json `
  --val-index .\refcoco_ready\splits\val.json `
  --epochs 5
```

`--ris-arch v33` 时使用文档 §3.3 结构（与旧 baseline 权重不通用）。

## 5. 推理界面

```powershell
streamlit run system/streamlit_app.py
```

首次若无本地 `result/**/best.pt`，侧栏需 **手动填写** 训练得到的 `.pt` 路径；有权重后会自动优先选用名称含 `v33` 的 `best.pt`。
