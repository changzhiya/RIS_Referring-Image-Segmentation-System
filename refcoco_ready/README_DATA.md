# RefCOCO 数据说明（本仓库提交范围）

本目录 **已包含** `splits/*.json` 索引（约 22MB，可随 Git 推送）。

**未包含**（体积约数 GB，不适合普通 GitHub 仓库）：

- `images/` — COCO / RefCOCO 对应 JPEG
- `masks/` — 指代分割二值掩码 PNG

## 补全数据后训练

1. 按你本地或官方 RefCOCO / COCO 发布包，将 **`images`** 与 **`masks`** 放到本目录，使 `splits/*.json` 里每条 `image`、`mask` 相对路径均可解析。
2. 自检：

```powershell
python tools\verify_refcoco_setup.py
```

若抽样路径均存在，会打印 `[OK]`；若有缺失会提示 `[MISS]`。

## 无完整数据时

可使用仓库内 **`demo_refcoco/`** 跑通环境与单条训练/推理（见根目录 `README.md`）。
