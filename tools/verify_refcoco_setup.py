"""检查 refcoco_ready 或 demo 数据布局，避免 train/eval 路径错误。"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _project_root() -> Path:
    return _script_dir().parent


def check_index(root: Path, index_path: Path, n_samples: int) -> int:
    if not index_path.is_file():
        print(f"[FAIL] 索引不存在: {index_path}")
        return 1
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items") or []
    if not items:
        print(f"[FAIL] 索引无 items: {index_path}")
        return 1
    print(f"[OK] {index_path.name}: {len(items)} 条")
    rng = random.Random(0)
    picks = items if len(items) <= n_samples else rng.sample(items, n_samples)
    bad = 0
    for it in picks:
        img = root / str(it.get("image", ""))
        msk = root / str(it.get("mask", ""))
        if not img.is_file():
            print(f"  [MISS] image: {img}")
            bad += 1
        if not msk.is_file():
            print(f"  [MISS] mask:  {msk}")
            bad += 1
    if bad:
        print(f"[WARN] 抽样 {len(picks)} 条中有 {bad} 个路径缺失（需补全 images/masks）")
        return 2
    print(f"[OK] 抽样 {len(picks)} 条 image/mask 均存在")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=str, default="", help="默认: 项目根下 refcoco_ready；可改为 demo_refcoco")
    p.add_argument("--samples", type=int, default=8, help="每条索引随机检查多少条样本路径")
    args = p.parse_args()
    root = _project_root()
    data_root = Path(args.data_root).resolve() if args.data_root else (root / "refcoco_ready")
    if not data_root.is_dir():
        demo = root / "demo_refcoco"
        if demo.is_dir():
            print(f"[INFO] 无 {data_root.name}，检查内置 {demo} …")
            data_root = demo
        else:
            print(f"[FAIL] 数据根不存在: {data_root}")
            return 1

    splits = data_root / "splits"
    if not splits.is_dir():
        print(f"[FAIL] 无 splits 目录: {splits}")
        return 1

    rc = 0
    for name in ("train.json", "val.json", "testA.json", "testB.json"):
        path = splits / name
        if path.is_file():
            r = check_index(data_root, path, args.samples)
            rc = max(rc, r)
        else:
            print(f"[SKIP] 无 {name}")
    return min(rc, 1)


if __name__ == "__main__":
    sys.exit(main())
