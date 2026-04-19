"""
在指定划分上评估已保存的 RIS 权重：加载 checkpoint 中的 ris_arch / clip 配置，输出 mIoU、cIoU 等。
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.refcoco_dataset import RefCOCOIndexDataset
from models.clip_ris import ClipTextGuidedRIS
from models.clip_ris_v33 import ClipRISV33
from utils.metrics import accumulate_miou_ciou


def _default_num_workers() -> int:
    # 与 train 一致：Windows 上多进程 DataLoader 易与 CUDA 冲突
    return 0 if sys.platform == "win32" else 4


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)  # 与 train.py 一致的 Dice，用于评估时与训练损失同量纲
    num = 2 * (probs * targets).sum(dim=(1, 2, 3))
    den = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


@torch.no_grad()
def evaluate_split(
    model,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float, float, int]:
    """返回 mean_loss、mIoU、cIoU、batch 数（与 train.py 验证口径一致）。"""
    model.eval()
    total_loss = 0.0
    total_per_iou_sum = 0.0
    total_n_samples = 0
    total_inter = 0.0
    total_union = 0.0
    eps = 1e-6
    n_batches = 0
    for images, masks, text_tokens, _ in tqdm(loader, desc="Eval", ncols=100, leave=False):
        images = images.to(device)
        masks = masks.to(device)
        text_tokens = text_tokens.to(device)
        logits = model(images, text_tokens)
        bce = F.binary_cross_entropy_with_logits(logits, masks)
        dloss = dice_loss(logits, masks)
        loss = (bce + dloss).item()
        total_loss += loss

        per_sum, n_samp, inter_b, union_b = accumulate_miou_ciou(logits, masks, eps)
        total_per_iou_sum += per_sum.item()
        total_n_samples += n_samp
        total_inter += inter_b.item()
        total_union += union_b.item()
        n_batches += 1
    if n_batches == 0 or total_n_samples == 0:
        return 0.0, 0.0, 0.0, 0
    mean_loss = total_loss / n_batches
    miou = total_per_iou_sum / total_n_samples
    ciou = total_inter / (total_union + eps)
    return mean_loss, miou, ciou, n_batches


def parse_split_args(split_strs: List[str]) -> List[Tuple[str, str]]:
    out = []
    for s in split_strs:
        if "=" not in s:
            raise ValueError(f'--split must be name=path, got: {s}')
        name, path = s.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f'Invalid --split: {s}')
        out.append((name, path))
    return out


def build_parser():
    p = argparse.ArgumentParser("Evaluate RefCOCO RIS checkpoint")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument(
        "--split",
        action="append",
        default=[],
        metavar="NAME=REL_PATH",
        help='Repeatable, e.g. --split val=splits/val.json --split testA=splits/testA.json',
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--num-workers",
        type=int,
        default=_default_num_workers(),
        help="DataLoader 进程数；Windows 默认 0，Linux 默认 4",
    )
    p.add_argument("--image-size", type=int, default=224, help="须与训练时 --image-size 一致（256 实验需显式指定）")
    p.add_argument("--clip-model", type=str, default="ViT-B/32")
    p.add_argument("--save-json", type=str, default="", help="Write metrics JSON to this path")
    p.add_argument("--max-samples", type=int, default=0, help="If >0, only use first N samples per split (debug)")
    return p


def main():
    args = build_parser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = parse_split_args(args.split)
    if not splits:
        raise SystemExit("Provide at least one --split name=relative/path.json")

    ckpt = torch.load(args.checkpoint, map_location=device)  # 含 model / optimizer / epoch 等（评估仅用 model）
    ckpt_args = dict(ckpt.get("args") or {})
    clip_name = str(ckpt_args.get("clip_model", args.clip_model))
    ris_arch = str(ckpt_args.get("ris_arch", "baseline")).lower()

    clip_model, _ = clip.load(clip_name, device=device)
    clip_model = clip_model.float()  # 与 train 中一致，保证 state_dict dtype 与可训练权重对齐
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    if ris_arch in ("v33", "33", "doc33"):
        model = ClipRISV33(clip_model=clip_model, clip_text_trainable=False).to(device)
    else:
        model = ClipTextGuidedRIS(clip_model=clip_model, clip_text_trainable=False).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    ckpt_epoch = ckpt.get("epoch")
    ckpt_best_miou = ckpt.get("best_val_miou", ckpt.get("best_val_iou"))

    results: Dict = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "ckpt_epoch": ckpt_epoch,
        "ckpt_best_val_miou": ckpt_best_miou,
        "splits": {},
    }

    print(f"Device: {device}")
    print(f"Loaded checkpoint: {args.checkpoint} (epoch={ckpt_epoch}, best_val_miou={ckpt_best_miou})")

    for name, rel_index in splits:
        index_path = rel_index if os.path.isabs(rel_index) else os.path.join(args.data_root, rel_index)
        ds = RefCOCOIndexDataset(index_path, root_dir=args.data_root, image_size=args.image_size)
        if args.max_samples > 0:  # 调试时可只评前 N 条，加快迭代
            n = min(args.max_samples, len(ds))
            ds = Subset(ds, range(n))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)  # 不打乱，结果可复现
        loss, miou, ciou, nb = evaluate_split(model, loader, device)
        results["splits"][name] = {
            "index": os.path.abspath(index_path),
            "num_samples": len(ds),
            "num_batches": nb,
            "mean_loss": loss,
            "mIoU": miou,
            "cIoU": ciou,
            "mean_iou": miou,
        }
        print(f"[{name}] samples={len(ds)} mean_loss={loss:.4f} mIoU={miou:.4f} cIoU={ciou:.4f}")

    if args.save_json:
        out_json = os.path.abspath(args.save_json)
        parent = os.path.dirname(out_json)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
