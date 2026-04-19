"""
从索引 JSON 随机抽样，将指代分割预测与原图/GT 叠加导出到目录（用于定性检查 checkpoint）。
"""
import argparse
import json
import os
import random

import clip
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.clip_ris import ClipTextGuidedRIS
from models.clip_ris_v33 import ClipRISV33


def parse_args():
    p = argparse.ArgumentParser("Visualize RIS predictions")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--index", type=str, required=True, help="Relative or absolute path to split JSON")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=224, help="须与训练 checkpoint 的 --image-size 一致")
    p.add_argument("--clip-model", type=str, default="ViT-B/32")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    index_path = args.index if os.path.isabs(args.index) else os.path.join(args.data_root, args.index)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(index_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    items = payload.get("items", [])
    if not items:
        raise SystemExit("No items in index")

    n = min(args.num_samples, len(items))  # 实际导出条数（不超过索引总量）
    indices = random.sample(range(len(items)), n) if n < len(items) else list(range(n))  # 可复现的随机抽样（受 seed 约束）

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = dict(ckpt.get("args") or {})
    clip_name = str(ckpt_args.get("clip_model", args.clip_model))
    ris_arch = str(ckpt_args.get("ris_arch", "baseline")).lower()

    clip_model, _ = clip.load(clip_name, device=device)
    clip_model = clip_model.float()
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    if ris_arch in ("v33", "33", "doc33"):
        model = ClipRISV33(clip_model=clip_model, clip_text_trainable=False).to(device)
    else:
        model = ClipTextGuidedRIS(clip_model=clip_model, clip_text_trainable=False).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    image_tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    mask_tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )

    meta = {"index": index_path, "checkpoint": os.path.abspath(args.checkpoint), "samples": []}

    with torch.no_grad():
        for rank, idx in enumerate(indices):
            item = items[idx]
            image_path = os.path.join(args.data_root, item["image"])
            mask_path = os.path.join(args.data_root, item["mask"])
            text = item["text"]

            pil_img = Image.open(image_path).convert("RGB")
            pil_mask = Image.open(mask_path).convert("L")

            x = image_tf(pil_img).unsqueeze(0).to(device)  # 与训练相同的 CLIP 归一化与尺寸
            tokens = clip.tokenize([text], truncate=True).to(device)  # 单句文本转 token
            logits = model(x, tokens)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()  # 模型输入分辨率下的前景概率图

            gt = mask_tf(pil_mask)  # 与 logits 同尺寸的 GT（用于辅助，文件名 IoU 用原分辨率）
            gt_np = (gt[0].numpy() > 0.5).astype(np.float32)  # 训练尺度二值 GT（此处未单独出图）

            h0, w0 = pil_img.size[1], pil_img.size[0]  # PIL 的 height, width
            pred_full = Image.fromarray((prob * 255).astype(np.uint8)).resize((w0, h0), Image.BILINEAR)  # 双线性拉到原图大小便于叠加
            pred_bin = (np.array(pred_full) > 127).astype(np.float32)  # 原图尺度二值预测

            gt_full = pil_mask.resize((w0, h0), Image.NEAREST)  # GT 掩码对齐原图，最近邻避免标签混叠
            gt_bin = (np.array(gt_full.convert("L")) > 127).astype(np.float32)
            inter = (pred_bin * gt_bin).sum()  # 原图分辨率上交集
            union = ((pred_bin + gt_bin) > 0).sum() + 1e-6  # 并集 + 数值稳定
            iou = float(inter / union)  # 写入文件名，便于肉眼看难易

            base = f"{rank:03d}_iou{iou:.3f}"
            pil_img.save(os.path.join(args.out_dir, f"{base}_input.jpg"))  # 纯原图，无叠加

            ov_pred = np.array(pil_img).astype(np.float32)
            overlay = np.zeros_like(ov_pred)
            overlay[:, :, 1] = pred_bin * 200.0  # G 通道高亮预测区域（绿色系）
            blended = np.clip(ov_pred * 0.55 + overlay * 0.45, 0, 255).astype(np.uint8)  # 与原图 alpha 混合
            Image.fromarray(blended).save(os.path.join(args.out_dir, f"{base}_pred_overlay.jpg"))  # 预测可视化

            ov_gt = np.array(pil_img).astype(np.float32)
            go = np.zeros_like(ov_gt)
            go[:, :, 0] = gt_bin * 200.0  # R 通道高亮真值（红色系）
            blended_g = np.clip(ov_gt * 0.55 + go * 0.45, 0, 255).astype(np.uint8)  # 与预测叠加相同的混合比例
            Image.fromarray(blended_g).save(os.path.join(args.out_dir, f"{base}_gt_overlay.jpg"))

            with open(os.path.join(args.out_dir, f"{base}_text.txt"), "w", encoding="utf-8") as tf:
                tf.write(text)

            meta["samples"].append(
                {"rank": rank, "index_in_split": idx, "iou": iou, "text": text}
            )

    with open(os.path.join(args.out_dir, "visualize_meta.json"), "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2, ensure_ascii=False)

    print(f"Saved {n} visualizations to: {args.out_dir}")


if __name__ == "__main__":
    main()
