import argparse
import json
import os
import pickle
import shutil
from collections import defaultdict

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def parse_args():
    parser = argparse.ArgumentParser("Convert official RefCOCO annotations to RIS index JSON format")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Directory containing instances.json and refs(...).p, e.g. .../refer/data/refcoco",
    )
    parser.add_argument(
        "--refs-file",
        type=str,
        default="refs(unc).p",
        help="RefCOCO refs file name inside dataset-dir (default: refs(unc).p)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="COCO train2014 image directory",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Output root directory matching ris_mvp expected layout",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,testA,testB",
        help="Comma-separated splits to export",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy used images into output_root/images",
    )
    return parser.parse_args()


def load_refs(path):
    with open(path, "rb") as f:
        refs = pickle.load(f)  # REFER 官方提供的 refs(unc).p 等，内含 split / ann_id / sentences
    if not isinstance(refs, list):
        raise ValueError(f"Unexpected refs data format in: {path}")
    return refs


def load_instances(path):
    with open(path, "r", encoding="utf-8") as f:
        instances = json.load(f)
    anns = {ann["id"]: ann for ann in instances["annotations"]}  # annotation id → 多边形 / RLE 分割
    imgs = {img["id"]: img for img in instances["images"]}  # image id → 高宽与文件名
    return anns, imgs


def ann_to_mask(ann, h, w):
    seg = ann["segmentation"]  # COCO 中为多边形列表或 RLE 字典
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, h, w)  # 多边形转 RLE
        rle = mask_utils.merge(rles)  # 多块合并为单 mask
    elif isinstance(seg, dict):
        if isinstance(seg.get("counts"), list):
            rle = mask_utils.frPyObjects(seg, h, w)  # 部分标注 counts 为 list，需再编码
        else:
            rle = seg  # 已是压缩 RLE
    else:
        raise ValueError("Unsupported segmentation format")

    m = mask_utils.decode(rle)  # H×W 或 H×W×n 二值
    if m.ndim == 3:
        m = np.any(m, axis=2).astype(np.uint8)  # 多通道合并为单通道
    else:
        m = (m > 0).astype(np.uint8)
    return m


def ensure_dirs(root, split_names):
    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    os.makedirs(os.path.join(root, "masks"), exist_ok=True)
    os.makedirs(os.path.join(root, "splits"), exist_ok=True)
    for sp in split_names:
        os.makedirs(os.path.join(root, "masks", sp), exist_ok=True)


def main():
    args = parse_args()
    split_names = [x.strip() for x in args.splits.split(",") if x.strip()]
    ensure_dirs(args.output_root, split_names)

    refs_path = os.path.join(args.dataset_dir, args.refs_file)
    instances_path = os.path.join(args.dataset_dir, "instances.json")

    refs = load_refs(refs_path)
    anns, imgs = load_instances(instances_path)

    split_items = defaultdict(list)
    used_images = set()
    missing_image = 0
    missing_ann = 0

    for ref in refs:
        split = ref.get("split")
        if split not in split_names:
            continue

        ann = anns.get(ref["ann_id"])
        if ann is None:
            missing_ann += 1
            continue

        img_meta = imgs.get(ref["image_id"])
        if img_meta is None:
            missing_image += 1
            continue

        file_name = img_meta["file_name"]
        image_path = os.path.join(args.images_dir, file_name)
        if not os.path.exists(image_path):
            missing_image += 1
            continue

        h, w = img_meta["height"], img_meta["width"]
        mask = ann_to_mask(ann, h, w)
        mask_u8 = (mask * 255).astype(np.uint8)

        for sent in ref.get("sentences", []):  # 同一物体多条自然语言描述，每条对应相同样式的二值 mask
            sent_text = sent.get("sent", "").strip()
            if not sent_text:
                continue

            ref_id = ref.get("ref_id", ref["ann_id"])
            sent_id = sent.get("sent_id", 0)
            mask_name = f"{ref['image_id']}_{ref_id}_{sent_id}.png"  # 文件名唯一定位 (图,指代,句子)
            mask_rel = os.path.join("masks", split, mask_name).replace("\\", "/")
            mask_abs = os.path.join(args.output_root, "masks", split, mask_name)

            if not os.path.exists(mask_abs):  # 避免重复写盘（多条句子共用同一 ann 时 mask 相同）
                Image.fromarray(mask_u8).save(mask_abs)

            image_rel = os.path.join("images", file_name).replace("\\", "/")  # 训练时与 --data-root 拼接
            split_items[split].append(
                {
                    "image": image_rel,
                    "mask": mask_rel,
                    "text": sent_text,
                }
            )
            used_images.add(file_name)

    if args.copy_images:
        copied = 0
        for file_name in used_images:
            src = os.path.join(args.images_dir, file_name)
            dst = os.path.join(args.output_root, "images", file_name)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied += 1
        print(f"Copied {copied} images to {os.path.join(args.output_root, 'images')}")

    for sp in split_names:
        out_json = os.path.join(args.output_root, "splits", f"{sp}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"items": split_items[sp]}, f, ensure_ascii=False, indent=2)
        print(f"[{sp}] items: {len(split_items[sp])} -> {out_json}")

    print(f"Done. missing_ann={missing_ann}, missing_image={missing_image}")
    if not args.copy_images:
        print("Reminder: use --copy-images or manually place COCO images into output_root/images.")


if __name__ == "__main__":
    main()
