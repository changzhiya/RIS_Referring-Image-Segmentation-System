import json
import os
import random
from typing import Dict, List, Optional, Tuple

import clip
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

# OpenAI CLIP 默认归一化（与 encode_text 训练分布一致；文档 3.1.3 的 ImageNet 统计不用于本工程）
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class RefCOCOIndexDataset(Dataset):
    """RefCOCO（或同类）索引式数据集，供 DataLoader 按批读取。

    作用：根据 JSON 中的相对路径，从 ``root_dir`` 加载图像、二值掩码与指代表达文本；
    对图像做与 CLIP 一致的归一化，对掩码做最近邻缩放并二值化；
    将文本转为 CLIP 词表索引，供 ``ClipTextGuidedRIS`` 使用。

    ``augment_train=True`` 时（文档 3.1.3）：对 **图像与 mask 同步** 做 RandomResizedCrop 与 RandomHorizontalFlip，
    再缩放到 ``image_size``；默认裁剪 ``scale=(0.8, 1.0)``，可用 ``aug_crop_scale`` 覆盖（如略抬高下界以减轻 train/val 分布差）。
    验证集应使用 ``augment_train=False``。

    索引 JSON 格式示例:
        {
          "items": [
            {
              "image": "images/COCO_train2014_000000000009.jpg",
              "mask": "masks/train/000000000009_0.png",
              "text": "the man with a red shirt"
            }
          ]
        }
    """

    def __init__(
        self,
        index_file: str,
        root_dir: str,
        image_size: int = 352,
        augment_train: bool = False,
        aug_crop_scale: Optional[Tuple[float, float]] = None,
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.augment_train = augment_train
        self.aug_crop_scale = aug_crop_scale if aug_crop_scale is not None else (0.8, 1.0)
        with open(index_file, "r", encoding="utf-8") as f:
            payload: Dict[str, List[Dict[str, str]]] = json.load(f)
        self.items = payload.get("items", [])
        if len(self.items) == 0:
            raise ValueError(f"No items found in index file: {index_file}")

        self.image_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
            ]
        )
        self.mask_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.items)

    def _apply_doc_augment(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """文档 3.1.3：同步随机裁剪与水平翻转，再 CLIP 归一化。"""
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=self.aug_crop_scale, ratio=(3.0 / 4.0, 4.0 / 3.0)
        )
        image = TF.resized_crop(
            image, i, j, h, w, (self.image_size, self.image_size), InterpolationMode.BILINEAR
        )
        mask = TF.resized_crop(
            mask, i, j, h, w, (self.image_size, self.image_size), InterpolationMode.NEAREST
        )
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        image_t = transforms.ToTensor()(image)
        image_t = transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD)(image_t)
        mask_t = transforms.ToTensor()(mask)
        mask_t = (mask_t > 0.5).float()
        return image_t, mask_t

    def __getitem__(self, idx: int):
        item = self.items[idx]
        image_path = os.path.join(self.root_dir, item["image"])
        mask_path = os.path.join(self.root_dir, item["mask"])
        text = item["text"]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment_train:
            image, mask = self._apply_doc_augment(image, mask)
        else:
            image = self.image_tf(image)
            mask = self.mask_tf(mask)
            mask = (mask > 0.5).float()

        text_tokens = clip.tokenize([text], truncate=True).squeeze(0)
        return image, mask, text_tokens, text
