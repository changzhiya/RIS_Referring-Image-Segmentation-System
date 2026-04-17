"""
分割指标：mIoU（逐样本 IoU 的算术平均）与 cIoU（累积 IoU，全验证集像素级总交/总比）。
"""

from typing import Tuple

import torch


def binary_per_sample_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """逐样本二值 IoU，形状 [B]。"""
    preds = (torch.sigmoid(logits) > 0.5).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
    return (inter + eps) / (union + eps)


def binary_batch_mean_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """batch 内先算逐样本 IoU 再取均值（与旧 train 中 iou_score 一致，用于训练集日志）。"""
    return binary_per_sample_iou(logits, targets, eps).mean()


def accumulate_miou_ciou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """
    从一个 batch 累加 mIoU / cIoU 所需的统计量。

    Returns:
        per_iou_sum: 本 batch 各样本 IoU 之和（标量 tensor）
        n_samples: 本 batch 样本数（int）
        inter_sum: 本 batch 交集像素总数（标量 tensor）
        union_sum: 本 batch 并集像素总数（标量 tensor）

    全划分上：mIoU = sum(per_iou_sum) / sum(n_samples)；
              cIoU = sum(inter_sum) / (sum(union_sum) + eps)。
    """
    per = binary_per_sample_iou(logits, targets, eps)
    preds = (torch.sigmoid(logits) > 0.5).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
    return per.sum(), int(per.numel()), inter.sum(), union.sum()
