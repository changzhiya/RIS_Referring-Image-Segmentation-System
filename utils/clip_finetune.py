"""仅解冻 CLIP 文本塔末尾若干层（本工程前向未使用 CLIP 视觉特征，解冻视觉无梯度）。"""

from typing import List

import torch.nn as nn


def unfreeze_clip_text_last_blocks(clip_model: nn.Module, n_blocks: int) -> List[nn.Parameter]:
    """冻结全部 CLIP 参数后，解冻文本 Transformer 最后 n 个 block 与 ln_final。返回需训练的 CLIP 参数列表。"""
    for p in clip_model.parameters():
        p.requires_grad = False
    if n_blocks <= 0:
        return []

    trainable: List[nn.Parameter] = []
    blocks = getattr(getattr(clip_model, "transformer", None), "resblocks", None)
    if blocks is None or len(blocks) == 0:
        return []

    n = min(n_blocks, len(blocks))
    for b in blocks[-n:]:
        for p in b.parameters():
            p.requires_grad = True
            trainable.append(p)
    ln = getattr(clip_model, "ln_final", None)
    if ln is not None:
        for p in ln.parameters():
            p.requires_grad = True
            trainable.append(p)
    return trainable
