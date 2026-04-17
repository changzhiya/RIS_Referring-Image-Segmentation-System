"""
one.docx §3.3：空间位置感知 + CLIP 文本 token 级特征 + 轻量跨模态注意力（独立分支，需单独训练）。

与 ClipTextGuidedRIS 不共享权重；checkpoint 中 args.ris_arch == \"v33\" 时由 predict/eval 加载本结构。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.clip_ris import TinyImageEncoder, _gn


def clip_text_token_features(clip_model, text_tokens: torch.Tensor) -> torch.Tensor:
    """CLIP 文本塔：返回每个 context 位置的 512 维特征 [B, L, 512]（含 padding 位置，下游用 mask 忽略）。"""
    # 与 train 中一致，在 float32 上算，避免半精度 transformer 数值问题
    te = clip_model.token_embedding(text_tokens).float()
    pos = clip_model.positional_embedding.float()[: te.shape[1]]
    x = te + pos
    x = x.permute(1, 0, 2)  # LND
    x = clip_model.transformer(x.float())
    x = x.permute(1, 0, 2)
    x = clip_model.ln_final(x).float()
    x = x @ clip_model.text_projection.float()
    return x


class SpatialPosResidual2D(nn.Module):
    """将归一化 x/y 网格编码为与图像特征同通道的残差（可学习），对应文档「空间位置编码」的轻量实现。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hid = max(32, channels // 4)
        self.net = nn.Sequential(
            nn.Conv2d(2, hid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, channels, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feat.shape
        device, dtype = feat.device, feat.dtype
        yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        coord = torch.cat([xx, yy], dim=1)
        return feat + self.net(coord)


class ClipRISV33(nn.Module):
    """3.3 优化头：TinyCNN + 空间残差 + token 文本 + 单头跨模态注意力 + 分割解码。"""

    def __init__(self, clip_model, feat_channels: int = 256, clip_text_trainable: bool = False):
        super().__init__()
        self.clip_model = clip_model
        self.clip_text_trainable = clip_text_trainable
        self.feat_channels = feat_channels
        self.image_encoder = TinyImageEncoder(out_channels=feat_channels)
        self.spatial_pos = SpatialPosResidual2D(feat_channels)
        self.token_proj = nn.Linear(512, feat_channels)
        self.token_gate = nn.Sequential(nn.Linear(feat_channels, feat_channels), nn.Sigmoid())
        self.cross_scale = float(feat_channels) ** -0.5
        self.decoder = nn.Sequential(
            nn.Conv2d(feat_channels * 2 + 1, feat_channels, kernel_size=3, padding=1),
            _gn(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def _encode_tokens(self, text_tokens: torch.Tensor) -> torch.Tensor:
        if self.clip_text_trainable:
            return clip_text_token_features(self.clip_model, text_tokens)
        with torch.no_grad():
            return clip_text_token_features(self.clip_model, text_tokens)

    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        img = self.spatial_pos(self.image_encoder(images))  # [B,C,h,w]
        b, c, h, w = img.shape
        dense = F.normalize(img, dim=1)

        tok512 = self._encode_tokens(text_tokens)  # [B,L,512]
        proj = self.token_proj(tok512)
        tok = self.token_gate(proj) * proj
        tok = F.normalize(tok, dim=-1)

        valid = (text_tokens != 0).float()  # [B,L]，padding 为 0
        # 跨模态：每个像素对 token 的注意力（单头，scaled dot-product）
        v = dense.view(b, c, h * w).permute(0, 2, 1)  # [B,N,C]
        scores = torch.bmm(v, tok.transpose(1, 2)) * self.cross_scale  # [B,N,L]
        scores = scores.masked_fill(valid.unsqueeze(1) == 0, -1e4)
        attn = F.softmax(scores, dim=-1)
        ctx = torch.bmm(attn, tok)  # [B,N,C]
        ctx_map = ctx.permute(0, 2, 1).view(b, c, h, w)

        # 全局句向量相似度支路（与 baseline 一致思想，稳定训练）
        wsum = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        gvec = (tok * valid.unsqueeze(-1)).sum(dim=1) / wsum
        gvec = F.normalize(gvec, dim=-1)
        sim = (dense * gvec.view(b, c, 1, 1)).sum(dim=1, keepdim=True)

        fused = torch.cat([img, ctx_map, sim], dim=1)
        logits_small = self.decoder(fused)
        return F.interpolate(logits_small, size=images.shape[-2:], mode="bilinear", align_corners=False)
