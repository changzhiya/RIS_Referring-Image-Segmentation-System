"""
CLIP 文本引导的轻量 RIS（MVP）。

one.docx 第 3 阶段（空间位置编码、细粒度 token 文本编码、跨模态注意力等）需改张量形状与 CLIP 接口，
并重新训练；当前仓库保持与已有 checkpoint 兼容，推理侧增强见 system/predict.py（温度 / 最大连通域）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels: int) -> nn.GroupNorm:
    """小 batch（如 micro_batch=2）时 BatchNorm 统计极不稳定，易导致权重 NaN；GroupNorm 与 batch 无关。"""
    g = 8 if num_channels >= 8 else 1
    while num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class TinyImageEncoder(nn.Module):
    """轻量卷积图像编码器。

    作用：从输入 RGB 图像提取中等分辨率的稠密特征图，作为分割支路的视觉表示。
    不使用 CLIP 视觉塔，以降低显存占用并加快训练，适合毕设/MVP 快速验证。
    输出通道数与文本投影维度一致，便于后续与句子向量逐像素对齐。
    """

    def __init__(self, out_channels: int = 256):
        super().__init__()
        self.net = nn.Sequential(  # 轻量 CNN：下采样得到稠密特征，参数量小、易在笔记本 GPU 上训练
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            _gn(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            _gn(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),
            _gn(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClipTextGuidedRIS(nn.Module):
    """CLIP 文本引导的指代表达图像分割模型（RIS 最小实现）。

    作用：根据自然语言描述，在图像上预测对应目标区域的二值分割 mask。
    - 文本侧：冻结的 CLIP encode_text 得到句子嵌入，经线性层映射到与图像特征相同的通道维，
      再与像素级特征做余弦相似度，得到「语义相关」得分图。
    - 图像侧：TinyImageEncoder 提取稠密局部特征。
    - 解码：将图像特征与相似度图拼接，经卷积解码为 logits，再上采样到输入分辨率。

    与前向相关的张量形状见 forward 方法内注释。
    """

    def __init__(self, clip_model, feat_channels: int = 256, clip_text_trainable: bool = False):
        super().__init__()
        self.clip_model = clip_model  # 预训练 CLIP，forward 中仅用 encode_text
        self.clip_text_trainable = clip_text_trainable  # True 时对 encode_text 反传（需解冻部分 CLIP 层）
        self.image_encoder = TinyImageEncoder(out_channels=feat_channels)  # 图像分支与 CLIP 视觉塔独立
        self.text_proj = nn.Linear(512, feat_channels)  # CLIP 文本特征维 512 → 与图像通道对齐
        self.decoder = nn.Sequential(  # 将融合特征解码为单通道 logits
            nn.Conv2d(feat_channels + 1, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        if self.clip_text_trainable:
            text_feat = self.clip_model.encode_text(text_tokens)
        else:
            with torch.no_grad():  # CLIP 全冻结，不参与梯度
                text_feat = self.clip_model.encode_text(text_tokens)
        text_feat = text_feat.float()
        text_feat = self.text_proj(text_feat)  # 可训练投影，适配分割特征维
        text_feat = F.normalize(text_feat, dim=-1)  # 与归一化后的像素特征做点积等价于余弦相似度
        return text_feat

    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        image_feat = self.image_encoder(images)  # [B, C, h, w] 低分辨率特征图
        b, c, h, w = image_feat.shape
        dense = F.normalize(image_feat, dim=1)  # 每像素 C 维向量 L2 归一化

        text_feat = self.encode_text(text_tokens)  # [B, C] 全局句子向量
        text_feat = text_feat.view(b, c, 1, 1)  # 广播到空间维

        sim = (dense * text_feat).sum(dim=1, keepdim=True)  # 像素-文本余弦相似度图 [B,1,h,w]
        fused = torch.cat([image_feat, sim], dim=1)  # 拼接低级视觉特征与语义相关图
        logits_small = self.decoder(fused)  # 在低分辨率上预测，减少计算量
        logits = F.interpolate(logits_small, size=images.shape[-2:], mode="bilinear", align_corners=False)  # 上采样到输入图尺寸，与 GT mask 对齐
        return logits
