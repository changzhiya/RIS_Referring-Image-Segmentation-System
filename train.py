import argparse
import json
import os
import sys
from datetime import datetime

import clip
import torch
import torch.nn.functional as F
from torch import amp as torch_amp
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.refcoco_dataset import RefCOCOIndexDataset
from models.clip_ris import ClipTextGuidedRIS
from models.clip_ris_v33 import ClipRISV33
from utils.clip_finetune import unfreeze_clip_text_last_blocks
from utils.metrics import accumulate_miou_ciou, binary_batch_mean_iou

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 固定默认输出目录，避免与历史实验混写；可仍用 --save-dir 覆盖
_DEFAULT_SAVE_DIR = os.path.join(_SCRIPT_DIR, "result", "checkpoints_active")
_DEFAULT_SAVE_DIR_DOC31 = os.path.join(_SCRIPT_DIR, "result", "checkpoints_doc31")


def _max_micro_batches_per_epoch(loader) -> int:
    return len(loader)


def _max_optimizer_steps_per_epoch(total_micro_batches: int, accum: int) -> int:
    if accum <= 0:
        return total_micro_batches
    return (total_micro_batches + accum - 1) // accum


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)  # 将分割 logits 转为概率，与 BCE 配合稳定训练
    num = 2 * (probs * targets).sum(dim=(1, 2, 3))  # Dice 分子：预测与 GT 的重叠部分
    den = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps  # 分母：并集近似（带平滑项 eps）
    return 1 - (num / den).mean()  # 最小化该式等价于最大化 Dice 系数


def soft_iou_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """可微 IoU 损失：1 - mean_soft_IoU（文档 3.1.1 中 iou_loss 项）。"""
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    return 1.0 - iou.mean()


def _autocast_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _default_num_workers() -> int:
    # Windows 上 DataLoader 多进程 + CUDA 易偶发卡死；主进程单线程读数据更稳（略慢）
    return 0 if sys.platform == "win32" else 2


def run_train_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler,
    use_amp: bool,
    grad_accum_steps: int,
    max_norm: float = 1.0,
    *,
    epoch: int = 1,
    num_epochs: int = 1,
    progress_log_every: int = 0,
    show_batch_progress: bool = True,
    loss_w_bce: float = 1.0,
    loss_w_dice: float = 1.0,
    loss_w_iou: float = 0.0,
    empty_cache_every: int = 0,
):
    """低显存训练：AMP + 梯度累积；loss 按 micro-step 缩放，末尾不足一整步时校正梯度尺度。"""
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)
    micro_in_step = 0
    optimizer_steps = 0

    total_batches = _max_micro_batches_per_epoch(loader)
    max_opt_steps = _max_optimizer_steps_per_epoch(total_batches, grad_accum_steps)
    if show_batch_progress:
        pbar = tqdm(
            loader,
            total=total_batches,
            ncols=120,
            leave=False,
            desc=f"Train ep{epoch}/{num_epochs}",
            mininterval=1.0,
            file=sys.stdout,
            unit="mb",
            bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} mb {postfix}",
        )
        iterator = pbar
    else:
        pbar = None
        iterator = loader

    for images, masks, text_tokens, _ in iterator:
        images = images.to(device, non_blocking=(device == "cuda"))
        masks = masks.to(device, non_blocking=(device == "cuda"))
        text_tokens = text_tokens.to(device, non_blocking=(device == "cuda"))

        if use_amp:
            with torch_amp.autocast("cuda", dtype=_autocast_dtype()):
                logits = model(images, text_tokens)
            logits = logits.float()
            masks_f = masks.float()
            bce = F.binary_cross_entropy_with_logits(logits, masks_f)
            dloss = dice_loss(logits, masks_f)
            iou_l = soft_iou_loss(logits, masks_f) if loss_w_iou > 0 else bce * 0.0
            loss_total = loss_w_bce * bce + loss_w_dice * dloss + loss_w_iou * iou_l
            loss = loss_total / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            logits = model(images, text_tokens)
            bce = F.binary_cross_entropy_with_logits(logits, masks)
            dloss = dice_loss(logits, masks)
            iou_l = soft_iou_loss(logits, masks) if loss_w_iou > 0 else bce * 0.0
            loss_total = loss_w_bce * bce + loss_w_dice * dloss + loss_w_iou * iou_l
            loss = loss_total / grad_accum_steps
            loss.backward()

        with torch.no_grad():
            running_loss += loss_total.detach().float().item()
            running_iou += binary_batch_mean_iou(logits.float(), masks).item()
        n_batches += 1
        micro_in_step += 1

        if empty_cache_every > 0 and device == "cuda" and n_batches % empty_cache_every == 0:
            torch.cuda.empty_cache()

        if progress_log_every > 0 and n_batches % progress_log_every == 0:
            print(
                f"[Train] epoch {epoch}/{num_epochs} batch {n_batches}/{total_batches} "
                f"loss_ma={running_loss / n_batches:.4f} iou_ma={running_iou / n_batches:.4f}",
                flush=True,
            )

        if micro_in_step >= grad_accum_steps:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro_in_step = 0
            optimizer_steps += 1
        if pbar is not None:
            pbar.set_postfix_str(f"opt {optimizer_steps}/{max_opt_steps}", refresh=False)

    if micro_in_step > 0:
        scale = grad_accum_steps / micro_in_step
        if use_amp:
            scaler.unscale_(optimizer)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        if pbar is not None:
            pbar.set_postfix_str(f"opt {optimizer_steps}/{max_opt_steps}", refresh=False)

    if n_batches == 0:
        return 0.0, 0.0
    return running_loss / n_batches, running_iou / n_batches


@torch.no_grad()
def run_val_epoch(
    model,
    loader,
    device,
    use_amp: bool = False,
    *,
    epoch: int = 1,
    num_epochs: int = 1,
    progress_log_every: int = 0,
    show_batch_progress: bool = True,
    loss_w_bce: float = 1.0,
    loss_w_dice: float = 1.0,
    loss_w_iou: float = 0.0,
):
    """验证集：返回 mean loss、mIoU（逐样本 IoU 平均）、cIoU（全集累积交并比）。"""
    model.eval()
    running_loss = 0.0
    total_per_iou_sum = 0.0
    total_n_samples = 0
    total_inter = 0.0
    total_union = 0.0
    eps = 1e-6

    total_batches = len(loader)
    n_batches = 0
    if show_batch_progress:
        iterator = tqdm(
            loader,
            total=total_batches,
            ncols=120,
            leave=False,
            desc=f"Val ep{epoch}/{num_epochs}",
            mininterval=1.0,
            file=sys.stdout,
            unit="mb",
            bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} mb",
        )
    else:
        iterator = loader

    for images, masks, text_tokens, _ in iterator:
        images = images.to(device, non_blocking=(device == "cuda"))
        masks = masks.to(device, non_blocking=(device == "cuda"))
        text_tokens = text_tokens.to(device, non_blocking=(device == "cuda"))
        if use_amp:
            with torch_amp.autocast("cuda", dtype=_autocast_dtype()):
                logits = model(images, text_tokens)
        else:
            logits = model(images, text_tokens)
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, masks)
        dloss = dice_loss(logits, masks)
        iou_l = soft_iou_loss(logits, masks) if loss_w_iou > 0 else bce * 0.0
        loss = (loss_w_bce * bce + loss_w_dice * dloss + loss_w_iou * iou_l).item()
        running_loss += loss

        per_sum, n_samp, inter_b, union_b = accumulate_miou_ciou(logits, masks, eps)
        total_per_iou_sum += per_sum.item()
        total_n_samples += n_samp
        total_inter += inter_b.item()
        total_union += union_b.item()
        n_batches += 1
        if progress_log_every > 0 and n_batches % progress_log_every == 0:
            print(
                f"[Val] epoch {epoch}/{num_epochs} batch {n_batches}/{total_batches} "
                f"loss_ma={running_loss / n_batches:.4f}",
                flush=True,
            )

    if n_batches == 0 or total_n_samples == 0:
        return 0.0, 0.0, 0.0
    va_loss = running_loss / n_batches
    va_miou = total_per_iou_sum / total_n_samples
    va_ciou = total_inter / (total_union + eps)
    return va_loss, va_miou, va_ciou


def append_val_metrics_jsonl(path: str, record: dict) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _apply_doc_cuda_env() -> None:
    """文档「前置显存极限优化」：TF32、cudnn、分配器提示（不保证所有驱动均支持 memory_fraction）。"""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
        try:
            torch.cuda.set_per_process_memory_fraction(0.95, 0)
        except Exception:
            pass


def build_args():
    parser = argparse.ArgumentParser("Minimal RefCOCO RIS training")
    parser.add_argument("--data-root", type=str, required=True, help="Root dir containing images/masks folders")
    parser.add_argument("--train-index", type=str, required=True, help="Train split index JSON path")
    parser.add_argument("--val-index", type=str, required=True, help="Val split index JSON path")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=_DEFAULT_SAVE_DIR,
        help=f"默认固定为 {_DEFAULT_SAVE_DIR}（可用本参数覆盖）",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2, help="Micro-batch for low VRAM (use grad accum)")
    parser.add_argument("--grad-accum-steps", type=int, default=8, help="Effective batch ≈ batch_size * this")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=_default_num_workers(),
        help="DataLoader 进程数。Windows 默认 0（避免多进程+CUDA 偶发卡死）；Linux 默认 2",
    )
    parser.add_argument(
        "--progress-log-every",
        type=int,
        default=2000,
        help="每 N 个 micro-batch 打印一行 [Train]/[Val] 进度；数值越大终端越少刷屏；0 关闭",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="LR for head (image encoder, decoder, text_proj)")
    parser.add_argument("--lr-clip", type=float, default=5e-6, help="LR for unfrozen CLIP text layers")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="输入边长；显存不足可改为 224（须与 eval 时 --image-size 一致）",
    )
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument(
        "--clip-unfreeze-last",
        type=int,
        default=1,
        help="Unfreeze last N text transformer blocks + ln_final (0 = fully frozen CLIP text)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA AMP (BatchNorm+accum 易触发 GradScaler 问题，默认关闭；显存吃紧时优先用低分辨率+累积)",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path for resuming training")
    parser.add_argument(
        "--pause-between-epochs",
        action="store_true",
        help="Before/after each full epoch, wait for Enter (use interactive terminal, not piped/background)",
    )
    parser.add_argument(
        "--metrics-log",
        type=str,
        default="",
        help="Append per-epoch val mIoU/cIoU as JSON lines (default: <save-dir>/val_metrics.jsonl)",
    )
    parser.add_argument(
        "--append-metrics",
        action="store_true",
        help="即使未 --resume 也追加写入 val_metrics（默认：非续训时清空该文件）",
    )
    parser.add_argument(
        "--no-batch-progress",
        action="store_true",
        help="不显示 train/val 的 tqdm 与 [Train]/[Val] 中间日志；每个 epoch 结束仍打印一行汇总",
    )
    parser.add_argument(
        "--doc-stage1",
        action="store_true",
        help="启用 project.txt 3.1.1–3.1.3：224、accum=4、AMP、AdamW 5e-5 & wd=1e-4、余弦调度、BCE+Dice+softIoU 0.5/0.3/0.2、"
        "CLIP 文本末 2 层、训练集同步增强、val batch=1、默认 save-dir=result/checkpoints_doc31",
    )
    parser.add_argument(
        "--preset-early-val-miou",
        action="store_true",
        help="面向前 10 个 epoch 抬高 val_mIoU/cIoU：256²、损失 0.15·BCE+0.45·Dice+0.40·softIoU、"
        "RandomResizedCrop scale 0.88–1、head_lr=8e-5 clip_lr=2e-5、CLIP 仅末 1 层、accum=4+AMP、"
        "余弦 T_max=epochs；未指定 --save-dir 时写入 result/checkpoints_miou10_<时间戳>/ 新目录。与 --doc-stage1 同时给出时以本预设为准。",
    )
    parser.add_argument(
        "--epoch-report-interval",
        type=int,
        default=1,
        help="每 N 个 epoch 在终端打印一行带时间戳的验证汇总（1=每个 epoch；metrics 仍每轮写入 val_metrics.jsonl）",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=0,
        help="验证 DataLoader batch；0 表示与训练 micro_batch 相同（文档建议省显存可用 1）",
    )
    parser.add_argument("--loss-w-bce", type=float, default=-1.0, help="<0 时用预设；否则 BCE 权重")
    parser.add_argument("--loss-w-dice", type=float, default=-1.0, help="<0 时用预设；否则 Dice 权重")
    parser.add_argument("--loss-w-iou", type=float, default=-1.0, help="<0 时用预设；否则 soft-IoU 权重")
    parser.add_argument(
        "--empty-cache-every",
        type=int,
        default=0,
        help="每 N 个 micro-batch 调用 torch.cuda.empty_cache()；0 关闭（文档 3.1.1 为 100，略拖速）",
    )
    parser.add_argument(
        "--ris-arch",
        type=str,
        default="baseline",
        choices=("baseline", "v33"),
        help="分割头结构：baseline=原 ClipTextGuidedRIS；v33=one.docx§3.3（空间+token 跨模态），须单独训练，与旧 .pt 不兼容",
    )
    return parser.parse_args()


def _pause_between_epochs(message: str, enabled: bool) -> None:
    if enabled:
        input(message)


def main():
    args = build_args()

    if args.preset_early_val_miou and args.doc_stage1:
        print(
            "[preset-early-val-miou] 与 --doc-stage1 同时指定：以本预设为准（256、不同 LR/损失/CLIP 层数）。",
            flush=True,
        )

    if getattr(args, "ris_arch", "baseline") == "v33" and args.save_dir == _DEFAULT_SAVE_DIR and not args.resume:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = os.path.join(_SCRIPT_DIR, "result", f"checkpoints_v33_{stamp}")
        print(f"[ris-arch=v33] 使用独立 save-dir: {args.save_dir}", flush=True)

    if args.preset_early_val_miou:
        _apply_doc_cuda_env()
        args.image_size = 256
        args.batch_size = 2
        args.grad_accum_steps = 4
        args.lr = 8e-5
        args.lr_clip = 2e-5
        args.weight_decay = 1e-4
        args.clip_unfreeze_last = 1
        args.amp = True
        args.loss_w_bce = 0.15
        args.loss_w_dice = 0.45
        args.loss_w_iou = 0.40
        if args.save_dir == _DEFAULT_SAVE_DIR:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.save_dir = os.path.join(_SCRIPT_DIR, "result", f"checkpoints_miou10_{stamp}")
        if args.val_batch_size == 0:
            args.val_batch_size = 1
        if args.empty_cache_every == 0:
            args.empty_cache_every = 100
        if not args.resume:
            args.epochs = max(args.epochs, 10)
        print(
            "[preset-early-val-miou] 256x256, accum=4, AMP, AdamW head=8e-5 clip=2e-5 wd=1e-4, cosine T_max=epochs, "
            "loss 0.15*BCE+0.45*Dice+0.40*softIoU, CLIP text 1 block, aug crop scale 0.88-1, val_batch=1",
            flush=True,
        )
        print(f"[preset-early-val-miou] save-dir: {args.save_dir}", flush=True)
    elif args.doc_stage1:
        _apply_doc_cuda_env()
        args.image_size = 224
        args.batch_size = 2
        args.grad_accum_steps = 4
        args.lr = 5e-5
        args.lr_clip = 5e-5
        args.weight_decay = 1e-4
        args.clip_unfreeze_last = 2
        args.amp = True
        if args.save_dir == _DEFAULT_SAVE_DIR:
            args.save_dir = _DEFAULT_SAVE_DIR_DOC31
        if args.val_batch_size == 0:
            args.val_batch_size = 1
        if args.empty_cache_every == 0:
            args.empty_cache_every = 100
        if not args.resume:
            args.epochs = max(args.epochs, 10)
        print(
            "[doc-stage1] 224x224, accum=4, AMP, AdamW 5e-5 wd=1e-4, cosine T_max=epochs, "
            "loss 0.5*BCE+0.3*Dice+0.2*softIoU, CLIP text 2 blocks, train aug, val_batch=1",
            flush=True,
        )

    segment_cosine_resume = bool(args.doc_stage1 or args.preset_early_val_miou)

    if args.loss_w_bce >= 0 and args.loss_w_dice >= 0 and args.loss_w_iou >= 0:
        loss_w_bce, loss_w_dice, loss_w_iou = args.loss_w_bce, args.loss_w_dice, args.loss_w_iou
    elif args.doc_stage1:
        loss_w_bce, loss_w_dice, loss_w_iou = 0.5, 0.3, 0.2
    else:
        loss_w_bce, loss_w_dice, loss_w_iou = 1.0, 1.0, 0.0

    os.makedirs(args.save_dir, exist_ok=True)
    show_bars = not args.no_batch_progress
    progress_log_every = 0 if args.no_batch_progress else args.progress_log_every

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda" and args.amp
    print(
        f"Using device: {device}, AMP={use_amp}, image_size={args.image_size}, "
        f"micro_batch={args.batch_size}, accum={args.grad_accum_steps} (eff_batch≈{args.batch_size * args.grad_accum_steps})",
        flush=True,
    )

    clip_model, _ = clip.load(args.clip_model, device=device)  # 加载预训练 CLIP（此处只用文本编码分支）
    # CUDA 上 CLIP 的 transformer 权重常为 float16；若解冻并用 AdamW 更新，一步即可把权重更新成 NaN，须先统一到 float32
    clip_model = clip_model.float()
    clip_trainable = args.clip_unfreeze_last > 0
    if clip_trainable:
        unfreeze_clip_text_last_blocks(clip_model, args.clip_unfreeze_last)
        print(
            f"CLIP text: unfrozen last {args.clip_unfreeze_last} transformer block(s) + ln_final",
            flush=True,
        )
    else:
        for p in clip_model.parameters():
            p.requires_grad = False

    if args.ris_arch == "v33":
        model = ClipRISV33(
            clip_model=clip_model,
            clip_text_trainable=clip_trainable,
        ).to(device)
        print("[ris-arch] ClipRISV33 (one.docx §3.3: spatial + token cross-attn)", flush=True)
    else:
        model = ClipTextGuidedRIS(
            clip_model=clip_model,
            clip_text_trainable=clip_trainable,
        ).to(device)

    train_augment = bool(args.doc_stage1 or args.preset_early_val_miou)
    aug_crop_scale = (0.88, 1.0) if args.preset_early_val_miou else None
    train_ds = RefCOCOIndexDataset(
        args.train_index,
        root_dir=args.data_root,
        image_size=args.image_size,
        augment_train=train_augment,
        aug_crop_scale=aug_crop_scale,
    )
    val_ds = RefCOCOIndexDataset(
        args.val_index,
        root_dir=args.data_root,
        image_size=args.image_size,
        augment_train=False,
    )
    pin = device == "cuda" and args.num_workers > 0
    val_bs = args.val_batch_size if args.val_batch_size > 0 else args.batch_size
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )

    head_params = []
    clip_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("clip_model."):
            clip_params.append(p)
        else:
            head_params.append(p)
    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay})
    if clip_params:
        param_groups.append({"params": clip_params, "lr": args.lr_clip, "weight_decay": args.weight_decay})
    if not param_groups:
        raise RuntimeError("No trainable parameters")
    optimizer = torch.optim.AdamW(param_groups)
    scaler = torch_amp.GradScaler("cuda", enabled=use_amp) if device == "cuda" else None
    # doc_stage1 + --resume: scheduler built after start_epoch so T_max matches remaining epochs
    # (avoids Cosine state mismatch when extending total epochs, e.g. 10 -> 15).
    scheduler = None
    if segment_cosine_resume and not args.resume:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    best_val_miou = -1.0
    best_val_ciou = -1.0  # 与 best miou 同一 epoch 上的 cIoU，仅作记录
    start_epoch = 1
    metrics_log_path = args.metrics_log or os.path.join(args.save_dir, "val_metrics.jsonl")
    if not args.resume and not args.append_metrics:
        os.makedirs(os.path.dirname(os.path.abspath(metrics_log_path)) or ".", exist_ok=True)
        with open(metrics_log_path, "w", encoding="utf-8"):
            pass
        print(f"Fresh metrics log (truncated): {metrics_log_path}", flush=True)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except (ValueError, RuntimeError):
                print("Warning: optimizer state not loaded (param groups may differ).")
        if scaler is not None and ckpt.get("scaler") is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                print(f"Warning: scaler state not loaded: {e}")
        if scheduler is not None and ckpt.get("scheduler") is not None and not segment_cosine_resume:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except (ValueError, RuntimeError) as e:
                print(f"Warning: scheduler state not loaded: {e}")
        best_val_miou = ckpt.get("best_val_miou", ckpt.get("best_val_iou", -1.0))
        best_val_ciou = ckpt.get("best_val_ciou", -1.0)
        start_epoch = ckpt.get("epoch", 0) + 1  # ckpt 里存的是已完成的 epoch，下次从 +1 开始
        print(f"Resumed from: {args.resume}, next epoch: {start_epoch}", flush=True)

    if segment_cosine_resume and args.resume:
        rem = max(1, args.epochs - start_epoch + 1)
        scheduler = CosineAnnealingLR(optimizer, T_max=rem, eta_min=1e-6)
        print(
            f"[scheduler] CosineAnnealingLR T_max={rem} (remaining epochs; LR from optimizer state)",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        _pause_between_epochs(
            f"[Pause] Before epoch {epoch}/{args.epochs}. Press Enter to start this epoch... ",
            args.pause_between_epochs,
        )
        tr_loss, tr_iou = run_train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            use_amp,
            args.grad_accum_steps,
            max_norm=args.max_grad_norm,
            epoch=epoch,
            num_epochs=args.epochs,
            progress_log_every=progress_log_every,
            show_batch_progress=show_bars,
            loss_w_bce=loss_w_bce,
            loss_w_dice=loss_w_dice,
            loss_w_iou=loss_w_iou,
            empty_cache_every=args.empty_cache_every,
        )
        va_loss, va_miou, va_ciou = run_val_epoch(
            model,
            val_loader,
            device,
            use_amp=use_amp,
            epoch=epoch,
            num_epochs=args.epochs,
            progress_log_every=progress_log_every,
            show_batch_progress=show_bars,
            loss_w_bce=loss_w_bce,
            loss_w_dice=loss_w_dice,
            loss_w_iou=loss_w_iou,
        )

        if scheduler is not None:
            scheduler.step()
            lr_head = optimizer.param_groups[0]["lr"]
        else:
            lr_head = optimizer.param_groups[0]["lr"]

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        do_report = (
            args.epoch_report_interval <= 1
            or (epoch % args.epoch_report_interval == 0)
            or (epoch == args.epochs)
        )
        if do_report:
            if scheduler is not None:
                print(
                    f"[{ts}] Epoch {epoch:02d}/{args.epochs} | "
                    f"train_loss={tr_loss:.4f} train_iou={tr_iou:.4f} | "
                    f"val_loss={va_loss:.4f} val_mIoU={va_miou:.4f} val_cIoU={va_ciou:.4f} | lr={lr_head:.2e}",
                    flush=True,
                )
            else:
                print(
                    f"[{ts}] Epoch {epoch:02d}/{args.epochs} | "
                    f"train_loss={tr_loss:.4f} train_iou={tr_iou:.4f} | "
                    f"val_loss={va_loss:.4f} val_mIoU={va_miou:.4f} val_cIoU={va_ciou:.4f}",
                    flush=True,
                )
        else:
            print(
                f"[{ts}] Epoch {epoch:02d}/{args.epochs} done "
                f"(full metrics line every {args.epoch_report_interval} ep; see val_metrics.jsonl)",
                flush=True,
            )

        append_val_metrics_jsonl(
            metrics_log_path,
            {
                "epoch": epoch,
                "val_loss": va_loss,
                "val_mIoU": va_miou,
                "val_cIoU": va_ciou,
                "train_loss": tr_loss,
                "train_iou_batch_mean": tr_iou,
            },
        )

        if va_miou > best_val_miou:  # 以验证集 mIoU 作为保存 best 的依据
            best_val_miou = va_miou
            best_val_ciou = va_ciou
            ckpt_path = os.path.join(args.save_dir, "best.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "args": vars(args),
                    "best_val_miou": best_val_miou,
                    "best_val_ciou": best_val_ciou,
                    "best_val_iou": best_val_miou,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(
                f"Saved best checkpoint to: {ckpt_path} (use best.pt for eval/deploy)",
                flush=True,
            )

        # 每轮结束都存 last，便于中断后从「上一轮整轮结束」处续训
        last_ckpt = os.path.join(args.save_dir, "last.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "args": vars(args),
                "best_val_miou": best_val_miou,
                "best_val_ciou": best_val_ciou,
                "best_val_iou": best_val_miou,
                "val_mIoU": va_miou,
                "val_cIoU": va_ciou,
                "epoch": epoch,
            },
            last_ckpt,
        )
        print(f"Saved latest checkpoint to: {last_ckpt}", flush=True)
        _pause_between_epochs(
            f"[Pause] After epoch {epoch}/{args.epochs}. Press Enter for next epoch or to exit if done... ",
            args.pause_between_epochs,
        )

    print(
        f"Done. Best val mIoU: {best_val_miou:.4f} (cIoU at that epoch: {best_val_ciou:.4f})",
        flush=True,
    )
    print(f"Val metrics log: {metrics_log_path}", flush=True)
    best_path = os.path.join(args.save_dir, "best.pt")
    print(f"Best checkpoint for eval: {os.path.abspath(best_path)}", flush=True)


if __name__ == "__main__":
    main()
