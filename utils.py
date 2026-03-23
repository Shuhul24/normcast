import math
import torch
import torch.optim as optim


# ------------------------------------------------------------------
# Learning-rate schedule
# ------------------------------------------------------------------

def cosine_schedule_with_warmup(optimizer: optim.Optimizer,
                                warmup_steps: int,
                                total_steps: int,
                                min_lr: float,
                                max_lr: float) -> optim.lr_scheduler.LambdaLR:
    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return min_lr / max_lr + step / max(1, warmup_steps) * (1.0 - min_lr / max_lr)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr / max_lr + 0.5 * (1.0 - min_lr / max_lr) * (1.0 + math.cos(math.pi * t))

    return optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


# ------------------------------------------------------------------
# Checkpointing
# ------------------------------------------------------------------

def save_checkpoint(path: str, model: torch.nn.Module,
                    optimizer: optim.Optimizer, epoch: int, step: int) -> None:
    torch.save({
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch':     epoch,
        'step':      step,
    }, path)


def load_checkpoint(path: str, model: torch.nn.Module,
                    optimizer: optim.Optimizer | None = None) -> tuple[int, int]:
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt.get('epoch', 0), ckpt.get('step', 0)


# ------------------------------------------------------------------
# Losses
# ------------------------------------------------------------------

def range_l1(pred: torch.Tensor, target: torch.Tensor,
             depth_mean: float = 12.0, depth_std: float = 12.0) -> torch.Tensor:
    """L1 on the depth channel (channel 0), skipping zero-depth GT pixels."""
    mask = (target[:, 0] != -depth_mean / depth_std).float()   # non-empty pixels
    diff = (pred[:, 0] - target[:, 0]).abs() * mask
    return diff.sum() / (mask.sum() + 1e-6)
