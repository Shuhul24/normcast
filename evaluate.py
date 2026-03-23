import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data

from dataset import KITTIRangeViewDataset
from model   import RangeViewFlowModel
from utils   import load_checkpoint, range_l1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',       type=pathlib.Path, required=True)
    p.add_argument('--ckpt',       type=str,          required=True)
    p.add_argument('--outdir',     type=pathlib.Path, default=pathlib.Path('eval_results'))
    p.add_argument('--test_seqs',  nargs=2, type=int, default=[8, 11],
                   metavar=('START', 'END'))
    # Range image
    p.add_argument('--img_h',      type=int,   default=64)
    p.add_argument('--img_w',      type=int,   default=2048)
    p.add_argument('--fov_up',     type=float, default=3.0)
    p.add_argument('--fov_down',   type=float, default=-25.0)
    p.add_argument('--depth_mean', type=float, default=12.0)
    p.add_argument('--depth_std',  type=float, default=12.0)
    # Model (must match training)
    p.add_argument('--condition_frames',  type=int, default=4)
    p.add_argument('--future_frames',     type=int, default=5)
    p.add_argument('--patch_h',           type=int, default=4)
    p.add_argument('--patch_w',           type=int, default=32)
    p.add_argument('--channels',          type=int, default=512)
    p.add_argument('--num_blocks',        type=int, default=4)
    p.add_argument('--layers_per_block',  type=int, default=4)
    p.add_argument('--context_layers',    type=int, default=2)
    p.add_argument('--head_dim',          type=int, default=64)
    # Sampling
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--num_vis',     type=int,   default=20)
    p.add_argument('--stride',      type=int,   default=1)
    return p.parse_args()


def _depth_map(img: torch.Tensor, depth_mean: float, depth_std: float) -> np.ndarray:
    depth = img[0].cpu().float().numpy() * depth_std + depth_mean
    return np.clip(depth / 50.0, 0.0, 1.0)


def _save_comparison(path: pathlib.Path, gt: torch.Tensor, pred: torch.Tensor,
                     depth_mean: float, depth_std: float) -> None:
    F = gt.size(0)
    fig, axes = plt.subplots(2, F, figsize=(4 * F, 3))
    if F == 1:
        axes = axes[:, None]
    for f in range(F):
        axes[0, f].imshow(_depth_map(gt[f],   depth_mean, depth_std), cmap='plasma')
        axes[0, f].set_title(f'GT t+{f+1}');   axes[0, f].axis('off')
        axes[1, f].imshow(_depth_map(pred[f], depth_mean, depth_std), cmap='plasma')
        axes[1, f].set_title(f'Pred t+{f+1}'); axes[1, f].axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.outdir.mkdir(parents=True, exist_ok=True)

    ds = KITTIRangeViewDataset(
        str(args.data),
        sequences=list(range(args.test_seqs[0], args.test_seqs[1])),
        condition_frames=args.condition_frames,
        future_frames=args.future_frames,
        h=args.img_h, w=args.img_w,
        fov_up=args.fov_up, fov_down=args.fov_down,
        depth_mean=args.depth_mean, depth_std=args.depth_std,
        stride=args.stride,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

    model = RangeViewFlowModel(
        in_channels=2,
        img_h=args.img_h,           img_w=args.img_w,
        patch_h=args.patch_h,       patch_w=args.patch_w,
        channels=args.channels,
        num_flow_blocks=args.num_blocks,
        layers_per_block=args.layers_per_block,
        context_layers=args.context_layers,
        head_dim=args.head_dim,
    ).to(device)
    load_checkpoint(args.ckpt, model)
    model.eval()

    all_l1: list[float] = []

    for idx, batch in enumerate(loader):
        past_frames   = batch['past_frames'].to(device)    # [1, T, 2, H, W]
        future_frames = batch['future_frames'].to(device)  # [1, F, 2, H, W]
        past_poses    = batch['past_poses'].to(device)     # [1, T, 4, 4]

        preds = model.predict_sequence(
            past_frames, past_poses, args.future_frames, args.temperature
        )  # [1, F, 2, H, W]

        l1 = range_l1(preds[0], future_frames[0], args.depth_mean, args.depth_std).item()
        all_l1.append(l1)

        if idx < args.num_vis:
            _save_comparison(
                args.outdir / f'sample_{idx:04d}.png',
                future_frames[0], preds[0],
                args.depth_mean, args.depth_std,
            )

        if (idx + 1) % 50 == 0:
            print(f'[{idx+1:4d}/{len(loader)}]  running mean L1 = {np.mean(all_l1):.4f}')

    mean_l1 = float(np.mean(all_l1))
    print(f'\nMean depth L1 over {len(all_l1)} samples: {mean_l1:.4f}')

    with open(args.outdir / 'metrics.txt', 'w') as f:
        f.write(f'mean_depth_l1: {mean_l1:.6f}\n')
        for i, v in enumerate(all_l1):
            f.write(f'{i},{v:.6f}\n')


if __name__ == '__main__':
    main()
