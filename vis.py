"""
Visualization utilities for NormCast.

Generates side-by-side comparisons of ground-truth vs. predicted range-view
images and BEV point clouds for t+1 … t+K future timesteps.

Range images are upscaled in height (scale_h) so the 64-row strips are legible
without extreme zooming.  BEV overlays use a DifforeCast-inspired colour scheme:
  GT only   → steel blue
  Pred only → brick red
  Both      → purple
  Empty     → white
"""

import math
import random

import matplotlib
matplotlib.use('Agg')            # headless, no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from dataset import KITTIRangeViewDataset

# ---------------------------------------------------------------------------
# Colour palette  (DifforeCast-inspired)
# ---------------------------------------------------------------------------
_RANGE_CMAP   = plt.get_cmap('plasma')

_C_GT   = np.array([0.15, 0.47, 0.71])   # steel blue
_C_PRED = np.array([0.84, 0.15, 0.16])   # brick red
_C_BOTH = np.array([0.58, 0.20, 0.58])   # purple
_C_BG   = np.array([1.00, 1.00, 1.00])   # white

# ---------------------------------------------------------------------------
# Range-image helpers
# ---------------------------------------------------------------------------

def _depth_to_rgb(depth_norm: np.ndarray,
                  depth_mean: float, depth_std: float,
                  invalid_val: float,
                  vmin: float = 0.0, vmax: float = 60.0) -> np.ndarray:
    """Normalised depth [H, W]  →  uint8 RGB [H, W, 3].  Invalid pixels = black."""
    depth_m = depth_norm * depth_std + depth_mean
    valid   = (depth_norm != invalid_val)
    normed  = np.clip((depth_m - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb     = (_RANGE_CMAP(normed)[:, :, :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def _range_img_for_display(frame: torch.Tensor,
                            depth_mean: float, depth_std: float,
                            invalid_val: float,
                            scale_h: int = 8) -> np.ndarray:
    """
    [2, H, W] tensor → uint8 RGB [H*scale_h, W, 3].

    The height is repeated ``scale_h`` times so the thin 64-row strip fills
    a reasonable number of pixels and fine structure is visible without
    zooming in multiple times.
    """
    depth_np = frame[0].float().cpu().numpy()
    rgb      = _depth_to_rgb(depth_np, depth_mean, depth_std, invalid_val)
    return np.repeat(rgb, scale_h, axis=0)


# ---------------------------------------------------------------------------
# 3-D back-projection  (exact inverse of RangeProjection.project)
# ---------------------------------------------------------------------------

def _range_to_xyz(frame: torch.Tensor,
                  depth_mean: float, depth_std: float,
                  fov_up_deg: float, fov_down_deg: float,
                  invalid_val: float) -> np.ndarray:
    """
    Unproject a [2, H, W] range-view frame to an (N, 3) XYZ point cloud
    (valid points only, in sensor frame: x-forward, y-left, z-up).

    Inverts RangeProjection.project() exactly:
        yaw   = (2*col/W − 1) · π          [= −arctan2(y, x)]
        pitch = fov_down + (1 − row/H)·fov
        x     = depth · cos(pitch) · cos(yaw)
        y     = −depth · cos(pitch) · sin(yaw)
        z     = depth · sin(pitch)
    """
    depth_norm = frame[0].float().cpu().numpy()        # [H, W]
    H, W       = depth_norm.shape
    valid      = (depth_norm != invalid_val)
    depth_m    = depth_norm * depth_std + depth_mean   # metres

    fov_up_r  = math.radians(fov_up_deg)
    fov_dn_r  = math.radians(fov_down_deg)
    fov_r     = fov_up_r - fov_dn_r

    cols = np.arange(W, dtype=np.float32)
    rows = np.arange(H, dtype=np.float32)
    yaw_map, row_map = np.meshgrid(                    # [H, W]
        (2.0 * cols / W - 1.0) * np.pi,
        rows,
    )
    pitch_map = fov_dn_r + (1.0 - row_map / H) * fov_r   # [H, W]

    r = depth_m * np.cos(pitch_map)
    x = r *  np.cos(yaw_map)
    y = r * -np.sin(yaw_map)
    z = depth_m * np.sin(pitch_map)

    xyz = np.stack([x, y, z], axis=-1)                # [H, W, 3]
    return xyz[valid]                                  # [N, 3]


# ---------------------------------------------------------------------------
# BEV rendering
# ---------------------------------------------------------------------------

def _bev_overlay(gt_xyz: np.ndarray, pred_xyz: np.ndarray,
                 bev_range: float = 50.0,
                 resolution: float = 0.2) -> np.ndarray:
    """
    Rasterise GT and prediction into a [G, G, 3] uint8 BEV image.

    Axes: x-forward = up, y-left = right.
    Only points within [−bev_range, bev_range] × [−bev_range, bev_range]
    in x/y are shown.
    """
    G = int(2 * bev_range / resolution)

    def _to_grid(xyz: np.ndarray) -> np.ndarray:
        mask = (
            (xyz[:, 0] > -bev_range) & (xyz[:, 0] < bev_range) &
            (xyz[:, 1] > -bev_range) & (xyz[:, 1] < bev_range)
        )
        pts = xyz[mask]
        # col = y  (left→right),  row = x  (forward→up, so we flip)
        ix  = np.clip(((pts[:, 1] + bev_range) / resolution).astype(int), 0, G - 1)
        iy  = np.clip(((bev_range - pts[:, 0]) / resolution).astype(int), 0, G - 1)
        g   = np.zeros((G, G), dtype=bool)
        g[iy, ix] = True
        return g

    g_gt   = _to_grid(gt_xyz)
    g_pred = _to_grid(pred_xyz)

    img = np.ones((G, G, 3), dtype=np.float32)        # white background
    img[g_gt   & ~g_pred] = _C_GT
    img[g_pred & ~g_gt  ] = _C_PRED
    img[g_gt   &  g_pred] = _C_BOTH
    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def visualize_predictions(
    model,
    val_ds,
    args,
    epoch: int,
    device,
    writer=None,
    use_wandb: bool = False,
    global_step: int = 0,
    num_future: int = 3,
    bev_range:   float = 50.0,
    scale_h:     int   = 8,
    temperature: float = 0.7,
):
    """
    Pick a random validation sequence, run autoregressive prediction for
    ``num_future`` steps, and save a figure with range-view and BEV comparisons.

    Layout per figure  (num_future rows × 3 columns):
        col 0 — GT range image      (plasma colourmap, height ×scale_h)
        col 1 — Predicted range image
        col 2 — BEV overlay         (GT blue / Pred red / Both purple)

    Output:  <args.logdir>/vis/epoch_<NNN>.png
    Also logged to TensorBoard and W&B when writers are provided.
    """
    import pathlib

    vis_dir = pathlib.Path(args.logdir) / 'vis'
    vis_dir.mkdir(parents=True, exist_ok=True)

    invalid_val = -args.depth_mean / args.depth_std

    # ------------------------------------------------------------------
    # Build a tiny vis dataset with future_frames = num_future so we get
    # num_future consecutive GT frames in a single sample.
    # ------------------------------------------------------------------
    vis_ds = KITTIRangeViewDataset(
        str(args.data),
        sequences=list(range(args.val_seqs[0], args.val_seqs[1])),
        condition_frames=args.condition_frames,
        future_frames=num_future,
        h=args.img_h, w=args.img_w,
        fov_up=args.fov_up, fov_down=args.fov_down,
        depth_mean=args.depth_mean, depth_std=args.depth_std,
        stride=args.stride * 5,
    )

    if len(vis_ds) == 0:
        print('[vis] visualisation dataset is empty — skipping')
        return

    model.eval()
    idx    = random.randint(0, len(vis_ds) - 1)
    sample = vis_ds[idx]

    past_frames  = sample['past_frames' ].unsqueeze(0).to(device)  # [1, T, 2, H, W]
    past_poses   = sample['past_poses'  ].unsqueeze(0).to(device)  # [1, T, 4, 4]
    gt_futures   = sample['future_frames'].cpu()                    # [num_future, 2, H, W]

    # ------------------------------------------------------------------
    # Autoregressive prediction
    # ------------------------------------------------------------------
    with torch.no_grad():
        pred_futures = model.predict_sequence(
            past_frames, past_poses,
            num_future=num_future, temperature=temperature,
        )[0].cpu()                                                  # [num_future, 2, H, W]

    # ------------------------------------------------------------------
    # Figure  (num_future rows × 3 cols)
    # ------------------------------------------------------------------
    # Column widths: range imgs are W-wide, BEV is square (bev_px × bev_px).
    bev_px   = int(2 * bev_range / 0.2)         # 500
    range_px = args.img_w                        # 2048

    # width_ratios so subplots scale proportionally
    col_ratio = [range_px, range_px, bev_px]
    total_w   = sum(col_ratio)

    # target figure width in inches (cap at 30 so files stay manageable)
    fig_w_in  = min(total_w / 100, 30.0)
    # each range image row: height = img_h * scale_h pixels → inches
    row_h_in  = max((args.img_h * scale_h) / 100, (bev_px / 100))
    fig_h_in  = row_h_in * num_future + 0.6     # +0.6 for suptitle / legend

    fig, axes = plt.subplots(
        num_future, 3,
        figsize=(fig_w_in, fig_h_in),
        dpi=150,
        gridspec_kw={'width_ratios': col_ratio, 'hspace': 0.35, 'wspace': 0.05},
    )
    if num_future == 1:
        axes = axes[np.newaxis, :]     # ensure 2-D indexing

    fig.patch.set_facecolor('white')

    for t in range(num_future):
        gt_f   = gt_futures[t]          # [2, H, W]
        pred_f = pred_futures[t]        # [2, H, W]

        # ---- range images ----
        gt_rgb   = _range_img_for_display(gt_f,   args.depth_mean,
                                          args.depth_std, invalid_val, scale_h)
        pred_rgb = _range_img_for_display(pred_f, args.depth_mean,
                                          args.depth_std, invalid_val, scale_h)

        for ax, img, label in [
            (axes[t, 0], gt_rgb,   f'GT   t+{t+1}'),
            (axes[t, 1], pred_rgb, f'Pred  t+{t+1}'),
        ]:
            ax.imshow(img, aspect='auto', interpolation='nearest')
            ax.set_title(label, fontsize=8, pad=2, loc='left')
            ax.axis('off')

        # ---- BEV overlay ----
        gt_xyz   = _range_to_xyz(gt_f,   args.depth_mean, args.depth_std,
                                  args.fov_up, args.fov_down, invalid_val)
        pred_xyz = _range_to_xyz(pred_f, args.depth_mean, args.depth_std,
                                  args.fov_up, args.fov_down, invalid_val)
        bev      = _bev_overlay(gt_xyz, pred_xyz, bev_range=bev_range)

        ax_bev = axes[t, 2]
        ax_bev.imshow(bev, origin='upper', interpolation='nearest',
                      extent=[-bev_range, bev_range, -bev_range, bev_range])
        ax_bev.set_title(f'BEV t+{t+1}', fontsize=8, pad=2, loc='left')
        ax_bev.set_xlabel('y (m)', fontsize=6, labelpad=1)
        ax_bev.set_ylabel('x (m)', fontsize=6, labelpad=1)
        ax_bev.tick_params(labelsize=6)

    # ---- shared legend for BEV ----
    legend_handles = [
        mpatches.Patch(color=_C_GT,   label='GT only'),
        mpatches.Patch(color=_C_PRED, label='Pred only'),
        mpatches.Patch(color=_C_BOTH, label='Both'),
    ]
    fig.legend(handles=legend_handles,
               loc='lower right', bbox_to_anchor=(1.0, 0.0),
               fontsize=7, frameon=True, ncol=3)

    fig.suptitle(
        f'Epoch {epoch}  |  val sample #{idx}  '
        f'(seq {args.val_seqs[0]}–{args.val_seqs[1]-1})',
        fontsize=10, y=1.01,
    )

    out_path = vis_dir / f'epoch_{epoch:03d}.png'
    fig.savefig(out_path, bbox_inches='tight', facecolor='white', dpi=150)
    plt.close(fig)

    # ---- TensorBoard ----
    if writer is not None:
        try:
            from PIL import Image as _PIL
            img_t = torch.from_numpy(
                np.array(_PIL.open(out_path).convert('RGB'))
            ).permute(2, 0, 1)
            writer.add_image('val/prediction', img_t, global_step=global_step)
        except Exception as e:
            print(f'[vis] TensorBoard image logging failed: {e}')

    # ---- W&B ----
    if use_wandb:
        try:
            import wandb
            wandb.log({'val/prediction': wandb.Image(str(out_path))},
                      step=global_step)
        except Exception as e:
            print(f'[vis] W&B image logging failed: {e}')

    print(f'[vis] epoch {epoch:03d} → {out_path}')
    return out_path
