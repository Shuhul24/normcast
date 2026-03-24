import argparse
import math
import pathlib

import torch
import torch.utils.data
import wandb
from torch.utils.tensorboard import SummaryWriter

from dataset import KITTIRangeViewDataset
from model   import RangeViewFlowModel
from utils   import cosine_schedule_with_warmup, save_checkpoint, load_checkpoint
from vis     import visualize_predictions

# -----------------------------------------------------------------------
# Dequantization noise std (in normalised-depth units).
# Adding a small amount of noise before the forward pass prevents the model
# from finding degenerate "spike" solutions on the discretised LiDAR grid.
# Applied only to valid (non-zero-depth) pixels so the empty-sky mask is
# not corrupted.
# -----------------------------------------------------------------------
DENOISING_STD = 0.01


def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument('--data',       type=pathlib.Path, required=True,
                   help='Root of KITTI Odometry dataset')
    p.add_argument('--train_seqs', nargs=2, type=int, default=[0, 6],
                   metavar=('START', 'END'))
    p.add_argument('--val_seqs',   nargs=2, type=int, default=[6, 8],
                   metavar=('START', 'END'))
    p.add_argument('--stride',     type=int,   default=5)
    # Range image
    p.add_argument('--img_h',      type=int,   default=64)
    p.add_argument('--img_w',      type=int,   default=2048)
    p.add_argument('--fov_up',     type=float, default=3.0)
    p.add_argument('--fov_down',   type=float, default=-25.0)
    p.add_argument('--depth_mean', type=float, default=12.0)
    p.add_argument('--depth_std',  type=float, default=12.0)
    # Model
    p.add_argument('--condition_frames',  type=int, default=4)
    p.add_argument('--patch_h',           type=int, default=4)
    p.add_argument('--patch_w',           type=int, default=32)
    p.add_argument('--channels',          type=int, default=512)
    p.add_argument('--num_blocks',        type=int, default=4)
    p.add_argument('--layers_per_block',  type=int, default=4)
    p.add_argument('--context_layers',    type=int, default=2)
    p.add_argument('--head_dim',          type=int, default=64)
    # Training
    p.add_argument('--batch_size',  type=int,   default=4)
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--lr',          type=float, default=3e-4)
    p.add_argument('--num_workers', type=int,   default=4)
    p.add_argument('--log_every',   type=int,   default=50)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--loss_skip_thresh', type=float, default=2.0,
                   help='Skip a training batch (and its backward pass) when the '
                        'forward-pass NLL exceeds this value, preventing '
                        'catastrophic-loss batches from corrupting Adam state.')
    p.add_argument('--logdet_penalty_weight', type=float, default=0.0,
                   help='Weight λ for the soft logdet ceiling λ·ReLU(logdet−target). '
                        'Disabled by default: the penalty interacts with the affine '
                        'coupling scale in a way that can cause late-stage collapse '
                        '(driving z to >>1σ while keeping logdet stable).  Enable '
                        'cautiously with small values (e.g. 0.05) if logdet grows '
                        'uncontrollably past epoch ~60.')
    p.add_argument('--logdet_target', type=float, default=2.5,
                   help='Logdet soft-ceiling for the penalty term.')
    p.add_argument('--cond_dropout_p', type=float, default=0.15,
                   help='Probability of zeroing the future-frame content fed into '
                        'each FlowBlock coupling network during training.  Forces '
                        'scale/shift to be derived from cross-attention to past '
                        'context, closing the training/sampling gap where the model '
                        'could otherwise rely on the true future patches (available '
                        'at training time but not at sampling time).')
    p.add_argument('--context_lr_scale', type=float, default=1.0,
                   help='LR multiplier applied to the context encoder relative to '
                        'the flow blocks.  Values above ~1.2 risk destabilising '
                        'training: the context encoder grows faster than the '
                        'coupling layers can adapt, driving z outside N(0,1).')
    # Paths
    p.add_argument('--logdir',      type=pathlib.Path, default=pathlib.Path('runs/normcast'))
    p.add_argument('--resume',      type=str, default='')
    p.add_argument('--wandb_project', type=str, default='normcast',
                   help='W&B project name (set to empty string to disable W&B)')
    p.add_argument('--wandb_run',   type=str, default=None,
                   help='Optional W&B run name')
    # Visualisation
    p.add_argument('--vis_every',   type=int, default=10,
                   help='Save prediction visualisations every N epochs '
                        '(0 = disable).  Saved to <logdir>/vis/')
    p.add_argument('--vis_futures', type=int, default=3,
                   help='Number of future steps to visualise (t+1 … t+K)')
    p.add_argument('--sample_temp', type=float, default=0.7,
                   help='Sampling temperature used for visualisation. '
                        'Values < 1.0 reduce prediction variance; '
                        '0.7 is a good default for a model in mid-training.')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    args.logdir.mkdir(parents=True, exist_ok=True)

    # Normalised value that empty-sky / invalid LiDAR pixels are set to.
    # Used to build the validity mask that excludes these constant regions
    # from the NLL prior term.
    invalid_depth: float = -args.depth_mean / args.depth_std   # = -1.0

    # ------------------------------------------------------------------ datasets
    train_ds = KITTIRangeViewDataset(
        str(args.data),
        sequences=list(range(args.train_seqs[0], args.train_seqs[1])),
        condition_frames=args.condition_frames, future_frames=1,
        h=args.img_h, w=args.img_w,
        fov_up=args.fov_up, fov_down=args.fov_down,
        depth_mean=args.depth_mean, depth_std=args.depth_std,
        stride=args.stride,
    )
    val_ds = KITTIRangeViewDataset(
        str(args.data),
        sequences=list(range(args.val_seqs[0], args.val_seqs[1])),
        condition_frames=args.condition_frames, future_frames=1,
        h=args.img_h, w=args.img_w,
        fov_up=args.fov_up, fov_down=args.fov_down,
        depth_mean=args.depth_mean, depth_std=args.depth_std,
        stride=args.stride * 5,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ------------------------------------------------------------------ model
    model = RangeViewFlowModel(
        in_channels=2,
        img_h=args.img_h,           img_w=args.img_w,
        patch_h=args.patch_h,       patch_w=args.patch_w,
        channels=args.channels,
        num_flow_blocks=args.num_blocks,
        layers_per_block=args.layers_per_block,
        context_layers=args.context_layers,
        head_dim=args.head_dim,
        cond_dropout_p=args.cond_dropout_p,
    ).to(device)

    # Give the context encoder a higher effective LR.  It only receives
    # gradients through cross-attention, so it converges more slowly than
    # the flow coupling blocks.  LambdaLR preserves the LR ratio throughout
    # the cosine schedule because it multiplies each group's base_lr.
    # NOTE: context_lr_scale > 1 risks destabilising the coupled system if
    # the context features grow faster than the coupling layers can adapt.
    # Keep context_lr_scale close to 1.0 (e.g. 1.2) for safety.
    _ctx_param_ids = {id(p) for p in model.context_encoder.parameters()}
    optimizer = torch.optim.AdamW([
        {'params': list(model.context_encoder.parameters()),
         'lr': args.lr * args.context_lr_scale},
        {'params': [p for p in model.parameters() if id(p) not in _ctx_param_ids],
         'lr': args.lr},
    ], betas=(0.9, 0.95), weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    scheduler   = cosine_schedule_with_warmup(
        optimizer, warmup_steps=len(train_loader),
        total_steps=total_steps, min_lr=1e-6, max_lr=args.lr,
    )
    # Note: GradScaler is designed for float16 and is not needed with bfloat16.
    # bfloat16 shares float32's exponent range, so it does not underflow.
    # We rely on gradient clipping alone for stability.
    writer = SummaryWriter(args.logdir)

    use_wandb = bool(args.wandb_project)
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=vars(args),
            resume='allow',
        )

    start_epoch, global_step = 0, 0
    best_val_nll = math.inf      # tracks the best finite val NLL seen so far
    if args.resume:
        start_epoch, global_step = load_checkpoint(args.resume, model, optimizer)

    # ------------------------------------------------------------------ training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for batch in train_loader:
            past_frames  = batch['past_frames'].to(device)          # [B, T, 2, H, W]
            future_frame = batch['future_frames'][:, 0].to(device)  # [B, 2, H, W]
            past_poses   = batch['past_poses'].to(device)           # [B, T, 4, 4]

            # ----------------------------------------------------------
            # Validity mask — built on float32 data BEFORE autocast so
            # that the exact-equality check on invalid_depth is reliable.
            # [B, N, 1]  (1 = patch with at least one valid LiDAR return)
            # ----------------------------------------------------------
            valid_mask = model.get_valid_patch_mask(future_frame, invalid_depth)

            # ----------------------------------------------------------
            # Dequantisation noise — added only to valid pixels to stop
            # the flow from collapsing to spike solutions on the LiDAR
            # sampling grid without corrupting the empty-pixel mask.
            # ----------------------------------------------------------
            pixel_valid = (future_frame[:, 0:1] != invalid_depth)  # [B,1,H,W]
            noise       = torch.randn_like(future_frame) * DENOISING_STD
            future_frame = future_frame + noise * pixel_valid.float()

            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                z, logdets        = model(future_frame, past_frames, past_poses)
                loss, components  = model.get_loss(
                    z, logdets, valid_mask,
                    logdet_penalty_weight=args.logdet_penalty_weight,
                    logdet_target=args.logdet_target,
                )

            # Skip batch when the forward-pass loss is non-finite OR exceeds
            # the spike threshold.  Both conditions corrupt Adam's moment
            # estimates; skipping the *entire* batch (no backward, no step)
            # is safer than clipping alone.
            if not torch.isfinite(loss) or loss.item() > args.loss_skip_thresh:
                print(f'[ep {epoch:03d} step {global_step:06d}]  '
                      f'WARNING: loss={loss.item():.4f} exceeds threshold '
                      f'({args.loss_skip_thresh:.1f}), skipping batch')
                global_step += 1
                continue   # optimizer.zero_grad() was already called above

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if global_step % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                writer.add_scalar('train/nll',            loss.item(),                          global_step)
                writer.add_scalar('train/prior',          components['prior'].item(),           global_step)
                writer.add_scalar('train/logdet',         components['logdet'].item(),          global_step)
                writer.add_scalar('train/logdet_penalty', components['logdet_penalty'].item(),  global_step)
                writer.add_scalar('train/grad_norm',      grad_norm.item(),                     global_step)
                writer.add_scalar('train/lr',             lr,                                   global_step)
                if use_wandb:
                    wandb.log({
                        'train/nll':            loss.item(),
                        'train/prior':          components['prior'].item(),
                        'train/logdet':         components['logdet'].item(),
                        'train/logdet_penalty': components['logdet_penalty'].item(),
                        'train/grad_norm':      grad_norm.item(),
                        'train/lr':             lr,
                    }, step=global_step)
                ld_pen = components['logdet_penalty'].item()
                print(f'[ep {epoch:03d} step {global_step:06d}]  '
                      f'nll={loss.item():.4f}  '
                      f'prior={components["prior"].item():.4f}  '
                      f'logdet={components["logdet"].item():.4f}  '
                      + (f'ld_pen={ld_pen:.4f}  ' if ld_pen > 0 else '')
                      + f'gnorm={grad_norm.item():.3f}  lr={lr:.2e}')
            global_step += 1

        # ---------------------------------------------------------------- validation
        model.eval()
        val_losses, val_priors, val_logdets = [], [], []
        skipped = 0
        with torch.no_grad():
            for batch in val_loader:
                past_frames  = batch['past_frames'].to(device)
                future_frame = batch['future_frames'][:, 0].to(device)
                past_poses   = batch['past_poses'].to(device)

                valid_mask = model.get_valid_patch_mask(future_frame, invalid_depth)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    z, logdets       = model(future_frame, past_frames, past_poses)
                    loss, components = model.get_loss(z, logdets, valid_mask)

                # Guard: discard batches with non-finite loss so a single
                # bad sample cannot send the epoch average to ±inf.
                if not torch.isfinite(loss):
                    skipped += 1
                    continue

                val_losses.append(loss.item())
                val_priors.append(components['prior'].item())
                val_logdets.append(components['logdet'].item())

        if val_losses:
            val_nll    = sum(val_losses)  / len(val_losses)
            val_prior  = sum(val_priors)  / len(val_priors)
            val_logdet = sum(val_logdets) / len(val_logdets)
        else:
            # All batches were non-finite — report NaN as a clear signal.
            val_nll = val_prior = val_logdet = math.nan

        writer.add_scalar('val/nll',    val_nll,    epoch)
        writer.add_scalar('val/prior',  val_prior,  epoch)
        writer.add_scalar('val/logdet', val_logdet, epoch)
        if use_wandb:
            wandb.log({
                'val/nll':    val_nll,
                'val/prior':  val_prior,
                'val/logdet': val_logdet,
                'epoch':      epoch,
            }, step=global_step)
        print(f'[ep {epoch:03d}]  '
              f'val_nll={val_nll:.4f}  '
              f'val_prior={val_prior:.4f}  '
              f'val_logdet={val_logdet:.4f}'
              + (f'  ({skipped} batches skipped)' if skipped else ''))

        # Persist the best-seen checkpoint permanently (never rolled over).
        # Protects against late-stage collapse wiping out all good weights.
        if math.isfinite(val_nll) and val_nll < best_val_nll:
            best_val_nll = val_nll
            best_path = args.logdir / 'ckpt_best.pth'
            save_checkpoint(str(best_path), model, optimizer, epoch + 1, global_step)
            print(f'[best] val_nll={val_nll:.4f} → {best_path.name}')

        new_ckpt = args.logdir / f'ckpt_{epoch:03d}.pth'
        save_checkpoint(str(new_ckpt), model, optimizer, epoch + 1, global_step)

        # Delete the previous epoch's checkpoint to save disk space.
        if epoch > 0:
            old_ckpt = args.logdir / f'ckpt_{epoch - 1:03d}.pth'
            if old_ckpt.exists():
                old_ckpt.unlink()
                print(f'Removed old checkpoint: {old_ckpt.name}')

        # -------------------------------------------------------- visualisation
        if args.vis_every > 0 and (epoch + 1) % args.vis_every == 0:
            visualize_predictions(
                model, val_ds, args,
                epoch=epoch,
                device=device,
                writer=writer,
                use_wandb=use_wandb,
                global_step=global_step,
                num_future=args.vis_futures,
                temperature=args.sample_temp,
            )

    writer.close()
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
