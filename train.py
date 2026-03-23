import argparse
import pathlib

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from dataset import KITTIRangeViewDataset
from model   import RangeViewFlowModel
from utils   import cosine_schedule_with_warmup, save_checkpoint, load_checkpoint


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
    # Paths
    p.add_argument('--logdir',  type=pathlib.Path, default=pathlib.Path('runs/normcast'))
    p.add_argument('--resume',  type=str, default='')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    args.logdir.mkdir(parents=True, exist_ok=True)

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
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4
    )
    total_steps = args.epochs * len(train_loader)
    scheduler   = cosine_schedule_with_warmup(
        optimizer, warmup_steps=len(train_loader),
        total_steps=total_steps, min_lr=1e-6, max_lr=args.lr,
    )
    scaler  = torch.amp.GradScaler()
    writer  = SummaryWriter(args.logdir)

    start_epoch, global_step = 0, 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(args.resume, model, optimizer)

    # ------------------------------------------------------------------ training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for batch in train_loader:
            past_frames  = batch['past_frames'].to(device)          # [B, T, 2, H, W]
            future_frame = batch['future_frames'][:, 0].to(device)  # [B, 2, H, W]
            past_poses   = batch['past_poses'].to(device)           # [B, T, 4, 4]

            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                z, logdets = model(future_frame, past_frames, past_poses)
                loss       = model.get_loss(z, logdets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if global_step % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                writer.add_scalar('train/nll', loss.item(), global_step)
                writer.add_scalar('train/lr',  lr,          global_step)
                print(f'[ep {epoch:03d} step {global_step:06d}]  '
                      f'nll={loss.item():.4f}  lr={lr:.2e}')
            global_step += 1

        # ---------------------------------------------------------------- validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                past_frames  = batch['past_frames'].to(device)
                future_frame = batch['future_frames'][:, 0].to(device)
                past_poses   = batch['past_poses'].to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    z, logdets = model(future_frame, past_frames, past_poses)
                    val_losses.append(model.get_loss(z, logdets).item())

        val_nll = sum(val_losses) / len(val_losses)
        writer.add_scalar('val/nll', val_nll, epoch)
        print(f'[ep {epoch:03d}]  val_nll={val_nll:.4f}')

        save_checkpoint(
            str(args.logdir / f'ckpt_{epoch:03d}.pth'),
            model, optimizer, epoch + 1, global_step,
        )

    writer.close()


if __name__ == '__main__':
    main()
