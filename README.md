# NormCast

Autoregressive LiDAR range-view prediction using a conditional normalizing flow model.

---

## Overview

NormCast predicts future LiDAR range-view frames from a window of past frames plus ground-truth sensor poses.  The model is inspired by [TAR-Flow](https://github.com/apple/ml-tarflow) (transformer-based NVP normalizing flow) and the range-view pipeline from [DifForecast](https://github.com/Shuhul24/difforecast).

### Architecture

```
Past frames [T, 2, H, W] + Past poses [T, 4, 4]
        │
        ▼
  ContextEncoder
  ─ Patchify past frames  → [T×N, patch_dim]
  ─ 6-DOF relative pose   → pose embedding (added to patch tokens)
  ─ Stacked self-attention → context tokens [T×N, C]
        │
        ▼  (cross-attention)
  NVP Flow Blocks  ×  num_blocks
  ─ Causal self-attention over current-frame patches  (intra-frame autoregression)
  ─ Cross-attention to context tokens                 (past-frame conditioning)
  ─ Affine coupling:  z_i = (x_i − shift_i) · exp(−scale_i)
  ─ Alternating flip permutation between blocks
        │
        ▼
  z ∼ N(0, I)   (training: NLL loss)
```

**Intra-frame autoregression** follows TAR-Flow: each patch token attends only to previous patches (causal mask during training; KV-cache during sampling).

**Cross-frame conditioning** uses cross-attention from current-frame patches to all context tokens, giving the model explicit access to past observations and their relative 6-DOF poses.

**Multi-step prediction** slides the conditioning window forward; each new predicted frame becomes part of the conditioning context for the next step.

---

## Data preparation

Download KITTI Odometry from <https://www.cvlibs.net/datasets/kitti/eval_odometry.php>:
- `data_odometry_velodyne.zip`  (point clouds)
- `data_odometry_poses.zip`     (ground-truth poses)

Expected layout:
```
/path/to/kitti/
├── sequences/
│   ├── 00/velodyne/000000.bin  ...
│   ├── 01/velodyne/000000.bin  ...
│   └── ...
└── poses/
    ├── 00.txt
    ├── 01.txt
    └── ...
```

KITTI split used by default:
| Split | Sequences |
|-------|-----------|
| train | 00 – 05   |
| val   | 06 – 07   |
| test  | 08 – 10   |

---

## Installation

### Conda environment (recommended)

The code uses Python 3.10+ syntax (`X | Y` type unions).
Tested with **Python 3.10 / 3.11** and **PyTorch 2.1+**.

**CUDA 11.8 (Ampere / Volta GPUs — A100, V100, RTX 30xx)**
```bash
conda create -n normcast python=3.10
conda activate normcast
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 \
      -c pytorch -c nvidia
pip install tensorboard matplotlib numpy
```

**CUDA 12.1 (Ada Lovelace GPUs — RTX 40xx, H100)**
```bash
conda create -n normcast python=3.11
conda activate normcast
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 \
      -c pytorch -c nvidia
pip install tensorboard matplotlib numpy
```

**CPU only (no GPU)**
```bash
conda create -n normcast python=3.10
conda activate normcast
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install tensorboard matplotlib numpy
```

Verify the installation:
```bash
python - <<'EOF'
import torch, sys
print(f"Python {sys.version}")
print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
EOF
```

---

## Training

```bash
python train.py \
    --data /path/to/kitti \
    --logdir runs/normcast \
    --train_seqs 0 6 \
    --val_seqs   6 8 \
    --condition_frames 4 \
    --channels 512 \
    --num_blocks 4 \
    --layers_per_block 4 \
    --context_layers 2 \
    --batch_size 4 \
    --epochs 100 \
    --lr 3e-4 \
    --stride 5
```

Resume from a checkpoint:
```bash
python train.py ... --resume runs/normcast/ckpt_049.pth
```

---

## Evaluation

```bash
python evaluate.py \
    --data /path/to/kitti \
    --ckpt runs/normcast/ckpt_099.pth \
    --outdir eval_results \
    --test_seqs 8 11 \
    --future_frames 5 \
    --temperature 1.0
```

Outputs:
- `eval_results/metrics.txt`         — mean depth L1 + per-sample values
- `eval_results/sample_XXXX.png`     — GT vs predicted depth maps (first `--num_vis` samples)

---

## Key arguments

| Argument | Default | Description |
|---|---|---|
| `--condition_frames` | 4 | Past frames used as conditioning |
| `--future_frames` | 5 | Steps to predict autoregressively (eval only) |
| `--patch_h / --patch_w` | 4 / 32 | Patch size for 64×2048 range images (→ 1024 patches) |
| `--channels` | 512 | Transformer hidden dimension |
| `--num_blocks` | 4 | Number of NVP flow blocks |
| `--layers_per_block` | 4 | Transformer layers per flow block |
| `--context_layers` | 2 | Self-attention layers in the past-frame encoder |
| `--temperature` | 1.0 | Noise temperature for sampling (< 1 = sharper) |

---

## Design notes

**Why normalizing flows?**  TAR-Flow demonstrates that transformer-based NVP flows match diffusion models on image generation tasks while providing exact log-likelihood computation, simpler training objectives, and faster deterministic sampling.

**Pose encoding**  Each past frame's absolute 4×4 pose is converted to a 6-DOF vector (dx, dy, dz, roll, pitch, yaw) relative to the most recent past frame, then projected to the model dimension and added to that frame's patch embeddings.  This gives the model explicit motion cues without embedding absolute scene coordinates.

**Intra-frame vs inter-frame autoregression**  The flow blocks apply causal self-attention within a single future frame (patch order: raster scan), which is the TAR-Flow design.  Inter-frame conditioning is handled separately via cross-attention to the context encoder output, keeping the two levels of structure cleanly separated.
