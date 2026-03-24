import os
import numpy as np
import torch
from torch.utils.data import Dataset


class RangeProjection:
    """Spherical projection of a 3-D LiDAR point cloud to a 2-D range image."""

    def __init__(self, h: int = 64, w: int = 2048,
                 fov_up: float = 3.0, fov_down: float = -25.0):
        self.h, self.w = h, w
        self.fov_up    = np.deg2rad(fov_up)
        self.fov_down  = np.deg2rad(fov_down)
        self.fov       = abs(self.fov_up) + abs(self.fov_down)

    def project(self, points: np.ndarray) -> np.ndarray:
        """
        points: [N, 4]  (x, y, z, intensity)
        returns: [2, H, W]  channels = (depth, intensity), zeros where no return
        """
        x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
        depth = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        valid = depth > 0.0

        yaw   = -np.arctan2(y[valid], x[valid])
        pitch = np.arcsin(np.clip(z[valid] / depth[valid], -1, 1))

        col = np.clip((0.5 * (yaw / np.pi + 1.0) * self.w).astype(np.int32), 0, self.w - 1)
        row = np.clip(((1.0 - (pitch - self.fov_down) / self.fov) * self.h).astype(np.int32),
                      0, self.h - 1)

        order  = np.argsort(depth[valid])[::-1]   # far-to-near so closer points overwrite
        cols_o = col[order]
        rows_o = row[order]

        img = np.zeros((2, self.h, self.w), dtype=np.float32)
        img[0, rows_o, cols_o] = depth[valid][order]
        img[1, rows_o, cols_o] = intensity[valid][order]
        return img


def _load_kitti_poses(pose_file: str) -> np.ndarray:
    poses = []
    with open(pose_file) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            P = np.eye(4, dtype=np.float32)
            P[:3, :] = np.array(vals, dtype=np.float32).reshape(3, 4)
            poses.append(P)
    return np.stack(poses)


class KITTIRangeViewDataset(Dataset):
    """
    KITTI Odometry range-view dataset.

    Returns windows of (condition_frames + future_frames) consecutive scans with
    their absolute 4×4 poses.  The model conditions on past_frames and predicts
    future_frames one step at a time.

    Directory layout expected:
        <root>/dataset/sequences/<seq:02d>/velodyne/*.bin
        <root>/poses/<seq:02d>.txt
    """

    def __init__(
        self,
        root:             str,
        sequences:        list[int],
        condition_frames: int   = 4,
        future_frames:    int   = 1,
        h:                int   = 64,
        w:                int   = 2048,
        fov_up:           float = 3.0,
        fov_down:         float = -25.0,
        depth_mean:       float = 12.0,
        depth_std:        float = 12.0,
        stride:           int   = 1,
    ):
        self.condition_frames = condition_frames
        self.future_frames    = future_frames
        self.depth_mean       = depth_mean
        self.depth_std        = depth_std
        self.projector        = RangeProjection(h, w, fov_up, fov_down)
        window                = condition_frames + future_frames

        self.samples: list[tuple[list[str], np.ndarray]] = []
        for seq in sequences:
            sid      = f'{seq:02d}'
            pc_dir   = os.path.join(root, 'dataset', 'sequences', sid, 'velodyne')
            pose_f   = os.path.join(root, 'poses', f'{sid}.txt')
            pc_files = sorted(
                os.path.join(pc_dir, f) for f in os.listdir(pc_dir) if f.endswith('.bin')
            )
            poses    = _load_kitti_poses(pose_f)
            for start in range(0, len(pc_files) - window + 1, stride):
                end = start + window
                self.samples.append((pc_files[start:end], poses[start:end]))

    def _load_frame(self, path: str) -> np.ndarray:
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        img = self.projector.project(pts)
        img[0] = (img[0] - self.depth_mean) / self.depth_std
        return img

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pc_files, poses = self.samples[idx]
        T = self.condition_frames

        frames = np.stack([self._load_frame(f) for f in pc_files])   # [T+F, 2, H, W]

        past_frames   = torch.from_numpy(frames[:T])
        future_frames = torch.from_numpy(frames[T:])
        past_poses    = torch.from_numpy(poses[:T].copy())
        future_poses  = torch.from_numpy(poses[T:].copy())

        return {
            'past_frames':   past_frames,     # [T,   2, H, W]
            'future_frames': future_frames,   # [F,   2, H, W]
            'past_poses':    past_poses,      # [T,   4, 4]
            'future_poses':  future_poses,    # [F,   4, 4]
        }
