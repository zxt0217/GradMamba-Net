import os
import os.path as osp
import numpy as np
import sys
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import time

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../"))


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def _build_scan_directions(num_scan_dirs):
    all_dirs = [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
    ]
    num_scan_dirs = max(1, min(int(num_scan_dirs), len(all_dirs)))
    return all_dirs[:num_scan_dirs]


def fps_series_func(points, voxel_indices, samplepoints_list, num_scan_dirs=4):
    pad_width = points.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    points = torch.as_tensor(points, dtype=torch.float32, device=device).unsqueeze(0)
    voxel_indices = torch.as_tensor(voxel_indices, dtype=torch.float32, device=device).unsqueeze(0)
    fps_index_list = []
    series_idx_lists = []

    series_list = _build_scan_directions(num_scan_dirs)

    for i in range(len(samplepoints_list)):
        S = samplepoints_list[i]
        xyz = points[:, :, :3]

        fps_index = farthest_point_sample(xyz, S)
        points = index_points(points, fps_index)
        new_voxel_indices = index_points(voxel_indices, fps_index).squeeze(0).cpu().data.numpy()
        voxel_indices = index_points(voxel_indices, fps_index)

        fps_index = fps_index.cpu().data.numpy()
        padded_fps_index = np.pad(fps_index, ((0, 0), (0, pad_width - fps_index.shape[1])), mode='constant')
        fps_index_list.append(padded_fps_index)

        series_idx_list = []
        for j in range(len(series_list)):
            series = series_list[j]
            new_voxel_indices_ForSeries = new_voxel_indices * series
            sorting_indices = np.expand_dims(np.lexsort(
                (new_voxel_indices_ForSeries[:, 0], new_voxel_indices_ForSeries[:, 1],
                 new_voxel_indices_ForSeries[:, 2])), axis=0)
            padded_sorting_indices = np.expand_dims(
                np.pad(sorting_indices, ((0, 0), (0, pad_width - sorting_indices.shape[1])), mode='constant'), axis=0)
            series_idx_list.append(padded_sorting_indices)

        series_idx_array = np.concatenate(series_idx_list, axis=1)
        series_idx_lists.append(series_idx_array)

    series_idx_arrays = np.concatenate(series_idx_lists, axis=0)
    fps_index_array = np.vstack(fps_index_list)

    return fps_index_array, series_idx_arrays


def voxelization(points, voxel_size):
    voxel_indices = np.floor(points[:, :3] / voxel_size).astype(np.int32)
    coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
    bounding_box = coord_max - coord_min
    voxel_total = np.ceil(bounding_box[0] * bounding_box[1] * bounding_box[2] / voxel_size ** 3).astype(np.int32)
    voxel_valid = np.unique(voxel_indices, axis=0)
    return points, voxel_indices, voxel_total, voxel_valid


class Toronto3DDataset(Dataset):
    def __init__(self, split='train', data_root='../data/Toronto3D_blocks/', fps_n_list=[512, 128, 32], label_number=8,
                 npoints=16384, fence_sample_boost=1.0, scan_directions=4):
        super().__init__()

        self.fps_n_list = fps_n_list
        self.npoints = npoints
        self.fence_sample_boost = max(1.0, float(fence_sample_boost))
        self.scan_directions = max(1, min(int(scan_directions), 8))

        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if room.endswith('.npy')]

        if split == 'train':
            rooms_split = [room for room in rooms if 'L002' not in room]
        else:
            rooms_split = [room for room in rooms if 'L002' in room]

        self.sample_points, self.sample_labels = [], []
        self.fps_index_array_list, self.series_idx_arrays_list = [], []
        self.sample_weights = []
        self.fence_block_count = 0
        self.fence_ratio_values = []
        labelweights = np.zeros(label_number)
        voxel_size = 0.4

        for room_name in tqdm(rooms_split, total=len(rooms_split), desc=f"Loading {split} data"):
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)

            for i in range(room_data.shape[0]):
                points, labels = room_data[i][:, :8], room_data[i][:, 8]

                labels = labels.astype(np.int32)
                labels = np.where(labels > 0, labels - 1, -1)

                tmp, _ = np.histogram(labels[labels >= 0], range(label_number + 1))
                labelweights += tmp

                array = np.arange(points.shape[0])
                np.random.shuffle(array)
                points = points[array]
                labels = labels[array]

                valid_mask = labels >= 0
                valid_count = int(np.sum(valid_mask))
                fence_count = int(np.sum(labels[valid_mask] == (label_number - 1))) if valid_count > 0 else 0
                fence_ratio = (fence_count / valid_count) if valid_count > 0 else 0.0
                has_fence = fence_ratio > 0.0
                if has_fence:
                    self.fence_block_count += 1

                points, voxel_indices, voxel_total, voxel_valid = voxelization(points, voxel_size)
                fps_index_array, series_idx_arrays = fps_series_func(
                    points, voxel_indices, self.fps_n_list, num_scan_dirs=self.scan_directions
                )

                self.sample_points.append(points)
                self.sample_labels.append(labels)
                self.fps_index_array_list.append(fps_index_array)
                self.series_idx_arrays_list.append(series_idx_arrays)
                # Continuous fence-aware sampling:
                # ratio=0 -> 1.0, ratio up -> approaches fence_sample_boost.
                fence_weight = 1.0 + (self.fence_sample_boost - 1.0) * np.sqrt(max(0.0, min(1.0, fence_ratio)))
                self.sample_weights.append(fence_weight)
                self.fence_ratio_values.append(fence_ratio)

        print(f"Totally {len(self.sample_points)} blocks in Toronto3D {split} set.")
        if len(self.sample_points) > 0:
            print(f"Fence-containing blocks: {self.fence_block_count}/{len(self.sample_points)}")
            mean_fence_ratio = float(np.mean(self.fence_ratio_values)) if len(self.fence_ratio_values) > 0 else 0.0
            print(f"Mean fence ratio per block: {mean_fence_ratio:.4f}, scan_directions={self.scan_directions}")

        self.labelweights = np.ones(label_number)
        if split == 'train':
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / (np.sum(labelweights) + 1e-6)
            self.labelweights = np.power(np.amax(labelweights) / (labelweights + 1e-6), 1 / 3.0)
        self.sample_weights = np.asarray(self.sample_weights, dtype=np.float64)

    def __getitem__(self, idx):
        points = self.sample_points[idx]
        labels = self.sample_labels[idx]
        fps_index_array = self.fps_index_array_list[idx]
        series_idx_arrays = self.series_idx_arrays_list[idx]
        return points, labels, fps_index_array, series_idx_arrays

    def __len__(self):
        return len(self.sample_points)
