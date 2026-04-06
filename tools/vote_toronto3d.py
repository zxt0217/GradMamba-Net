import argparse
import hashlib
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

# --- 路径 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from utils.helper_ply import read_ply, write_ply
from utils.helper_tool import DataProcessing as DP
from data_utils.Toronto3DBlockDataLoader import voxelization, fps_series_func

# --- 默认配置 ---
DEFAULT_MODEL_NAME = 'diffconv_umamba'
DEFAULT_CHECKPOINT = './log/toronto3d_seg/<exp_name>/checkpoints/best_model.pth'
DEFAULT_TEST_FILE = './data/Toronto_3D/L002.ply'
DEFAULT_OUTPUT = './last_L002.ply'

NUM_CLASSES = 8
FPS_N_LIST = [512, 128, 32]
CLASSES = ['Ground', 'Road_markings', 'Natural', 'Building', 'Utility_line', 'Pole', 'Car', 'Fence']

# 类别可视化颜色（1-8 标签映射到如下 RGB）
CLASS_COLORS = np.array([
    [130, 130, 130],  # Ground
    [255, 255, 255],  # Road_markings
    [0, 180, 0],      # Natural
    [220, 40, 40],    # Building
    [255, 180, 0],    # Utility_line
    [0, 120, 255],    # Pole
    [255, 0, 255],    # Car
    [255, 0, 0],      # Fence
], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser("Toronto3D whole-scene voting + PLY export")
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--test_file', type=str, default=DEFAULT_TEST_FILE)
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT, help='输出文件前缀或 .ply 路径')
    parser.add_argument('--save_mode', type=str, choices=['raw_rgb', 'pred_rgb', 'both'], default='both')
    parser.add_argument('--npoints', type=int, default=16384)
    parser.add_argument('--block_size', type=float, default=20.0)
    parser.add_argument('--stride', type=float, default=5.0)
    parser.add_argument('--grid_size', type=float, default=0.06)
    parser.add_argument('--scan_directions', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0, help='固定随机采样，方便比较不同权重')
    return parser.parse_args()


def coordinate_normalize(input_xyz):
    mean_xyz = (np.max(input_xyz, axis=0) + np.min(input_xyz, axis=0)) / 2
    max_val = np.max(np.abs(input_xyz - mean_xyz))
    if max_val == 0:
        max_val = 1e-6
    return (input_xyz - mean_xyz) / max_val


def file_sha1(path):
    sha1 = hashlib.sha1()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha1.update(chunk)
    return sha1.hexdigest()


def remap_state_dict_keys(old_state_dict):
    new_state_dict = {}
    for k, v in old_state_dict.items():
        name = k.replace('module.', '')
        if 'mvsa_' in name and 'mvsa_gamma' not in name:
            name = name.replace('mvsa_', 'mvsa_module.')
        new_state_dict[name] = v
    return new_state_dict


def get_state_dict_from_checkpoint(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        if 'model_state_dict' in checkpoint_obj:
            return checkpoint_obj['model_state_dict']
        if 'state_dict' in checkpoint_obj:
            return checkpoint_obj['state_dict']
        if all(torch.is_tensor(v) for v in checkpoint_obj.values()):
            return checkpoint_obj
    raise ValueError("无法从 checkpoint 中解析 state_dict，请检查权重文件格式。")


def labels_to_rgb(raw_labels_1_to_8):
    idx = np.clip(raw_labels_1_to_8.astype(np.int64) - 1, 0, NUM_CLASSES - 1)
    return CLASS_COLORS[idx]


def resolve_output_paths(output_arg):
    out = Path(output_arg)
    if out.suffix.lower() != '.ply':
        out = out.with_suffix('.ply')
    raw_path = out.with_name(f"{out.stem}_rawrgb.ply")
    pred_path = out.with_name(f"{out.stem}_predrgb.ply")
    return str(raw_path), str(pred_path)


@torch.no_grad()
def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_path = os.path.abspath(args.checkpoint)
    test_file = os.path.abspath(args.test_file)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")
    if not os.path.isfile(test_file):
        raise FileNotFoundError(f"测试点云不存在: {test_file}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"推理设备: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint SHA1: {file_sha1(checkpoint_path)}")

    model_module = importlib.import_module(args.model_name)
    classifier = model_module.get_model(NUM_CLASSES, FPS_N_LIST, normal_channel=True).to(device)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    old_state_dict = get_state_dict_from_checkpoint(checkpoint)
    new_state_dict = remap_state_dict_keys(old_state_dict)
    classifier.load_state_dict(new_state_dict, strict=True)
    classifier.eval()
    print("权重加载成功，strict=True 校验通过。")

    # 读取原始点云和标签
    pc = read_ply(test_file)
    xyz_raw = np.vstack((pc['x'], pc['y'], pc['z'])).T.astype(np.float32)
    labels_raw = pc['scalar_Label'].astype(np.int32)  # 原始标签 1-8
    intensity = (pc['scalar_Intensity'].astype(np.float32) / 255.0).reshape(-1, 1)
    rgb_feat = np.vstack((pc['red'], pc['green'], pc['blue'])).T.astype(np.float32) / 255.0
    feat_raw = np.concatenate([intensity, rgb_feat], axis=-1)

    # 滑动窗口投票
    sub_xyz, sub_feat, _ = DP.grid_sub_sampling(xyz_raw, feat_raw, labels_raw.reshape(-1, 1), args.grid_size)
    vote_probs = np.zeros((sub_xyz.shape[0], NUM_CLASSES), dtype=np.float32)
    vote_counts = np.zeros((sub_xyz.shape[0], 1), dtype=np.float32)
    rng = np.random.default_rng(args.seed)

    coord_min, coord_max = np.amin(sub_xyz, axis=0), np.amax(sub_xyz, axis=0)
    grid_x = np.arange(coord_min[0], coord_max[0] + args.stride, args.stride)
    grid_y = np.arange(coord_min[1], coord_max[1] + args.stride, args.stride)

    print("开始全场景滑动窗口投票...")
    for cx in tqdm(grid_x):
        for cy in grid_y:
            mask = (sub_xyz[:, 0] >= cx - args.block_size / 2) & (sub_xyz[:, 0] <= cx + args.block_size / 2) & \
                   (sub_xyz[:, 1] >= cy - args.block_size / 2) & (sub_xyz[:, 1] <= cy + args.block_size / 2)
            point_idxs = np.where(mask)[0]
            if point_idxs.size < 100:
                continue

            sel = rng.choice(point_idxs, args.npoints, replace=(point_idxs.size < args.npoints))
            cur_xyz = sub_xyz[sel]
            rel_h = (cur_xyz[:, 2] - np.min(cur_xyz[:, 2])).reshape(-1, 1)
            norm_xyz = coordinate_normalize(cur_xyz)
            input_feat = np.concatenate([norm_xyz, sub_feat[sel], rel_h], axis=-1)

            _, v_indices, _, _ = voxelization(input_feat, 0.4)
            fps_idx, s_idx = fps_series_func(input_feat, v_indices, FPS_N_LIST, num_scan_dirs=args.scan_directions)

            input_t = torch.from_numpy(input_feat).float().to(device).unsqueeze(0).transpose(2, 1)
            fps_t = torch.from_numpy(fps_idx).long().to(device).unsqueeze(0)
            s_t = torch.from_numpy(s_idx).long().to(device).unsqueeze(0)

            pred = classifier(input_t, fps_t, s_t)  # [1, N, C], log_softmax 输出
            prob = torch.exp(pred).cpu().numpy()[0]
            vote_probs[sel] += prob
            vote_counts[sel] += 1.0

    print("正在计算 Voting mIoU 分数...")
    never_voted = int(np.sum(vote_counts.squeeze(-1) == 0))
    print(f"子采样点未被投票数: {never_voted}/{vote_counts.shape[0]}")

    final_sub_probs = vote_probs / (vote_counts + 1e-6)
    tree = cKDTree(sub_xyz)
    _, nearest_idx = tree.query(xyz_raw, k=1)
    final_raw_labels = np.argmax(final_sub_probs[nearest_idx], axis=1) + 1  # 0-7 -> 1-8

    total_correct_class = np.zeros(NUM_CLASSES)
    total_iou_deno_class = np.zeros(NUM_CLASSES)
    for l in range(NUM_CLASSES):
        target_label = l + 1
        total_correct_class[l] += np.sum((final_raw_labels == target_label) & (labels_raw == target_label))
        total_iou_deno_class[l] += np.sum((final_raw_labels == target_label) | (labels_raw == target_label))

    print("\n" + "=" * 30)
    print(f"Final Voting Results (Whole Scene {Path(test_file).stem})")
    for i in range(NUM_CLASSES):
        iou = total_correct_class[i] / (total_iou_deno_class[i] + 1e-6)
        print(f"{CLASSES[i]:15s} IoU: {iou:.4f}")
    overall_miou = np.mean(total_correct_class / (total_iou_deno_class + 1e-6))
    print("-" * 30)
    print(f"Overall Voting mIoU: {overall_miou:.4f}")
    print("=" * 30 + "\n")

    pred_unique, pred_counts = np.unique(final_raw_labels, return_counts=True)
    pred_hist = {int(k): int(v) for k, v in zip(pred_unique, pred_counts)}
    print(f"预测标签分布(1-8): {pred_hist}")

    xyz_save = xyz_raw.astype(np.float32)
    raw_r = pc['red'].reshape(-1, 1).astype(np.uint8)
    raw_g = pc['green'].reshape(-1, 1).astype(np.uint8)
    raw_b = pc['blue'].reshape(-1, 1).astype(np.uint8)
    pred_label_save = final_raw_labels.reshape(-1, 1).astype(np.int32)
    gt_label_save = labels_raw.reshape(-1, 1).astype(np.int32)

    raw_path, pred_path = resolve_output_paths(args.output)
    common_names = ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'gt_label']

    if args.save_mode in ['raw_rgb', 'both']:
        raw_fields = [xyz_save, raw_r, raw_g, raw_b, pred_label_save, gt_label_save]
        write_ply(raw_path, raw_fields, common_names)
        print(f"已保存(原始RGB + 预测标签): {raw_path}")

    if args.save_mode in ['pred_rgb', 'both']:
        pred_rgb = labels_to_rgb(final_raw_labels)
        pred_r = pred_rgb[:, 0].reshape(-1, 1).astype(np.uint8)
        pred_g = pred_rgb[:, 1].reshape(-1, 1).astype(np.uint8)
        pred_b = pred_rgb[:, 2].reshape(-1, 1).astype(np.uint8)
        pred_fields = [xyz_save, pred_r, pred_g, pred_b, pred_label_save, gt_label_save]
        write_ply(pred_path, pred_fields, common_names)
        print(f"已保存(预测RGB着色 + 预测标签): {pred_path}")

    print("提示: 之前看起来“结果一样”，通常是因为 PLY 里保存了原始 RGB，默认显示不会反映 label 变化。")


if __name__ == '__main__':
    main()
