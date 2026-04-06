import argparse
import csv
import hashlib
import importlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from utils.helper_ply import read_ply, write_ply
from utils.helper_tool import DataProcessing as DP
from data_utils.Toronto3DBlockDataLoader import voxelization, fps_series_func


DEFAULT_MODEL_NAME = 'diffconv_umamba'
DEFAULT_CHECKPOINT = './log/toronto3d_seg/<exp_name>/checkpoints/best_model.pth'
DEFAULT_TEST_FILE = './data/Toronto_3D/L002.ply'
DEFAULT_OUTPUT_DIR = './checkpoint_history_exports'

NUM_CLASSES = 8
FPS_N_LIST = [512, 128, 32]
CLASSES = ['Ground', 'Road_markings', 'Natural', 'Building', 'Utility_line', 'Pole', 'Car', 'Fence']
CLASS_COLORS = np.array([
    [130, 130, 130],
    [255, 255, 255],
    [0, 180, 0],
    [220, 40, 40],
    [255, 180, 0],
    [0, 120, 255],
    [255, 0, 255],
    [255, 0, 0],
], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser("Export checkpoint metadata/history and whole-scene PLY")
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help='单个 checkpoint 路径')
    parser.add_argument('--checkpoint_dir', type=str, default='',
                        help='批量导出目录；若指定则忽略 --checkpoint')
    parser.add_argument('--checkpoint_glob', type=str, default='*.pth',
                        help='批量导出时使用的 glob')
    parser.add_argument('--test_file', type=str, default=DEFAULT_TEST_FILE)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--history_log', type=str, default='',
                        help='训练日志路径；不填则尝试从 checkpoint 相邻目录自动查找')
    parser.add_argument('--save_mode', type=str, choices=['raw_rgb', 'pred_rgb', 'both'], default='pred_rgb')
    parser.add_argument('--npoints', type=int, default=16384)
    parser.add_argument('--block_size', type=float, default=20.0)
    parser.add_argument('--stride', type=float, default=5.0)
    parser.add_argument('--grid_size', type=float, default=0.06)
    parser.add_argument('--scan_directions', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
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
        if checkpoint_obj and all(torch.is_tensor(v) for v in checkpoint_obj.values()):
            return checkpoint_obj
    raise ValueError("无法从 checkpoint 中解析 state_dict。")


def labels_to_rgb(raw_labels_1_to_8):
    idx = np.clip(raw_labels_1_to_8.astype(np.int64) - 1, 0, NUM_CLASSES - 1)
    return CLASS_COLORS[idx]


def resolve_output_paths(base_prefix, save_mode):
    base = Path(base_prefix)
    pred_path = str(base.with_name(f"{base.name}_predrgb.ply"))
    raw_path = str(base.with_name(f"{base.name}_rawrgb.ply"))
    if save_mode == 'pred_rgb':
        return '', pred_path
    if save_mode == 'raw_rgb':
        return raw_path, ''
    return raw_path, pred_path


def discover_checkpoints(args):
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
        if not ckpt_dir.is_dir():
            raise FileNotFoundError(f"checkpoint_dir 不存在: {ckpt_dir}")
        checkpoints = sorted(str(p) for p in ckpt_dir.glob(args.checkpoint_glob) if p.is_file())
    else:
        checkpoints = [str(Path(args.checkpoint).expanduser().resolve())]

    if not checkpoints:
        raise FileNotFoundError("没有找到任何 checkpoint 文件。")
    return checkpoints


def auto_find_history_log(checkpoint_path):
    ckpt = Path(checkpoint_path).resolve()
    exp_dir = ckpt.parent.parent
    log_dir = exp_dir / 'logs'
    if not log_dir.is_dir():
        return ''
    txt_files = sorted(log_dir.glob('*.txt'))
    return str(txt_files[0]) if txt_files else ''


def parse_training_history(log_path):
    if not log_path or not os.path.isfile(log_path):
        return []

    epoch_re = re.compile(r"\*{4} Epoch (\d+) \(")
    eval_re = re.compile(r"eval point avg class IoU:\s*([0-9.]+)")
    best_re = re.compile(r"Best mIoU:\s*([0-9.]+)")

    history = []
    current_epoch = None
    current_eval = None
    previous_best = -1.0

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = epoch_re.search(line)
            if m:
                current_epoch = int(m.group(1))
                current_eval = None
                continue

            m = eval_re.search(line)
            if m:
                current_eval = float(m.group(1))
                continue

            m = best_re.search(line)
            if m and current_epoch is not None:
                best_val = float(m.group(1))
                if best_val > previous_best + 1e-12:
                    history.append({
                        'epoch': current_epoch,
                        'eval_miou': current_eval,
                        'best_miou': best_val,
                    })
                    previous_best = best_val
    return history


def load_scene(test_file):
    pc = read_ply(test_file)
    xyz_raw = np.vstack((pc['x'], pc['y'], pc['z'])).T.astype(np.float32)
    labels_raw = pc['scalar_Label'].astype(np.int32)
    intensity = (pc['scalar_Intensity'].astype(np.float32) / 255.0).reshape(-1, 1)
    rgb_feat = np.vstack((pc['red'], pc['green'], pc['blue'])).T.astype(np.float32) / 255.0
    feat_raw = np.concatenate([intensity, rgb_feat], axis=-1)
    return pc, xyz_raw, labels_raw, feat_raw


def prepare_subscene(xyz_raw, feat_raw, labels_raw, grid_size):
    sub_xyz, sub_feat, _ = DP.grid_sub_sampling(xyz_raw, feat_raw, labels_raw.reshape(-1, 1), grid_size)
    tree = cKDTree(sub_xyz)
    _, nearest_idx = tree.query(xyz_raw, k=1)
    return sub_xyz, sub_feat, nearest_idx


def build_windows(sub_xyz, sub_feat, args):
    rng = np.random.default_rng(args.seed)
    coord_min, coord_max = np.amin(sub_xyz, axis=0), np.amax(sub_xyz, axis=0)
    grid_x = np.arange(coord_min[0], coord_max[0] + args.stride, args.stride)
    grid_y = np.arange(coord_min[1], coord_max[1] + args.stride, args.stride)

    windows = []
    for cx in tqdm(grid_x, desc='prepare windows'):
        for cy in grid_y:
            mask = (
                (sub_xyz[:, 0] >= cx - args.block_size / 2) & (sub_xyz[:, 0] <= cx + args.block_size / 2) &
                (sub_xyz[:, 1] >= cy - args.block_size / 2) & (sub_xyz[:, 1] <= cy + args.block_size / 2)
            )
            point_idxs = np.where(mask)[0]
            if point_idxs.size < 100:
                continue

            sel = rng.choice(point_idxs, args.npoints, replace=(point_idxs.size < args.npoints))
            cur_xyz = sub_xyz[sel]
            rel_h = (cur_xyz[:, 2] - np.min(cur_xyz[:, 2])).reshape(-1, 1)
            norm_xyz = coordinate_normalize(cur_xyz)
            input_feat = np.concatenate([norm_xyz, sub_feat[sel], rel_h], axis=-1)

            _, v_indices, _, _ = voxelization(input_feat, 0.4)
            fps_idx, s_idx = fps_series_func(
                input_feat, v_indices, FPS_N_LIST, num_scan_dirs=args.scan_directions
            )
            windows.append({
                'sel': sel.astype(np.int64),
                'input_feat': input_feat.astype(np.float32),
                'fps_idx': fps_idx.astype(np.int64),
                's_idx': s_idx.astype(np.int64),
            })
    return windows


def build_model(model_name, device):
    model_module = importlib.import_module(model_name)
    classifier = model_module.get_model(NUM_CLASSES, FPS_N_LIST, normal_channel=True).to(device)
    classifier.eval()
    return classifier


def checkpoint_meta(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    meta = {
        'checkpoint_path': checkpoint_path,
        'checkpoint_sha1': file_sha1(checkpoint_path),
        'stored_epoch': None,
        'stored_class_avg_iou': None,
    }
    if isinstance(ckpt, dict):
        if 'epoch' in ckpt:
            meta['stored_epoch'] = int(ckpt['epoch']) + 1
        if 'class_avg_iou' in ckpt:
            meta['stored_class_avg_iou'] = float(ckpt['class_avg_iou'])
    return ckpt, meta


@torch.no_grad()
def run_voting_export(classifier, checkpoint_obj, checkpoint_path, pc, xyz_raw, labels_raw,
                      sub_xyz, nearest_idx, windows, args, output_dir):
    state_dict = remap_state_dict_keys(get_state_dict_from_checkpoint(checkpoint_obj))
    classifier.load_state_dict(state_dict, strict=True)
    classifier.eval()

    vote_probs = np.zeros((sub_xyz.shape[0], NUM_CLASSES), dtype=np.float32)
    vote_counts = np.zeros((sub_xyz.shape[0], 1), dtype=np.float32)
    device = next(classifier.parameters()).device

    print(f"\n开始导出: {checkpoint_path}")
    for win in tqdm(windows, desc='voting'):
        input_t = torch.from_numpy(win['input_feat']).float().to(device).unsqueeze(0).transpose(2, 1)
        fps_t = torch.from_numpy(win['fps_idx']).long().to(device).unsqueeze(0)
        s_t = torch.from_numpy(win['s_idx']).long().to(device).unsqueeze(0)
        pred = classifier(input_t, fps_t, s_t)
        prob = torch.exp(pred).cpu().numpy()[0]
        sel = win['sel']
        vote_probs[sel] += prob
        vote_counts[sel] += 1.0

    final_sub_probs = vote_probs / (vote_counts + 1e-6)
    final_raw_labels = np.argmax(final_sub_probs[nearest_idx], axis=1) + 1

    total_correct_class = np.zeros(NUM_CLASSES, dtype=np.float64)
    total_iou_deno_class = np.zeros(NUM_CLASSES, dtype=np.float64)
    per_class_iou = {}
    for l in range(NUM_CLASSES):
        target_label = l + 1
        total_correct_class[l] += np.sum((final_raw_labels == target_label) & (labels_raw == target_label))
        total_iou_deno_class[l] += np.sum((final_raw_labels == target_label) | (labels_raw == target_label))
        per_class_iou[CLASSES[l]] = float(total_correct_class[l] / (total_iou_deno_class[l] + 1e-6))
    voting_miou = float(np.mean(total_correct_class / (total_iou_deno_class + 1e-6)))

    ckpt_name = Path(checkpoint_path).stem
    base_prefix = Path(output_dir) / ckpt_name
    raw_path, pred_path = resolve_output_paths(base_prefix, args.save_mode)

    xyz_save = xyz_raw.astype(np.float32)
    pred_label_save = final_raw_labels.reshape(-1, 1).astype(np.int32)
    gt_label_save = labels_raw.reshape(-1, 1).astype(np.int32)
    common_names = ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'gt_label']

    if raw_path:
        raw_r = pc['red'].reshape(-1, 1).astype(np.uint8)
        raw_g = pc['green'].reshape(-1, 1).astype(np.uint8)
        raw_b = pc['blue'].reshape(-1, 1).astype(np.uint8)
        write_ply(raw_path, [xyz_save, raw_r, raw_g, raw_b, pred_label_save, gt_label_save], common_names)

    if pred_path:
        pred_rgb = labels_to_rgb(final_raw_labels)
        pred_r = pred_rgb[:, 0].reshape(-1, 1).astype(np.uint8)
        pred_g = pred_rgb[:, 1].reshape(-1, 1).astype(np.uint8)
        pred_b = pred_rgb[:, 2].reshape(-1, 1).astype(np.uint8)
        write_ply(pred_path, [xyz_save, pred_r, pred_g, pred_b, pred_label_save, gt_label_save], common_names)

    return {
        'checkpoint_path': checkpoint_path,
        'voting_miou': voting_miou,
        'per_class_iou': per_class_iou,
        'never_voted_points': int(np.sum(vote_counts.squeeze(-1) == 0)),
        'raw_ply': raw_path,
        'pred_ply': pred_path,
    }


def save_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoints = discover_checkpoints(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    history_log = args.history_log or auto_find_history_log(checkpoints[0])
    history = parse_training_history(history_log)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"推理设备: {device}")
    print(f"待处理 checkpoint 数量: {len(checkpoints)}")
    if history_log:
        print(f"训练日志: {history_log}")
    else:
        print("训练日志: 未找到")

    pc, xyz_raw, labels_raw, feat_raw = load_scene(os.path.abspath(args.test_file))
    sub_xyz, sub_feat, nearest_idx = prepare_subscene(xyz_raw, feat_raw, labels_raw, args.grid_size)
    windows = build_windows(sub_xyz, sub_feat, args)
    print(f"有效滑窗数: {len(windows)}")

    classifier = build_model(args.model_name, device)

    summary_rows = []
    detailed_results = []
    checkpoint_names = {Path(p).name for p in checkpoints}

    for checkpoint_path in checkpoints:
        checkpoint_path = os.path.abspath(checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")

        ckpt_obj, meta = checkpoint_meta(checkpoint_path)
        print(f"\nCheckpoint: {checkpoint_path}")
        print(f"SHA1: {meta['checkpoint_sha1']}")
        print(f"存储 epoch: {meta['stored_epoch']}")
        print(f"存储 class_avg_iou: {meta['stored_class_avg_iou']}")

        result = run_voting_export(
            classifier, ckpt_obj, checkpoint_path, pc, xyz_raw, labels_raw,
            sub_xyz, nearest_idx, windows, args, output_dir
        )

        combined = {
            **meta,
            'voting_miou': result['voting_miou'],
            'never_voted_points': result['never_voted_points'],
            'raw_ply': result['raw_ply'],
            'pred_ply': result['pred_ply'],
        }
        summary_rows.append(combined)
        detailed_results.append({**combined, 'per_class_iou': result['per_class_iou']})

        print(f"Voting mIoU: {result['voting_miou']:.6f}")
        print(f"未被投票点数: {result['never_voted_points']}")
        if result['pred_ply']:
            print(f"预测着色 PLY: {result['pred_ply']}")
        if result['raw_ply']:
            print(f"原始 RGB PLY: {result['raw_ply']}")

    save_json(output_dir / 'history_best_miou.json', history)
    save_json(output_dir / 'checkpoint_export_results.json', detailed_results)
    save_csv(output_dir / 'checkpoint_export_results.csv', summary_rows)

    print("\n历史 best mIoU 轨迹:")
    if history:
        for item in history:
            eval_miou = item['eval_miou']
            eval_text = 'None' if eval_miou is None else f'{eval_miou:.6f}'
            print(f"  epoch {item['epoch']:>3d}: eval_mIoU={eval_text}, best_mIoU={item['best_miou']:.6f}")
    else:
        print("  未找到可解析的历史日志记录。")

    if len(checkpoints) == 1 and history and checkpoint_names == {'best_model.pth'} and len(history) > 1:
        print("\n注意:")
        print("  训练日志里发现多次历史最佳，但当前目录只有一个 best_model.pth。")
        print("  旧的历史最佳权重已经被覆盖，无法仅靠这个目录回放每一次历史最佳的 PLY。")
        print("  这个脚本已导出当前 best_model.pth 对应的 PLY，并保存了历史 best mIoU 记录。")


if __name__ == '__main__':
    main()
