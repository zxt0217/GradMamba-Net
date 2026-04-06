import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from utils.helper_ply import read_ply, write_ply


DEFAULT_COLORS_XML = './data/Toronto_3D/Colors.xml'
DEFAULT_INPUT = './data/Toronto_3D/L002.ply'
DEFAULT_OUTPUT = './data/Toronto_3D/L002_officialrgb.ply'

COLOR_FIELDS = ('red', 'green', 'blue')


def parse_args():
    parser = argparse.ArgumentParser('Recolor Toronto3D PLY with official Colors.xml palette')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT,
                        help='single PLY file or a directory containing PLY files')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help='single output PLY path or output directory for batch mode')
    parser.add_argument('--colors_xml', type=str, default=DEFAULT_COLORS_XML,
                        help='official Toronto3D Colors.xml path')
    parser.add_argument('--label_field', type=str, default='auto',
                        help='label field to recolor; default priority: label > scalar_Label > gt_label > class')
    parser.add_argument('--suffix', type=str, default='_officialrgb',
                        help='batch mode output suffix appended before .ply')
    parser.add_argument('--keep_original_rgb', action='store_true',
                        help='preserve original RGB as orig_red/orig_green/orig_blue')
    return parser.parse_args()


def parse_official_colors(colors_xml_path):
    tree = ET.parse(colors_xml_path)
    root = tree.getroot()

    props = root.find('.//Properties')
    if props is None:
        raise ValueError(f'Colors.xml 缺少 Properties: {colors_xml_path}')

    min_value = float(props.findtext('minValue', default='0'))
    value_range = float(props.findtext('range', default='0'))
    if value_range <= 0:
        raise ValueError(f'Colors.xml range 非法: {value_range}')

    steps = []
    for step in root.findall('.//step'):
        steps.append({
            'pos': float(step.attrib['pos']),
            'rgb': np.array([int(step.attrib['r']), int(step.attrib['g']), int(step.attrib['b'])], dtype=np.uint8),
        })

    labels = [int(node.attrib['val']) for node in root.findall('.//label')]
    if not labels:
        raise ValueError(f'Colors.xml 未找到 label 定义: {colors_xml_path}')

    color_map = {}
    eps = 1e-9
    for val in labels:
        target_pos = (val - min_value) / value_range
        matched = [step['rgb'] for step in steps if abs(step['pos'] - target_pos) < eps]
        if matched:
            # 与 CloudCompare 的 step 顺序保持一致；重复位置时取最后一个
            color_map[val] = matched[-1]
            continue

        # 回退到线性插值，避免非整数标签或 XML 变体时报错
        prev_step = None
        next_step = None
        for step in steps:
            if step['pos'] <= target_pos:
                prev_step = step
            if step['pos'] >= target_pos:
                next_step = step
                break
        if prev_step is None:
            color_map[val] = steps[0]['rgb']
        elif next_step is None:
            color_map[val] = steps[-1]['rgb']
        elif abs(next_step['pos'] - prev_step['pos']) < eps:
            color_map[val] = next_step['rgb']
        else:
            alpha = (target_pos - prev_step['pos']) / (next_step['pos'] - prev_step['pos'])
            interp = (1.0 - alpha) * prev_step['rgb'].astype(np.float32) + alpha * next_step['rgb'].astype(np.float32)
            color_map[val] = np.round(interp).astype(np.uint8)

    return color_map


def resolve_label_field(pc, requested_name):
    field_names = pc.dtype.names
    if requested_name != 'auto':
        if requested_name not in field_names:
            raise KeyError(f'指定的标签字段不存在: {requested_name}, 可用字段: {field_names}')
        return requested_name

    for candidate in ('label', 'scalar_Label', 'gt_label', 'class', 'scalar_class'):
        if candidate in field_names:
            return candidate
    raise KeyError(f'未找到标签字段，可用字段: {field_names}')


def labels_to_rgb(labels, color_map):
    labels = labels.astype(np.int64)
    unique_labels = np.unique(labels)
    missing = [int(x) for x in unique_labels.tolist() if int(x) not in color_map]
    if missing:
        raise ValueError(f'Colors.xml 中没有这些标签颜色: {missing}')

    rgb = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    for label_value in unique_labels.tolist():
        mask = labels == label_value
        rgb[mask] = color_map[int(label_value)]
    return rgb


def build_output_fields(pc, new_rgb, keep_original_rgb):
    fields = []
    names = []

    original_rgb = {}
    for color_name in COLOR_FIELDS:
        if color_name in pc.dtype.names:
            original_rgb[color_name] = pc[color_name].reshape(-1, 1).astype(np.uint8)

    for name in pc.dtype.names:
        if name == 'red':
            arr = new_rgb[:, 0].reshape(-1, 1).astype(np.uint8)
        elif name == 'green':
            arr = new_rgb[:, 1].reshape(-1, 1).astype(np.uint8)
        elif name == 'blue':
            arr = new_rgb[:, 2].reshape(-1, 1).astype(np.uint8)
        else:
            arr = pc[name].reshape(-1, 1)
        fields.append(arr)
        names.append(name)

    if 'red' not in pc.dtype.names:
        fields.append(new_rgb[:, 0].reshape(-1, 1).astype(np.uint8))
        names.append('red')
    if 'green' not in pc.dtype.names:
        fields.append(new_rgb[:, 1].reshape(-1, 1).astype(np.uint8))
        names.append('green')
    if 'blue' not in pc.dtype.names:
        fields.append(new_rgb[:, 2].reshape(-1, 1).astype(np.uint8))
        names.append('blue')

    if keep_original_rgb and len(original_rgb) == 3:
        fields.extend([original_rgb['red'], original_rgb['green'], original_rgb['blue']])
        names.extend(['orig_red', 'orig_green', 'orig_blue'])

    return fields, names


def resolve_io_paths(input_arg, output_arg, suffix):
    input_path = Path(input_arg).expanduser().resolve()
    output_path = Path(output_arg).expanduser().resolve()

    if input_path.is_file():
        return [(input_path, output_path)]

    if not input_path.is_dir():
        raise FileNotFoundError(f'输入路径不存在: {input_path}')

    ply_files = sorted(p for p in input_path.glob('*.ply') if p.is_file())
    if not ply_files:
        raise FileNotFoundError(f'目录中没有找到 PLY 文件: {input_path}')

    output_path.mkdir(parents=True, exist_ok=True)
    pairs = []
    for ply_file in ply_files:
        out_file = output_path / f'{ply_file.stem}{suffix}.ply'
        pairs.append((ply_file, out_file))
    return pairs


def process_one_file(input_path, output_path, color_map, label_field_name, keep_original_rgb):
    pc = read_ply(str(input_path))
    label_field = resolve_label_field(pc, label_field_name)
    labels = pc[label_field]
    rgb = labels_to_rgb(labels, color_map)
    fields, names = build_output_fields(pc, rgb, keep_original_rgb)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_ply(str(output_path), fields, names)

    unique_labels, counts = np.unique(labels.astype(np.int64), return_counts=True)
    return {
        'input_path': str(input_path),
        'output_path': str(output_path),
        'label_field': label_field,
        'label_hist': {int(k): int(v) for k, v in zip(unique_labels.tolist(), counts.tolist())},
    }


def main():
    args = parse_args()
    color_map = parse_official_colors(os.path.abspath(args.colors_xml))

    print('官方 Colors.xml 标签颜色:')
    for label_value in sorted(color_map.keys()):
        rgb = color_map[label_value].tolist()
        print(f'  label {label_value}: rgb={rgb}')

    io_pairs = resolve_io_paths(args.input, args.output, args.suffix)

    for input_path, output_path in io_pairs:
        result = process_one_file(
            input_path=input_path,
            output_path=output_path,
            color_map=color_map,
            label_field_name=args.label_field,
            keep_original_rgb=args.keep_original_rgb,
        )
        print(f"\n输入文件: {result['input_path']}")
        print(f"输出文件: {result['output_path']}")
        print(f"使用标签字段: {result['label_field']}")
        print(f"标签统计: {result['label_hist']}")


if __name__ == '__main__':
    main()
