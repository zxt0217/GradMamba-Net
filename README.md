# GradMamba-Net 

This repository contains the official release for the paper:

**Geometric Enhancement and Feature Reconstruction for Mamba-Based LiDAR Point Cloud Semantic Segmentation**

This release focuses on the **Toronto3D** experiment reported in the paper. It provides the training code, the released checkpoint, and auxiliary scripts for scene-level export and visualization.
The released checkpoint corresponds to the main Toronto3D result reported in the paper and is selected by the highest validation block-level mIoU on the held-out `L002` split.

## Scope

This release corresponds to the **main Toronto3D experiment**.

Included:
- model training on Toronto3D
- block-level validation during training
- released checkpoint for the main result
- optional scene-level export and visualization scripts

Not included:
- unrelated historical baselines
- obsolete experiment artifacts
- raw-data preprocessing scripts

## Environment

The release follows an environment consistent with the 3D-UMamba / PointMamba setup.

- **OS**: Ubuntu 20.04
- **GPU**: NVIDIA RTX 4080 SUPER
- **Python**: 3.9
- **PyTorch**: 1.13.1+cu117
- **CUDA**: 11.7

For a cleaner installation, we recommend installing PyTorch first and then the remaining dependencies:

```bash
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

Notes:
- `mamba-ssm` is required for `diffconv_umamba`.
- The Toronto3D `grid_subsampling` C++ wrapper is optional. If unavailable, the code falls back to the Python/NumPy implementation.

## Data Preparation

Prepare the dataset as follows:

```text
GradMamba-Net/
└── data/
    ├── Toronto3D_blocks/
    │   ├── <preprocessed_block_file_1>
    │   ├── <preprocessed_block_file_2>
    │   └── ...
    └── Toronto_3D/
        ├── L001.ply
        ├── L002.ply
        ├── L003.ply
        ├── L004.ply
        └── Colors.xml
```

Important:
- `Toronto3D_blocks/` should contain the preprocessed training/test blocks used by this release.
- Files whose names contain `L002` are used as the held-out split.
- `Toronto_3D/` contains the original scene files used for scene-level export and visualization.

## Quick Start

Start training with:

```bash
python train_Toronto3D.py \
  --gpu 0 \
  --data_root ./data/Toronto3D_blocks \
  --log_dir gradmamba_toronto3d_main
```

The default released setting in `train_Toronto3D.py` uses:
- `model=diffconv_umamba`
- `epoch=150`
- `batch_size=8`
- `weighted_loss=True`
- `fence_weight_factor=1.6`
- `fence_sample_boost=1.3`
- `scan_directions=4`

Training outputs are saved to:

```text
log/toronto3d_seg/gradmamba_toronto3d_main/
├── checkpoints/
│   └── best_model.pth
├── logs/
│   └── diffconv_umamba.txt
└── diffconv_umamba.py
```

`best_model.pth` is selected automatically by the **highest validation block-level mIoU** on the held-out `L002` split.

## Released Checkpoint

We provide the checkpoint for the main Toronto3D result.

- **File**: `diffconv_best.zip`
- **Checkpoint inside**: `best_model.pth`
- **Selection rule**: best validation mIoU on the `L002` split
- **Baidu Netdisk**: https://pan.baidu.com/s/1yFi_RGZ3K3pbrRoFKwbTDw?pwd=1234
- **Extraction code**: `1234`


Please place the extracted checkpoint at:

```text
log/toronto3d_seg/gradmamba_toronto3d_main/checkpoints/best_model.pth
```

## Optional Scene-Level Export

If you would like to generate scene-level PLY outputs, run:

```bash
python tools/vote_toronto3d.py \
  --checkpoint ./log/toronto3d_seg/gradmamba_toronto3d_main/checkpoints/best_model.pth \
  --test_file ./data/Toronto_3D/L002.ply \
  --output ./outputs/L002_vote \
  --scan_directions 4
```

Typical output files:
- `./outputs/L002_vote_rawrgb.ply`
- `./outputs/L002_vote_predrgb.ply`

To convert the exported PLY to the official Toronto3D palette, run:

```bash
python tools/recolor_to_official_toronto3d.py \
  --input ./outputs/L002_vote_predrgb.ply \
  --output ./outputs/L002_vote_officialrgb.ply \
  --colors_xml ./data/Toronto_3D/Colors.xml \
  --label_field label
```

## Reported Main Result

The released checkpoint corresponds to the main Toronto3D result reported in the paper.

- **Dataset**: Toronto3D
- **Main metric**: mIoU
- **Reported result**: 82.3 mIoU
- **Checkpoint selection**: highest validation block-level mIoU on `L002`

## Main Workflow

1. Prepare the preprocessed Toronto3D block data in `data/Toronto3D_blocks/` and the original scene files in `data/Toronto_3D/`.
2. Train the model with the command above, or download the released checkpoint.
3. Use `best_model.pth`, which is selected by the best validation mIoU on `L002`.
4. Optionally generate scene-level PLY outputs for visualization.

## Citation

If you find this repository useful, please cite the paper.

```bibtex
@article{gradmambanet2026,
  title   = {Geometric Enhancement and Feature Reconstruction for Mamba-Based LiDAR Point Cloud Semantic Segmentation},
  author  = {<Author Names>},
  note    = {Under review at The Visual Computer},
  year    = {2026}
}
```

## Acknowledgement

This project builds upon the 3D-UMamba baseline and follows an environment setup consistent with PointMamba. Please also cite the original Toronto3D dataset and relevant baseline works when using this release.


