# DeepSFM: Structure From Motion Via Deep Bundle Adjustment

A modern PyTorch implementation of DeepSFM (ECCV 2020 Oral) - Structure From Motion Via Deep Bundle Adjustment.

## Overview

This project implements the DeepSFM architecture described in the paper "DeepSFM: Structure From Motion Via Deep Bundle Adjustment" (ECCV 2020 Oral). The implementation is built with modern PyTorch practices and follows a modular, extensible design.

**Key Features:**
- Modern PyTorch 2.0+ implementation
- Modular architecture with separate depth and pose subnets
- Support for multiple datasets (DeMoN, KITTI, custom)
- Iterative bundle adjustment inference
- Comprehensive training and evaluation pipelines
- GPU acceleration with automatic mixed precision

## Architecture

DeepSFM consists of two main components that operate in an alternating fashion:

1. **Depth Subnet (PSNet)**: Estimates dense depth maps from multi-view images
2. **Pose Subnet (PoseNet)**: Estimates relative camera poses between views

The two subnets are trained separately but can be run iteratively during inference, mimicking traditional bundle adjustment.

## Project Structure

```
deepsfm/
├── configs/              # Configuration files
├── data/                 # Data preparation scripts
├── datasets/             # Dataset loaders and interfaces
├── models/               # Neural network architectures
├── losses/               # Loss functions
├── training/             # Training scripts and utilities
├── evaluation/           # Evaluation scripts and metrics
├── utils/                # Utility functions
├── scripts/              # Utility scripts
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/deep-sfm.git
cd deep-sfm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Data Preparation
```bash
# Download and prepare DeMoN dataset (example)
python scripts/download_demon_data.py
python scripts/prepare_data.py --dataset demon
```

### Training
```bash
# Train depth subnet
python training/train_depth.py --config configs/depth_default.yaml

# Train pose subnet  
python training/train_pose.py --config configs/pose_default.yaml
```

### Evaluation
```bash
# Evaluate on test set
python evaluation/evaluate.py --config configs/eval_default.yaml
```

### Iterative Inference
```bash
# Run iterative bundle adjustment
python scripts/iterative_inference.py --input_dir data/samples/
```

## Datasets

The framework supports multiple datasets:

1. **DeMoN Dataset**: Primary dataset used in the original paper
2. **KITTI**: Outdoor driving scenes for depth estimation
3. **Custom Datasets**: Easily extensible for your own data

## Results

Performance on DeMoN dataset (to be updated with training results):

| Metric | Depth Subnet | Pose Subnet |
|--------|--------------|-------------|
| Abs Rel | - | - |
| Sq Rel | - | - |
| RMSE | - | - |
| δ<1.25 | - | - |
| Rotation Error (°) | - | - |
| Translation Error | - | - |

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@inproceedings{wei2020deepsfm,
  title={DeepSFM: Structure From Motion Via Deep Bundle Adjustment},
  author={Wei, Xingkui and Zhang, Yinda and Li, Zhuwen and Fu, Yanwei and Xue, Xiangyang},
  booktitle={European Conference on Computer Vision},
  pages={620--636},
  year={2020},
  organization={Springer}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Original DeepSFM authors for the excellent paper
- PyTorch community for amazing tools and libraries
- All contributors to open-source computer vision research