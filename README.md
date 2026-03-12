# RAGTTA: Reliability-Aware Guided Continual Test-Time Adaptation with Multimodal Priors for Medical Anatomical Region Segmentation

## Project Overview

RAGTTA is a continual test-time adaptation (CTTA) framework for medical anatomical region segmentation under cross-center distribution shifts.  
It is designed for annotation-free online deployment, where models must adapt safely to evolving target-domain streams without offline retraining.

RAGTTA addresses three key CTTA challenges in clinical streaming settings:

1. unreliable sample-driven updates that cause drift,
2. unstable use of structural/anatomical priors,
3. lack of recovery once harmful updates are introduced.

To tackle this, RAGTTA introduces a reliability-aware closed-loop adaptation strategy with:

- sample-wise reliability gating for update triggering and strength control,
- risk-triggered rollback/recovery to prevent error accumulation,
- teacher-student stabilized online updates,
- controllable multimodal prior injection for semantic-structural consistency,
- Shift-Decomposed Adaptation (SDA) to balance adaptation plasticity and stability.

This repository provides minimal reproducibility packages for:

- OD/OC segmentation experiments,
- Prostate online adaptation experiments.

## Environment

Tested base environment:

- Python 3.7.9
- PyTorch 1.13.1
- TorchVision 0.14.1
- NumPy 1.19.2

Quick install (pip):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==1.13.1 torchvision==0.14.1
pip install numpy==1.19.2 scipy pandas matplotlib seaborn tqdm requests
pip install SimpleITK nibabel medpy opencv-python scikit-image einops batchgenerators
```

## Repository layout

- `ragtta_odoc_min_release/`  
  Minimal OD/OC package, including table data for the three OD/OC tables and renamed runnable entry scripts.

- `code/`, `results/`, `scripts/`  
  Minimal prostate online package with Table 4 result files and verification script.

- `upload_manifest.md`  
  Upload/include guidance for compact sharing.

## Prostate quick verify

```bash
cd <repo_root>
bash scripts/extract_table4_from_logs.sh <log_root>/log_dir_online_restore_v1
cat results/table4_ragtta_online_prostate.csv
```

## OD/OC quick verify

```bash
cd <repo_root>/ragtta_odoc_min_release
bash scripts/verify_odoc_tables.sh
```

## Notes

- This repository excludes raw data, full experiment logs, and large model weights.
- Paths in documents are intentionally generic for portability.
