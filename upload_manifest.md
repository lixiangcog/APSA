# Upload Manifest (Prostate Supplementary Only)

## Keep (recommended)

- `test_time_training_online/`
- `models/`
- `datasets/`
- `utils/`
- `inference/`
- `loss_functions/`
- `scripts/`
- `README.md`
- `experiments.md`
- `ragtta_prostate_online_release/`

## Exclude (do not upload)

- `data/`
- `log_dir/`
- `outputs/`
- `vis_results_*/`
- `vis_prostate_*/`
- `UNet_Source_Prostate.model`
- `*.pth`, `*.pt`, `*.ckpt`, `*.model` (if large)
- `*.bak`, `*.backup_*`
- `.idea/`, `.vscode/`

## Minimum evidence for this supplementary table

- `ragtta_prostate_online_release/results/table4_ragtta_online_prostate.csv`
- `ragtta_prostate_online_release/results/table4_ragtta_online_prostate_logs.md`
