# RAGTTA OD/OC Minimal Release

This folder is a minimal OD/OC code package for paper-stage sharing.
It is isolated from the main repository to avoid breaking existing code.

## Included

- `code/test_time_training_online/ragtta_odoc_online.py` (renamed copy from `sptta_RIGA_online.py`)
- `code/test_time_training_online/vptta_odoc_online.py` (renamed copy from `vptta_RIGA_online.py`)
- Required dependency folders: `models/`, `utils/`, `datasets/dataloaders/`, `test_time_training_online/`
- Config script: `code/scripts/odoc_tta_configs.py`
- Table data (from your 3 OD/OC tables):
  - `test_results/ragtta_table1_odoc.csv`
  - `test_results/ragtta_table2_odoc_long_term.csv`
  - `test_results/ragtta_table3_odoc_ablation.csv`

## Quick consistency check

```bash
cd ragtta_odoc_min_release
bash scripts/verify_odoc_tables.sh
```

## Notes

- This package uses copy+rename only in this folder.
- Original files in the main repository are untouched.
- If needed for anonymous review, remove personal paths/IDs before final upload.
