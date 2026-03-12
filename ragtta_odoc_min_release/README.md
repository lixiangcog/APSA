# RAGTTA OD/OC Minimal Release

This folder provides a compact OD/OC package for reproducibility of the three OD/OC tables.

## Included

- Renamed runnable entries:
  - `code/test_time_training_online/ragtta_odoc_online.py`
  - `code/test_time_training_online/vptta_odoc_online.py`
- Required dependencies:
  - `code/models/`
  - `code/utils/`
  - `code/datasets/dataloaders/`
  - `code/test_time_training_online/`
- Config script:
  - `code/scripts/odoc_tta_configs.py`
- Table data:
  - `test_results/ragtta_table1_odoc.csv`
  - `test_results/ragtta_table2_odoc_long_term.csv`
  - `test_results/ragtta_table3_odoc_ablation.csv`
- Consistency report:
  - `docs/table_consistency_report.txt`

## Quick consistency check

```bash
cd <repo_root>/ragtta_odoc_min_release
bash scripts/verify_odoc_tables.sh
```

## Notes

- This folder was prepared by copy-and-rename from the working codebase.
- The original code in parent folders is not modified by this package.
