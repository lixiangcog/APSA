# RAGTTA Minimal Reproducibility Package

This repository contains minimal code and result artifacts for two datasets:

- OD/OC segmentation
- Prostate online adaptation

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
