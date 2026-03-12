# RAGTTA Prostate Online (Supplementary)

This folder is a clean, standalone package for the **prostate online Table 4 result** used in the paper supplementary experiment.

## What this package is for

- Re-check the final reported numbers for 5 target sets:
  - BMC, I2CVB, UCL, BIDMC, HK
- Map each table number to the exact `run.log` line.
- Provide a minimal upload checklist for GitHub.

## Included files

- `code/`: minimal runnable code snapshot for prostate online table reproduction.
- `upload_manifest.md`: what to upload / what not to upload.
- `results/table4_ragtta_online_prostate.csv`: final table numbers.
- `results/table4_ragtta_online_prostate_logs.md`: source log paths and lines.
- `scripts/extract_table4_from_logs.sh`: parse `dice_final` and `hd_final` from logs and regenerate CSV.

## Quick verify

```bash
cd <repo_root>/ragtta_prostate_online_release
bash scripts/extract_table4_from_logs.sh <log_root>/log_dir_online_restore_v1
cat results/table4_ragtta_online_prostate.csv
```

## Notes

- This folder does **not** include raw data, full log directories, or large model weights.
- It is intended for reproducibility of the reported supplementary table numbers.
