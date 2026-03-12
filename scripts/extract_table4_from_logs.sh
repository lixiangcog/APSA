#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/extract_table4_from_logs.sh /path/to/log_dir_online_restore_v1

ROOT="${1:-./log_dir_online_restore_v1}"
OUT_CSV="$(cd "$(dirname "$0")/.." && pwd)/results/table4_ragtta_online_prostate.csv"

declare -A LOGS
LOGS["BMC"]="$ROOT/Tent-EMA_Prostate_RUNMC2BMC_bmc_postlcc_otf_vida_noautoblend_sptta/run.log"
LOGS["I2CVB"]="$ROOT/Tent-EMA_Prostate_RUNMC2I2CVB_ellip_slicegate_restore_v1_sptta/run.log"
LOGS["UCL"]="$ROOT/Tent-EMA_Prostate_RUNMC2UCL_ucl_currcode_quick_v1_noautoblend_sptta/run.log"
LOGS["BIDMC"]="$ROOT/Tent-EMA_Prostate_RUNMC2BIDMC_preclip_restore_v1_sptta/run.log"
LOGS["HK"]="$ROOT/Tent-EMA_Prostate_RUNMC2HK_hk_unified_v3_vida_sptta/run.log"

extract_vals() {
  local file="$1"
  local line
  line="$(grep -E 'Test Avg- dice_final:' "$file" | tail -n 1)"
  python - "$line" <<'PY'
import re,sys
s=sys.argv[1]
m=re.search(r'dice_final:([0-9.]+)\+([0-9.]+)\s+hd_final:([0-9.]+)\+([0-9.]+)', s)
if not m:
    raise SystemExit("parse_failed")
d,ds,h,hs=map(float,m.groups())
print(f"{d*100:.2f},{ds*100:.2f},{h:.2f},{hs:.2f}")
PY
}

echo "dataset,dice_mean_percent,dice_std_percent,hd95_mean,hd95_std" > "$OUT_CSV"

datasets=(BMC I2CVB UCL BIDMC HK)
dice_sum=0
hd_sum=0

tmp_file="$(mktemp)"
trap 'rm -f "$tmp_file"' EXIT
> "$tmp_file"

for ds in "${datasets[@]}"; do
  f="${LOGS[$ds]}"
  if [ ! -f "$f" ]; then
    echo "missing log: $f" >&2
    exit 1
  fi
  vals="$(extract_vals "$f")"
  echo "$ds,$vals" >> "$OUT_CSV"
  echo "$vals" >> "$tmp_file"
done

read avg_d avg_ds avg_h avg_hs < <(
  awk -F, '
    {d[NR]=$1; ds[NR]=$2; h[NR]=$3; hs[NR]=$4; sd+=$1; sh+=$3}
    END{
      md=sd/NR; mh=sh/NR;
      for(i=1;i<=NR;i++){vd+=(d[i]-md)^2; vh+=(h[i]-mh)^2}
      printf "%.2f %.2f %.2f %.2f\n", md, sqrt(vd/NR), mh, sqrt(vh/NR)
    }' "$tmp_file"
)

echo "Average,$avg_d,$avg_ds,$avg_h,$avg_hs" >> "$OUT_CSV"
echo "written: $OUT_CSV"
