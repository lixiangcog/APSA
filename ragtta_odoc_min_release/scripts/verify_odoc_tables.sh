#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[Check] Table1: average over A-E vs Average_DSC"
awk -F, 'NR==1{next}
{
  calc=($2+$3+$4+$5+$6)/5;
  diff=calc-$7; if(diff<0) diff=-diff;
  printf "%-22s calc=%.2f listed=%.2f diff=%.2f\n", $1, calc, $7, diff;
}' test_results/ragtta_table1_odoc.csv

echo
echo "[Check] Table3: average over A-E vs Average_DSC"
awk -F, 'NR==1{next}
{
  calc=($5+$6+$7+$8+$9)/5;
  diff=calc-$10; if(diff<0) diff=-diff;
  printf "RG%s PU%s MP%s SDA%s calc=%.2f listed=%.2f diff=%.2f\n", $1,$2,$3,$4,calc,$10,diff;
}' test_results/ragtta_table3_odoc_ablation.csv

echo
echo "[Check] Table2: R1/R2/R3 domain avg + overall avg + degradation"
awk -F, 'NR==1{next}
{
  r1=($2+$3+$4+$5+$6)/5;
  r2=($8+$9+$10+$11+$12)/5;
  r3=($14+$15+$16+$17+$18)/5;
  all=(r1+r2+r3)/3;
  deg=r1-all;
  d1=r1-$7; if(d1<0)d1=-d1;
  d2=r2-$13; if(d2<0)d2=-d2;
  d3=r3-$19; if(d3<0)d3=-d3;
  d4=all-$20; if(d4<0)d4=-d4;
  if($21!="-"){
    d5=deg-$21; if(d5<0)d5=-d5;
    printf "%-14s R1d=%.2f R2d=%.2f R3d=%.2f ALLd=%.2f DEGD=%.2f\n", $1,d1,d2,d3,d4,d5;
  }else{
    printf "%-14s R1d=%.2f R2d=%.2f R3d=%.2f ALLd=%.2f DEGD=n/a\n", $1,d1,d2,d3,d4;
  }
}' test_results/ragtta_table2_odoc_long_term.csv
