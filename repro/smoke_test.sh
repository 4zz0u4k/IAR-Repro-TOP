#!/usr/bin/env bash
# Quick end-to-end test of every method. Run LOCALLY from the repro/ folder:
#     bash smoke_test.sh            (uses GPU if available)
#     DEVICE=cpu bash smoke_test.sh
#
# 1) env checks, 2) each of the 6 run types for 3k steps on Hopper (finishes in a few minutes),
# 3) a 20k-step TOP run on HalfCheetah: its steps_per_s column tells you the REAL speed,
#    i.e. how long a 1M-step run will take on that machine.
set -euo pipefail
DEVICE=${DEVICE:-auto}
OUT=smoke_results
rm -rf "$OUT"

python check_env.py

for spec in "top --fixed_beta 0" "top --fixed_beta -1" "top" "td3" "sac" "oac"; do
  echo "== smoke: $spec"
  python train.py --algo $spec --env Hopper-v4 --seed 0 --total_steps 3000 \
      --n_random_actions 1500 --eval_every 1000 --device "$DEVICE" --out_dir "$OUT"
done

echo "== timing run: TOP, HalfCheetah, 20k steps"
python train.py --algo top --env HalfCheetah-v4 --seed 0 --total_steps 20000 \
    --eval_every 1000 --device "$DEVICE" --out_dir "$OUT"

echo
echo "== results (each file should have 3 rows; timing file 20 rows)"
find "$OUT" -name '*.csv' | sort | while read -r f; do echo "$(($(wc -l < "$f") - 1)) rows  $f"; done
echo
echo "== speed after learning starts (steps/s, last 5 rows of the timing run):"
tail -n 5 "$OUT/top/HalfCheetah-v4/seed0.csv" | cut -d, -f1,10
echo "hours per 1M-step run  ~=  1e6 / steps_per_s / 3600   (evaluation time is already included)"
echo "run this once alone, then again with several copies at once, to see how speed drops when sharing"
