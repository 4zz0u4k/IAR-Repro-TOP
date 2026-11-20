#!/usr/bin/env bash
# Launch every run needed for Fig. 2 + Fig. 3 (HalfCheetah, Hopper). Run from repro/.
# DO NOT launch before smoke_test.sh passes locally.
#
#   PARALLEL=6 NGPU=1 bash run_all.sh          # e.g. local RTX 5060
#   PARALLEL=120 NGPU=4 bash run_all.sh        # e.g. 4x A100 node with >=120 free cores
#   SEEDS="0 1 2 3 4" bash run_all.sh          # 5 seeds instead of 10
#   ONLY=fig2 bash run_all.sh                  # only the 40 Fig. 2 runs
#
# Runs are independent; finished runs (CSV with 1000 rows) are skipped, so re-running resumes.
set -euo pipefail
PARALLEL=${PARALLEL:-4}
NGPU=${NGPU:-0}                      # 0 = CPU only
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
ENVS=${ENVS:-"HalfCheetah-v4 Hopper-v4"}
ONLY=${ONLY:-all}
OUT=${OUT:-results}
mkdir -p logs

declare -a SPECS
case "$ONLY" in
  fig2) SPECS=("optimistic|--algo top --fixed_beta 0" "pessimistic|--algo top --fixed_beta -1") ;;
  fig3) SPECS=("top|--algo top" "td3|--algo td3" "sac|--algo sac" "oac|--algo oac") ;;
  all)  SPECS=("optimistic|--algo top --fixed_beta 0" "pessimistic|--algo top --fixed_beta -1"
               "top|--algo top" "td3|--algo td3" "sac|--algo sac" "oac|--algo oac") ;;
  *) echo "ONLY must be fig2|fig3|all"; exit 1 ;;
esac

# GPU ids to use: the ones SLURM gave us (CUDA_VISIBLE_DEVICES) if set, else 0..NGPU-1
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else GPU_IDS=($(seq 0 $((NGPU > 0 ? NGPU - 1 : 0)))); fi
if [[ "$NGPU" -gt "${#GPU_IDS[@]}" ]]; then NGPU=${#GPU_IDS[@]}; fi

CMDS=$(mktemp)
i=0
for seed in $SEEDS; do               # seed-major order: early seeds of every method finish first
  for env in $ENVS; do
    for spec in "${SPECS[@]}"; do
      tag=${spec%%|*}; flags=${spec#*|}
      csv="$OUT/$tag/$env/seed$seed.csv"
      if [[ -f "$csv" && $(wc -l < "$csv") -ge 1001 ]]; then continue; fi
      if [[ "$NGPU" -gt 0 ]]; then dev="CUDA_VISIBLE_DEVICES=${GPU_IDS[$((i % NGPU))]} "; extra="--device cuda"; else dev=""; extra="--device cpu"; fi
      echo "${dev}python train.py $flags --env $env --seed $seed $extra --out_dir $OUT > logs/${tag}_${env}_s${seed}.log 2>&1" >> "$CMDS"
      i=$((i + 1))
    done
  done
done

echo "$(wc -l < "$CMDS") runs to launch, $PARALLEL at a time"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
xargs -P "$PARALLEL" -I{} bash -c "{}" < "$CMDS"
rm -f "$CMDS"
echo "done. Now: python plot.py --results $OUT"
