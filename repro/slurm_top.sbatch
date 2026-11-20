#!/usr/bin/env bash
# SLURM job for the TOP reproduction on one full node (8x RTX PRO 6000 Blackwell 96 GB, 128 cores).
#
# Submit from the repro/ folder:
#   sbatch --export=ALL,MODE=smoke slurm_top.sbatch          # ~10-15 min: env checks + short runs + speed
#   sbatch --export=ALL,MODE=full  slurm_top.sbatch          # all 120 runs (Fig. 2 + Fig. 3), then plots
#   sbatch --export=ALL,MODE=full,ONLY=fig2 slurm_top.sbatch # only the 40 Fig. 2 runs
#   sbatch --export=ALL,MODE=full,SEEDS="0 1 2 3 4" slurm_top.sbatch   # 5 seeds
#
# Re-submitting MODE=full resumes: finished runs (1000 evaluation rows) are skipped.
#
#SBATCH --job-name=top-repro
#SBATCH --partition=pgi15
#SBATCH --nodelist=pgi15-20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=slurm_%x_%j.out
# Notes on the lines above:
#  --mem=0 asks for all of the node's RAM (we need ~100 GB). If your cluster refuses it, use e.g. --mem=200G.
#  --gres=gpu:8: some clusters want the GPU type, e.g. --gres=gpu:<type>:8 (`sinfo -o "%N %G"` shows it).
#  --time=12:00:00 is a safety margin: the estimate for MODE=full is ~2-4 h (not measured yet).

set -euo pipefail
MODE=${MODE:-smoke}
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# ---- Python environment: point VENV at a venv with requirements.txt installed
#      (torch must be a CUDA 12.8+ build for Blackwell GPUs, see setup notes at the bottom)
VENV=${VENV:-../.venv}
# shellcheck disable=SC1091
source "$VENV/bin/activate"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "== job ${SLURM_JOB_ID:-local} on $(hostname), $(date)"
echo "== mode=$MODE  cpus=${SLURM_CPUS_PER_TASK:-?}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
python -c "import torch, gymnasium, mujoco; print('torch', torch.__version__, 'cuda', torch.version.cuda, \
'| gpus', torch.cuda.device_count(), '| gymnasium', gymnasium.__version__, '| mujoco', mujoco.__version__)"

# ---- always check the environments first; stop if anything is wrong
python check_env.py

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")

if [[ "$MODE" == "smoke" ]]; then
  bash smoke_test.sh
  # speed when sharing: 15 copies per GPU at once, like the real job (20k steps each)
  echo "== shared-speed test: 15 runs per GPU, 20k steps"
  rm -rf smoke_shared
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
  else GPU_IDS=($(seq 0 $((NGPU - 1)))); fi
  PIDS=()
  for ((k = 0; k < 15 * NGPU; k++)); do
    CUDA_VISIBLE_DEVICES=${GPU_IDS[$((k % NGPU))]} \
      python train.py --algo top --env HalfCheetah-v4 --seed "$k" --total_steps 20000 \
      --device cuda --out_dir smoke_shared > /dev/null 2>&1 &
    PIDS+=($!)
  done
  wait "${PIDS[@]}"
  echo "steps/s per run when sharing (last row of each run):"
  for f in smoke_shared/top/HalfCheetah-v4/seed*.csv; do tail -n 1 "$f" | cut -d, -f10; done \
    | sort -n | awk '{a[NR]=$1} END {print "  min", a[1], " median", a[int((NR+1)/2)], " max", a[NR]; \
      print "  => hours per 1M-step run ~", 1e6 / a[int((NR+1)/2)] / 3600}'
elif [[ "$MODE" == "full" ]]; then
  PARALLEL=${PARALLEL:-120} NGPU=$NGPU bash run_all.sh
  python plot.py --results results --out figures
else
  echo "MODE must be smoke or full"; exit 1
fi
echo "== finished $(date)"

# ---------------------------------------------------------------------------------------------
# One-time setup on the cluster (login node), e.g. with uv:
#   cd TOP && uv venv .venv --python 3.12 && source .venv/bin/activate
#   uv pip install torch --index-url https://download.pytorch.org/whl/cu128
#   uv pip install -r repro/requirements.txt
# or with plain pip:  python3 -m venv .venv && source .venv/bin/activate && pip install ... (same two lines)
