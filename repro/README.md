# Running the TOP reproduction (Fig. 2 + Fig. 3, HalfCheetah and Hopper) on SLURM

**What this does:** it trains 120 independent RL agents and plots the paper's Figures 2 and 3. Each run is one method on one environment with one seed, trained for 1M steps:

| Figure | Methods | Runs |
|---|---|---|
| Fig. 2 | Optimistic, Pessimistic | 2 × 2 envs × 10 seeds = 40 |
| Fig. 3 | TOP-TD3, TD3, SAC, OAC | 4 × 2 envs × 10 seeds = 80 |

**Target node:** partition `pgi15`, node `pgi15-20`, with 8× RTX PRO 6000 Blackwell (96 GB) and 128 cores.
All 120 runs go at once: 1 CPU core each, 15 per GPU.

> ⚠️ This code has **never been run on this cluster**. Always do the smoke test (step 2) before the full run (step 3).

---

## 0. Versions (important)
| What | Version |
|---|---|
| Python | 3.12 (3.10 or 3.11 also work) |
| PyTorch | **≥ 2.7, CUDA 12.8 build** (older builds do not work on Blackwell GPUs) |
| NVIDIA driver | **≥ 570** (check with `nvidia-smi`) |
| gymnasium | **1.4.0** (pinned in `requirements.txt`) |
| mujoco | the version pinned in `requirements.txt` |
| Environments | `HalfCheetah-v4`, `Hopper-v4`. **Never `-v5`.** |

**Do not upgrade anything partway through.** All 120 runs must use the same software.
The job prints the versions at the top of its output file.

---

## 1. Setup (once, on the login node, ~5 min)
```bash
git clone -b repro <REPO_URL> TOP && cd TOP   # the code is on the `repro` branch
uv venv .venv --python 3.12              # no uv?  ->  python3.12 -m venv .venv
source .venv/bin/activate
uv pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128
uv pip install -r repro/requirements.txt  # without uv: use plain `pip install` for both lines
cd repro
```
If `python3.12` is not found, load it first, e.g. `module load python/3.12`. Module names depend on the cluster; `module avail python` lists them.

The job script expects the environment at `TOP/.venv`. If yours is elsewhere, add `,VENV=/path/to/venv` to the `--export=` list in the `sbatch` commands below.

## 2. Smoke test (~10–15 min)
```bash
sbatch --export=ALL,MODE=smoke slurm_top.sbatch
squeue -u $USER                     # wait until the job is gone
cat slurm_top-repro_<JOBID>.out
```
Check the output for:
- [ ] `ALL CHECKS PASSED`
- [ ] no Python errors or tracebacks
- [ ] every smoke CSV listed with **3 rows** (the timing run has **20**)
- [ ] a line `=> hours per 1M-step run ~ X`. This is the real speed with 15 runs per GPU, and the full job takes roughly this long.

**Send the whole `.out` file back before continuing.**

## 3. Full run (all 120 runs; ≈ the "hours per run" number from step 2)
```bash
sbatch --export=ALL,MODE=full slurm_top.sbatch
```
Options, added inside `--export=` and separated by commas:
| Option | Effect |
|---|---|
| `ONLY=fig2` | only the 40 Fig. 2 runs (`ONLY=fig3` for the 80 Fig. 3 runs) |
| `SEEDS="0 1 2 3 4"` | 5 seeds instead of 10 |
| `PARALLEL=100` | fewer runs at once (use this if the smoke test showed the node is overloaded) |

## 4. While it runs
```bash
tail -f slurm_top-repro_<JOBID>.out                                       # progress
find results -name '*.csv' -exec wc -l {} + | awk '$1==1001' | wc -l      # finished runs (target: 120)
tail logs/top_Hopper-v4_s0.log                                            # one run's log
scancel <JOBID>                                                           # stop the job
```
**If the job crashes or hits the time limit:** submit the same `sbatch` command again. Finished runs are skipped and unfinished ones restart from zero.

## 5. Results (all inside `repro/`)
| Path | Content |
|---|---|
| `figures/fig2.png` | Fig. 2: Pessimistic vs Optimistic |
| `figures/fig3.png` | Fig. 3: TD3 / SAC / OAC / TOP-TD3, with the authors' own TOP curves dashed |
| `figures/fig5.png` | Pessimistic vs Optimistic vs TOP |
| `figures/final_scores.txt` | Final score per method (mean ± std over seeds) |
| `results/<method>/<env>/seed<k>.csv` | Raw data, one row per evaluation (every 1000 steps) |

To re-make the plots: `python plot.py --results results --out figures`.

---

## Troubleshooting
| Problem | Fix |
|---|---|
| `sbatch` refuses `--mem=0` | Edit `slurm_top.sbatch`: `--mem=200G` |
| `sbatch` complains about `--gres` | Run `sinfo -o "%N %G"` to see the GPU type, then use `--gres=gpu:<type>:8` |
| `CUDA error: no kernel image` / `sm_120 not compatible` | The PyTorch build is too old. Reinstall it with the `cu128` line from step 1 |
| `ModuleNotFoundError` | The venv is not at `TOP/.venv`. Pass `VENV=...` (see step 1) |
| `SOME CHECKS FAILED` | Stop and send the output back. Do not launch the full run |

What was changed relative to the authors' code, and why: see `REPRO_NOTES.md`.
