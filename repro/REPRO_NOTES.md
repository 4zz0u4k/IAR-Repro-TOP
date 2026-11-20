# Reproduction of TOP (Moskovitz et al., NeurIPS 2021): Fig. 2 and Fig. 3 on HalfCheetah and Hopper

The authors' code in `../mujoco/` is left untouched as the reference.
Everything used for the reproduction is in this folder.

## Files
| File | What it is |
|---|---|
| `train.py` | One run (algorithm × environment × seed). It writes `results/<tag>/<env>/seed<k>.csv` with one row per evaluation. |
| `agents.py` | TOP-TD3 (also used for the fixed-β variants), TD3, SAC and OAC |
| `utils.py` | Networks, numpy replay buffer, quantile Huber loss |
| `bandit.py` | The authors' bandit, copied unchanged |
| `check_env.py` | Checks that the v4 environments behave like v2 (about 1 minute, CPU) |
| `smoke_test.sh` | Short runs of every method plus a timing run. **Run this locally first.** |
| `run_all.sh` | Launches all runs (resumable) |
| `plot.py` | Fig. 2, Fig. 3 and Fig. 5 panels, plus a table of final scores. Overlays the authors' saved TOP curves from `../runs/`. |

## Run matrix (10 seeds, 0–9)
| Figure | Run tag | Command flags | Runs |
|---|---|---|---|
| Fig. 2 | `optimistic` | `--algo top --fixed_beta 0` | 20 |
| Fig. 2 | `pessimistic` | `--algo top --fixed_beta -1` | 20 |
| Fig. 3 | `top` | `--algo top` (bandit over β ∈ {−1, 0}) | 20 |
| Fig. 3 | `td3` / `sac` / `oac` | `--algo td3`, `--algo sac`, `--algo oac` | 60 |
| | | **Total** | **120** |

Fig. 5 (Pessimistic vs Optimistic vs TOP) needs no extra runs.

## Changes from the authors' code
| # | Change | Changes results? | Why |
|---|---|---|---|
| 1 | `DOPE_Agent` → `TOP_Agent` | No | The original crashes on start |
| 2 | gym → gymnasium API; `real_done = terminated` | No (see note A) | The old gym + mujoco-py stack no longer installs cleanly |
| 3 | GIF and checkpoint code removed | No | It crashed at step 1M on headless machines; the checkpoint code referred to a missing `log_alpha` |
| 4 | `--fixed_beta` option (single-arm bandit) | No; this is the paper's definition | Needed for Fig. 2 |
| 5 | Baselines added (see the next section) | Yes: baseline numbers depend on the implementation | Not included in the TOP repo |
| 6 | **Evaluation runs on its own environment** | **Yes, on Hopper** (HalfCheetah unaffected) | The original reset the *training* environment mid-episode (note B) |
| 7 | `-v2` → `-v4` environments | Negligibly (see the v2 vs v4 section) | mujoco-py and MuJoCo 2.x are hard to install |
| 8 | 10k random warm-up steps (the code had 25k) | Yes, mostly early in the curve | Follows the paper's Table 3, for all methods |
| 9 | 2 hidden layers of 256 (the code had 3) | Possibly | Follows the paper's Table 3, for all methods. Note: Ball's TD3 and SAC code also uses 3 layers. `--n_hidden 3` restores it. |
| 10 | Replay buffer: numpy arrays instead of a Python deque | No | Same data and the same uniform sampling *without replacement*; only faster (Ball's own buffer) |
| 11 | Logs go to CSV instead of TensorBoard; no `.item()` sync on every step | No | Easier plotting; faster |
| – | Bandit (`bandit.py`) | Unchanged | It differs from Eq. 11 in the paper (decay 0.9, feedback normalised over a 5-episode window, division by exp(w)). Worth discussing in the report. |
| – | Seeds 0–9; evaluation env seeded with `seed + 10000` | No (statistically) | The authors' seeds are unknown |

**Note A (episode endings).** An episode ends either because the robot fell (`terminated`: the future is worth 0) or because the 1000-step limit was hit (`truncated`: not a real end). The original code rebuilt this distinction by hand. Gymnasium reports the two separately, and we store only `terminated` as a real end. The only case that differs: if Hopper falls on exactly step 1000, the original counted it as a time-out and we count it as a fall. This is negligible.

**Note B (evaluation bug).** Every 1000 steps the original called `evaluate_agent(env, ...)` on the training environment. That resets the robot and runs a full test episode. Training then resumes on the test episode's final state, while its own `state` variable still holds the old observation. The result is one fake transition stored in the buffer (usually a fake "fall" on Hopper), a training episode cut short, and wrong feedback to the bandit. HalfCheetah episodes always last exactly 1000 steps, so there the test always happens between episodes and nothing breaks. Ball's TD3 and SAC have the same bug, so all of the paper's Hopper curves include it.

## Baselines: where they come from
| Method | Source | Notes |
|---|---|---|
| TD3 | `fiorenza2/TD3_PyTorch` (the paper says baselines are "the same as Ball and Roberts [2021]") | Maths copied as is. TOP itself was built on this code. |
| SAC | `fiorenza2/SAC_PyTorch` | Automatic temperature, target entropy = −dim(A), log-std clamped to [−20, 2] |
| OAC | The official `microsoft/oac-explore` needs ray, a modified rlkit and old gym/mujoco-py, so it is not usable as is. **Standard approach:** SAC + OAC's exploration rule, ported line by line from `optimistic_exploration.py`, with their published β_UB = 4.66 and δ = 23.53 (`reproduce.sh`). | The original OAC training loop (rlkit: 1000 env steps then 1000 updates) is replaced by our loop (1 update per step, the same ratio). |

All methods share the same training loop, evaluation, warm-up (10k), batch size (256), learning rate (3e-4), buffer size (1e6) and network size (2×256).

## v2 vs v4: checked line by line
Source compared: `openai/gym` v0.21.0 (`hopper.py`, `half_cheetah.py`, XML files) against `Gymnasium` v1.4.0 (`hopper_v4.py`, `half_cheetah_v4.py`, XML files).

| Item | HalfCheetah | Hopper |
|---|---|---|
| Reward formula | Identical (forward velocity − 0.1·‖a‖²) | Identical (forward velocity + 1 − 1e-3·‖a‖²; v4 still gives the +1 every step, unlike v5) |
| Termination | Identical (never) | Identical (height > 0.7, \|angle\| < 0.2, \|state[2:]\| < 100) |
| Observation | Identical (qpos[1:], qvel) | Identical (qpos[1:], clip(qvel, ±10)) |
| Reset noise, frame skip, timestep, 1000-step limit | Identical | Identical |
| Robot model (XML) | **Byte-for-byte identical** | **Rewritten** from `coordinate="global"` to local coordinates, because MuJoCo 3 removed global coordinates. Every joint and geom position and size was recomputed in global coordinates and matches v2 exactly. Masses and inertias are computed from the geoms, so they are unchanged. |
| Physics engine | MuJoCo 2.x (mujoco-py) → MuJoCo 3.x | same |

**Conclusion:** no major difference. The only differences are numerical ones inside the physics engine (MuJoCo 2 vs 3) and Hopper's XML being rewritten into an equivalent form. State this in the report. `check_env.py` re-checks the reward and termination formulas at run time.

## Order of operations
1. Locally: `pip install -r requirements.txt` (or `uv add` the same packages), then `cd repro && bash smoke_test.sh`.
   It must print `ALL CHECKS PASSED`, every smoke CSV must have 3 rows, and it reports steps/s.
2. Use the measured steps/s to plan the full runs: hours per run ≈ 1e6 / steps_per_s / 3600.
3. `bash run_all.sh` (set `PARALLEL`, `NGPU`, `SEEDS`, and `ONLY=fig2|fig3|all`).
4. `python plot.py --results results --out figures`.
