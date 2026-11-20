"""One training run = one algorithm x one environment x one seed.

Examples (Fig. 2 / Fig. 3):
    python train.py --algo top --fixed_beta 0  --env Hopper-v4 --seed 0   # "Optimistic"
    python train.py --algo top --fixed_beta -1 --env Hopper-v4 --seed 0   # "Pessimistic"
    python train.py --algo top --env Hopper-v4 --seed 0                   # TOP-TD3
    python train.py --algo td3 | sac | oac --env Hopper-v4 --seed 0       # baselines

Training loop = original mujoco/train_top_agent.py, with these changes (see REPRO_NOTES.md):
  * gymnasium API; `real_done` = gymnasium's `terminated` (time-limit cut-offs are not deaths)
  * evaluation uses its OWN environment (the original reset the training env mid-episode)
  * 10k random-action steps (paper Table 3; the code default was 25k)
  * no GIF / checkpoint code; results written to a CSV (one row per evaluation)
"""
import argparse
import csv
import json
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RescaleAction

from agents import ALGOS, make_agent


def make_env(env_id: str) -> gym.Env:
    env = gym.make(env_id)                       # includes TimeLimit(1000)
    return RescaleAction(env, min_action=-1.0, max_action=1.0)


def evaluate(agent, eval_env: gym.Env, n_episodes: int) -> float:
    """Deterministic policy on a SEPARATE env, so training episodes are never disturbed."""
    total = 0.0
    for _ in range(n_episodes):
        state, _ = eval_env.reset()
        done = False
        while not done:
            state, reward, terminated, truncated, _ = eval_env.step(agent.get_action(state, deterministic=True))
            total += reward
            done = terminated or truncated
    return total / n_episodes


def run_tag(args) -> str:
    if args.algo != "top" or args.fixed_beta is None:
        return args.algo
    return {0.0: "optimistic", -1.0: "pessimistic"}.get(args.fixed_beta, f"top_beta{args.fixed_beta:g}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=ALGOS, default="top")
    p.add_argument("--env", default="HalfCheetah-v4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total_steps", type=int, default=int(1e6))
    p.add_argument("--n_random_actions", type=int, default=10_000)   # paper Table 3
    p.add_argument("--n_collect_steps", type=int, default=1_000)     # paper Table 3
    p.add_argument("--eval_every", type=int, default=1_000)
    p.add_argument("--n_evals", type=int, default=1)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--n_hidden", type=int, default=2)                # paper: 256:256
    p.add_argument("--batch_size", type=int, default=256)
    # TOP
    p.add_argument("--fixed_beta", type=float, default=None, help="None = bandit over {-1,0}")
    p.add_argument("--n_quantiles", type=int, default=50)
    p.add_argument("--bandit_lr", type=float, default=0.1)
    # OAC (oac-explore reproduce.sh values)
    p.add_argument("--beta_UB", type=float, default=4.66)
    p.add_argument("--delta", type=float, default=23.53)
    # infrastructure (does not change the algorithm)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--threads", type=int, default=1, help="torch CPU threads per run")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()
    assert args.n_collect_steps > args.batch_size

    torch.set_num_threads(args.threads)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)

    # ---- seeding (same calls as the original, gymnasium style)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env, eval_env = make_env(args.env), make_env(args.env)
    env.action_space.seed(args.seed)
    state, _ = env.reset(seed=args.seed)
    eval_env.reset(seed=args.seed + 10_000)       # different stream from training, fixed per seed

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = make_agent(args.algo, args.seed, state_dim, action_dim, device, args)

    # ---- output
    tag = run_tag(args)
    out_dir = os.path.join(args.out_dir, tag, args.env)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"seed{args.seed}.csv")
    with open(os.path.join(out_dir, f"seed{args.seed}.json"), "w") as f:
        json.dump({**vars(args), "tag": tag, "device_used": str(device)}, f, indent=2)
    fields = ["step", "eval_return", "train_return", "n_episodes", "beta_mean", "arm_probs",
              "q_loss", "pi_loss", "elapsed_s", "steps_per_s"]
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()

    # ---- training loop
    ep_info = agent.start_episode()
    episode_reward, finished_returns, betas = 0.0, [], [ep_info.get("beta", np.nan)]
    last = {}
    t0 = t_last = time.time()
    for step in range(1, args.total_steps + 1):
        if step < args.n_random_actions:
            action = env.action_space.sample()
        else:
            action = agent.get_action(state)
        nextstate, reward, terminated, truncated, _ = env.step(action)
        # time-limit cut-off (truncated) is NOT a real terminal state
        agent.replay_pool.push(state, action, reward, nextstate, terminated)
        state = nextstate
        episode_reward += reward

        if step > args.n_collect_steps:
            last = agent.optimize()

        if terminated or truncated:
            agent.end_episode(episode_reward)          # TOP: bandit feedback = R_m - R_{m-1}
            finished_returns.append(episode_reward)
            episode_reward = 0.0
            state, _ = env.reset()
            ep_info = agent.start_episode()            # TOP: sample beta for the new episode
            betas.append(ep_info.get("beta", np.nan))

        if step % args.eval_every == 0:
            now = time.time()
            row = {
                "step": step,
                "eval_return": evaluate(agent, eval_env, args.n_evals),
                "train_return": float(np.mean(finished_returns)) if finished_returns else np.nan,
                "n_episodes": len(finished_returns),
                "beta_mean": float(np.nanmean(betas)) if args.algo == "top" else np.nan,
                "arm_probs": " ".join(f"{x:.3f}" for x in agent.arm_probs()) if args.algo == "top" else "",
                "q_loss": float(last["q_loss"]) if "q_loss" in last else np.nan,
                "pi_loss": float(last["pi_loss"]) if "pi_loss" in last else np.nan,
                "elapsed_s": round(now - t0, 1),
                "steps_per_s": round(args.eval_every / max(now - t_last, 1e-9), 1),
            }
            writer.writerow(row)
            csv_file.flush()
            t_last = now
            finished_returns, betas = [], [betas[-1]]
            if step % (10 * args.eval_every) == 0:
                print(f"[{tag} {args.env} s{args.seed}] step {step:>8d}  eval {row['eval_return']:9.1f}  "
                      f"{row['steps_per_s']:7.1f} steps/s  elapsed {row['elapsed_s'] / 3600:5.2f} h", flush=True)

    csv_file.close()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
