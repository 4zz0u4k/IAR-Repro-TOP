"""Sanity checks for the gymnasium port. CPU only, ~1 minute. Run LOCALLY:
    python check_env.py

Checks that the v4 environments behave like the v2 ones the paper used:
  1. observation / action sizes match v2 (HalfCheetah 17/6, Hopper 11/3), time limit 1000
  2. HalfCheetah: never `terminated`, always `truncated` at exactly step 1000
  3. Hopper: falls -> `terminated` (a real end), and terminated/truncated are never mixed up
  4. reward = v2 formula, recomputed by hand from the simulator state
  5. same seed -> same start state (seeding works)
"""
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RescaleAction

ok = True


def check(cond, msg):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    ok &= bool(cond)


def make(env_id):
    return RescaleAction(gym.make(env_id), min_action=-1.0, max_action=1.0)


for env_id, obs_dim, act_dim in [("HalfCheetah-v4", 17, 6), ("Hopper-v4", 11, 3)]:
    print(f"\n== {env_id}")
    env = make(env_id)
    check(env.observation_space.shape == (obs_dim,), f"obs dim {env.observation_space.shape} == ({obs_dim},)")
    check(env.action_space.shape == (act_dim,), f"action dim {env.action_space.shape} == ({act_dim},)")
    check(env.spec.max_episode_steps == 1000, f"time limit {env.spec.max_episode_steps} == 1000")

    # seeding
    a, _ = env.reset(seed=123)
    b, _ = env.reset(seed=123)
    c, _ = env.reset(seed=124)
    check(np.allclose(a, b) and not np.allclose(a, c), "same seed -> same start, different seed -> different start")

    # episodes with random actions
    env.action_space.seed(0)
    lengths, term_flags, trunc_flags, reward_err = [], [], [], []
    u = env.unwrapped
    for ep in range(5):
        env.reset(seed=ep)
        t, done = 0, False
        while not done:
            act = env.action_space.sample()
            x_before = u.data.qpos[0]
            _, r, terminated, truncated, _ = env.step(act)
            t += 1
            # v2 reward formula, recomputed by hand (RescaleAction is identity: both spaces are [-1,1])
            fwd = (u.data.qpos[0] - x_before) / u.dt
            if env_id.startswith("HalfCheetah"):
                r_v2 = fwd - 0.1 * np.square(act).sum()
            else:
                r_v2 = fwd + 1.0 - 1e-3 * np.square(act).sum()
            reward_err.append(abs(r - r_v2))
            done = terminated or truncated
        lengths.append(t)
        term_flags.append(terminated)
        trunc_flags.append(truncated)

    check(max(reward_err) < 1e-6, f"reward == v2 formula (max error {max(reward_err):.2e})")
    if env_id.startswith("HalfCheetah"):
        check(all(l == 1000 for l in lengths), f"episode lengths all 1000: {lengths}")
        check(not any(term_flags) and all(trunc_flags), "never terminated, always truncated at 1000")
    else:
        check(all(l < 1000 for l in lengths), f"random Hopper falls early: lengths {lengths}")
        check(all(term_flags) and not any(trunc_flags), "fall -> terminated=True, truncated=False")
        # v2 termination rule recomputed by hand on the final state
        s = u.state_vector()
        height, ang = u.data.qpos[1:3]
        v2_done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and height > 0.7 and abs(ang) < 0.2)
        check(v2_done, "v2 termination rule agrees on the final state")
    env.close()

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED -- do not launch runs")
