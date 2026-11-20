"""Make Fig. 2, Fig. 3 (HalfCheetah + Hopper panels) and Fig. 5, plus a final-score table.

    python plot.py --results results --out figures

Same conventions as the paper: mean over seeds, shading = 0.5 * std, curves uniformly smoothed
(the paper does not give the window; we use a centred moving average of --smooth points = 10k steps).
The authors' saved TOP-TD3 curves (../runs/<Env>-v2.csv, 10 seeds x 1000 points) are overlaid
dashed in Fig. 3 / Fig. 5 for direct comparison.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ENVS = ["HalfCheetah-v4", "Hopper-v4"]
STYLE = {  # colours chosen to match the paper's figures
    "pessimistic": ("Pessimistic", "#e8473f"),
    "optimistic": ("Optimistic", "#f5a31a"),
    "td3": ("TD3", "#e8473f"),
    "sac": ("SAC", "#f5a31a"),
    "oac": ("OAC", "#7cc8c8"),
    "top": ("TOP-TD3", "#1f4e66"),
    "authors_top": ("TOP-TD3 (authors' runs)", "#888888"),
}
FIGURES = {
    "fig2": ["pessimistic", "optimistic"],
    "fig3": ["td3", "sac", "oac", "top"],
    "fig5": ["pessimistic", "optimistic", "top"],
}


def smooth(y, k):
    if k <= 1:
        return y
    kernel = np.ones(k) / k
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, kernel, mode="valid")


def load_runs(results, tag, env):
    """-> (steps [T], returns [n_seeds, T]) or None. Only complete-length runs are stacked."""
    curves, steps = [], None
    for f in sorted(glob.glob(os.path.join(results, tag, env, "seed*.csv"))):
        d = np.genfromtxt(f, delimiter=",", names=True, dtype=None, encoding=None)
        if d.size == 0:
            continue
        s, r = np.atleast_1d(d["step"]), np.atleast_1d(d["eval_return"]).astype(float)
        if steps is None or len(s) > len(steps):
            steps = s
        curves.append(r)
    if not curves:
        return None
    T = len(steps)
    full = [c for c in curves if len(c) == T]
    if len(full) < len(curves):
        print(f"  note: {tag}/{env}: {len(curves) - len(full)} unfinished run(s) skipped")
    return steps, np.stack(full) if full else None


def load_authors(runs_dir, env):
    f = os.path.join(runs_dir, env.replace("-v4", "-v2") + ".csv")
    if not os.path.exists(f):
        return None
    r = np.loadtxt(f, delimiter=",")                  # [10, 1000], one point per 1k steps
    return np.arange(1, r.shape[1] + 1) * 1000, r


def draw(ax, steps, R, label, color, k, dashed=False):
    curves = np.stack([smooth(c, k) for c in R])
    m, s = curves.mean(0), curves.std(0)
    ax.plot(steps, m, color=color, lw=1.6, ls="--" if dashed else "-", label=f"{label} (n={len(R)})")
    if not dashed:
        ax.fill_between(steps, m - 0.5 * s, m + 0.5 * s, color=color, alpha=0.25, lw=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--runs_dir", default=os.path.join(os.path.dirname(__file__), "..", "runs"))
    p.add_argument("--out", default="figures")
    p.add_argument("--smooth", type=int, default=10)
    p.add_argument("--last", type=int, default=10, help="final score = mean of the last N evaluations")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    table = []
    for fig, tags in FIGURES.items():
        fig_h, axes = plt.subplots(1, len(ENVS), figsize=(5.2 * len(ENVS), 3.6))
        for ax, env in zip(axes, ENVS):
            for tag in tags:
                loaded = load_runs(args.results, tag, env)
                if loaded is None or loaded[1] is None:
                    continue
                steps, R = loaded
                draw(ax, steps, R, *STYLE[tag], k=args.smooth)
                if fig == "fig3":
                    final = R[:, -args.last:].mean(1)
                    table.append((env, STYLE[tag][0], final.mean(), final.std(), len(final)))
            if fig in ("fig3", "fig5"):
                a = load_authors(args.runs_dir, env)
                if a is not None:
                    draw(ax, a[0], a[1], *STYLE["authors_top"], k=args.smooth, dashed=True)
                    if fig == "fig3":
                        final = a[1][:, -args.last:].mean(1)
                        table.append((env, "TOP-TD3 (authors' runs)", final.mean(), final.std(), len(final)))
            ax.set_title(env)
            ax.set_xlabel("Time steps")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Reward")
        axes[-1].legend(fontsize=8, loc="lower right")
        fig_h.tight_layout()
        path = os.path.join(args.out, f"{fig}.png")
        fig_h.savefig(path, dpi=150)
        plt.close(fig_h)
        print("wrote", path)

    if table:
        lines = [f"Final score = mean of last {args.last} evaluations; mean ± std over seeds\n",
                 f"{'env':<16}{'method':<26}{'mean':>10}{'std':>9}{'n':>4}"]
        lines += [f"{e:<16}{m:<26}{mu:>10.0f}{sd:>9.0f}{n:>4}" for e, m, mu, sd, n in table]
        txt = "\n".join(lines)
        print(txt)
        with open(os.path.join(args.out, "final_scores.txt"), "w") as f:
            f.write(txt + "\n")


if __name__ == "__main__":
    main()
