#!/usr/bin/env python3

import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_ppo_rnd import train


def step_hold_interp(steps, values, grid):
    if steps.size == 0:
        return np.full(grid.shape, np.nan, dtype=np.float64)
    steps = np.asarray(steps, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    out = np.empty(grid.shape, dtype=np.float64)
    idx = np.searchsorted(steps, grid, side="right") - 1
    for i in range(grid.size):
        j = int(idx[i])
        out[i] = np.nan if j < 0 else values[j]
    return out


def load_run_y(npz_path, metric, success_window=40, success_threshold=0.5):
    d = np.load(npz_path)
    steps = d["steps"]
    er = np.asarray(d["episode_ext_returns"], dtype=np.float64)
    if metric == "success_rate":
        w = max(1, int(success_window))
        thr = float(success_threshold)
        hit = (er >= thr).astype(np.float64)
        values = np.empty_like(er)
        for i in range(er.size):
            lo = max(0, i - w + 1)
            values[i] = float(np.mean(hit[lo : i + 1]))
        return steps, values
    if metric == "best":
        if "best_ext_so_far" in d:
            values = np.asarray(d["best_ext_so_far"], dtype=np.float64)
        else:
            values = np.maximum.accumulate(er)
        return steps, values
    return steps, er


def first_success_env_steps(npz_path, threshold=0.5):
    d = np.load(npz_path)
    er = np.asarray(d["episode_ext_returns"], dtype=np.float64)
    st = np.asarray(d["steps"], dtype=np.int64)
    for i in range(er.size):
        if er[i] >= threshold:
            return int(st[i])
    return None


def summarize_first_successes(npz_paths, threshold):
    vals = []
    for p in npz_paths:
        fs = first_success_env_steps(p, threshold)
        if fs is not None:
            vals.append(fs)
    if not vals:
        return {"n_solved": 0, "mean": None, "median": None}
    arr = np.array(vals, dtype=np.float64)
    return {
        "n_solved": len(vals),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
    }


def aggregate_curves(npz_paths, grid, metric, success_window=40, success_threshold=0.5):
    rows = []
    for path in npz_paths:
        steps, rets = load_run_y(
            path,
            metric,
            success_window=success_window,
            success_threshold=success_threshold,
        )
        rows.append(step_hold_interp(steps, rets, grid))
    mat = np.vstack(rows)
    mean = np.zeros(grid.shape[0], dtype=np.float64)
    std = np.zeros(grid.shape[0], dtype=np.float64)
    for i in range(grid.shape[0]):
        col = mat[:, i]
        valid = col[~np.isnan(col)]
        if valid.size == 0:
            mean[i] = np.nan
            std[i] = np.nan
        else:
            mean[i] = float(np.mean(valid))
            std[i] = float(np.std(valid, ddof=0))
    return mean, std


def plot_comparison(grid, mean_ppo, std_ppo, mean_rnd, std_rnd, out_path, labels, mean_int=None, std_int=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = grid / 1e3
    ok = lambda a: ~np.isnan(a)

    m = ok(mean_ppo) & ok(std_ppo)
    if np.any(m):
        ax.plot(x[m], mean_ppo[m], color="#c0392b", label="PPO", linewidth=2)
        ax.fill_between(x[m], (mean_ppo - std_ppo)[m], (mean_ppo + std_ppo)[m], color="#c0392b", alpha=0.2)
    m = ok(mean_rnd) & ok(std_rnd)
    if np.any(m):
        ax.plot(x[m], mean_rnd[m], color="#2980b9", label="PPO + RND", linewidth=2)
        ax.fill_between(x[m], (mean_rnd - std_rnd)[m], (mean_rnd + std_rnd)[m], color="#2980b9", alpha=0.2)
    if mean_int is not None and std_int is not None:
        m = ok(mean_int) & ok(std_int)
        if np.any(m):
            ax.plot(x[m], mean_int[m], color="#1e8449", label="PPO + RND (intrinsic only)", linewidth=2)
            ax.fill_between(x[m], (mean_int - std_int)[m], (mean_int + std_int)[m], color="#1e8449", alpha=0.2)
    ax.set_xlabel(labels["xlabel"])
    ax.set_ylabel(labels["ylabel"])
    ax.set_title(labels.get("title", "PPO vs PPO + RND vs intrinsic-only"))
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    y_lim = labels.get("y_lim")
    if y_lim is not None:
        ax.set_ylim(y_lim)
    fig.tight_layout()
    _, ext = os.path.splitext(out_path)
    save_path = out_path + ".png" if not ext else out_path
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="SparseGridLarge-v0")
    p.add_argument("--total-timesteps", type=int, default=150_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out-dir", type=str, default="runs/paper_compare")
    p.add_argument("--plot", type=str, default="figures/sparse_grid_large_ppo_vs_rnd.png")
    p.add_argument("--rnd-coef", type=float, default=0.6)
    p.add_argument("--intrinsic-clip", type=float, default=1.0)
    p.add_argument("--grid-points", type=int, default=120)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument(
        "--metric",
        type=str,
        choices=("success_rate", "best"),
        default="success_rate",
    )
    p.add_argument("--success-window", type=int, default=40)
    p.add_argument("--success-threshold", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.025)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--no-obs-norm", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ppo_paths = []
    rnd_paths = []
    int_paths = []

    manifest = {
        "env_id": args.env_id,
        "total_timesteps": args.total_timesteps,
        "seeds": args.seeds,
        "rnd_coef": args.rnd_coef,
        "intrinsic_clip": args.intrinsic_clip,
        "gamma": args.gamma,
        "ppo_npz": [],
        "rnd_npz": [],
        "intrinsic_only_npz": [],
    }

    if not args.skip_train:
        for s in args.seeds:
            ppo_out = os.path.join(args.out_dir, "ppo_seed%s.npz" % s)
            rnd_out = os.path.join(args.out_dir, "ppo_rnd_seed%s.npz" % s)
            int_out = os.path.join(args.out_dir, "ppo_intrinsic_only_seed%s.npz" % s)
            print("\n=== PPO seed %s -> %s" % (s, ppo_out))
            train(
                env_id=args.env_id,
                total_timesteps=args.total_timesteps,
                seed=s,
                use_rnd=False,
                out_path=ppo_out,
                ent_coef=args.ent_coef,
                intrinsic_clip=args.intrinsic_clip,
                gamma=args.gamma,
                normalize_obs=not args.no_obs_norm,
            )
            print("\n=== PPO+RND seed %s -> %s" % (s, rnd_out))
            train(
                env_id=args.env_id,
                total_timesteps=args.total_timesteps,
                seed=s,
                use_rnd=True,
                out_path=rnd_out,
                rnd_coef=args.rnd_coef,
                ent_coef=args.ent_coef,
                intrinsic_clip=args.intrinsic_clip,
                gamma=args.gamma,
                normalize_obs=not args.no_obs_norm,
            )
            print("\n=== PPO+RND (intrinsic-only policy) seed %s -> %s" % (s, int_out))
            train(
                env_id=args.env_id,
                total_timesteps=args.total_timesteps,
                seed=s,
                use_rnd=True,
                out_path=int_out,
                rnd_coef=args.rnd_coef,
                ent_coef=args.ent_coef,
                intrinsic_clip=args.intrinsic_clip,
                gamma=args.gamma,
                normalize_obs=not args.no_obs_norm,
                intrinsic_only=True,
            )
            ppo_paths.append(ppo_out)
            rnd_paths.append(rnd_out)
            int_paths.append(int_out)
            manifest["ppo_npz"].append(ppo_out)
            manifest["rnd_npz"].append(rnd_out)
            manifest["intrinsic_only_npz"].append(int_out)
    else:
        for s in args.seeds:
            ppo_paths.append(os.path.join(args.out_dir, "ppo_seed%s.npz" % s))
            rnd_paths.append(os.path.join(args.out_dir, "ppo_rnd_seed%s.npz" % s))
            int_paths.append(os.path.join(args.out_dir, "ppo_intrinsic_only_seed%s.npz" % s))

    t_max = float(args.total_timesteps)
    grid = np.linspace(0.0, t_max, args.grid_points, dtype=np.float64)

    mean_ppo, std_ppo = aggregate_curves(
        ppo_paths,
        grid,
        args.metric,
        success_window=args.success_window,
        success_threshold=args.success_threshold,
    )
    mean_rnd, std_rnd = aggregate_curves(
        rnd_paths,
        grid,
        args.metric,
        success_window=args.success_window,
        success_threshold=args.success_threshold,
    )
    mean_int, std_int = aggregate_curves(
        int_paths,
        grid,
        args.metric,
        success_window=args.success_window,
        success_threshold=args.success_threshold,
    )

    metric_xlabel = "Environment steps (×10³)"
    metric_ylabel = {
        "success_rate": "Success rate",
        "best": "Best extrinsic return",
    }
    plot_path = plot_comparison(
        grid,
        mean_ppo,
        std_ppo,
        mean_rnd,
        std_rnd,
        args.plot,
        {
            "title": "PPO vs PPO + RND vs intrinsic-only policy",
            "xlabel": metric_xlabel,
            "ylabel": metric_ylabel[args.metric],
            "y_lim": (-0.02, 1.02) if args.metric == "success_rate" else None,
        },
        mean_int=mean_int,
        std_int=std_int,
    )

    summary_path = os.path.join(args.out_dir, "paper_compare_summary.json")
    manifest["plot"] = plot_path
    manifest["metric"] = args.metric
    manifest["success_window"] = args.success_window
    manifest["success_threshold"] = args.success_threshold
    manifest["first_success_env_steps_ppo"] = summarize_first_successes(ppo_paths, args.success_threshold)
    manifest["first_success_env_steps_rnd"] = summarize_first_successes(rnd_paths, args.success_threshold)
    manifest["first_success_env_steps_intrinsic_only"] = summarize_first_successes(int_paths, args.success_threshold)
    manifest["ent_coef"] = args.ent_coef
    manifest["gamma"] = args.gamma
    manifest["normalize_obs"] = not args.no_obs_norm
    manifest["final_mean_ppo_tail"] = float(np.nanmean(mean_ppo[-10:]))
    manifest["final_mean_rnd_tail"] = float(np.nanmean(mean_rnd[-10:]))
    manifest["final_mean_intrinsic_only_tail"] = float(np.nanmean(mean_int[-10:]))
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("\nPlot: %s\nSummary: %s" % (plot_path, summary_path))


if __name__ == "__main__":
    main()
