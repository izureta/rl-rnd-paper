#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    compare = os.path.join(root, "compare_paper_style.py")

    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=150_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--skip-train", action="store_true")
    p.add_argument(
        "--metric",
        type=str,
        default="success_rate",
        choices=("success_rate", "best"),
    )
    args = p.parse_args()

    experiments = [
        (
            "SparseGridLarge-v0",
            "runs/exp_sparse_grid_large",
            "figures/exp_sparse_grid_large.png",
            "SparseGridLarge-v0 (29x29, hard exploration)",
        ),
        (
            "SparseGridMedium-v0",
            "runs/exp_sparse_grid_medium",
            "figures/exp_sparse_grid_medium.png",
            "SparseGridMedium-v0 (23x23)",
        ),
    ]

    seeds_s = [str(s) for s in args.seeds]
    for env_id, out_dir, plot_path, desc in experiments:
        cmd = [
            sys.executable,
            compare,
            "--env-id",
            env_id,
            "--out-dir",
            out_dir,
            "--plot",
            plot_path,
            "--total-timesteps",
            str(args.total_timesteps),
            "--seeds",
            *seeds_s,
            "--metric",
            args.metric,
        ]
        if args.skip_train:
            cmd.append("--skip-train")
        print("\n%s\n%s\n" % ("=" * 60, desc))
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=root, check=True)

    print(
        "\nSummaries:\n - %s\n - %s"
        % (
            os.path.join(root, "runs/exp_sparse_grid_large/paper_compare_summary.json"),
            os.path.join(root, "runs/exp_sparse_grid_medium/paper_compare_summary.json"),
        )
    )


if __name__ == "__main__":
    main()
