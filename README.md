# PPO + RND (Burda et al., arXiv:1810.12894)

Minimal PPO with Random Network Distillation on two custom sparse-reward grids. Compares against PPO-only.

## Dependencies

```bash
pip install numpy torch gymnasium matplotlib
```

## Environments

`SparseGridLarge-v0` / `SparseGridMedium-v0` — room with **border walls and sparse random inner obstacles** (low density so policies stabilize within **~100–150k** env steps). Start to goal is always reachable. Observation is a local 3x3 window. Reward is 1 only at the goal.

## Running

```bash
cd paper
python compare_paper_style.py --out-dir runs/repro_large --plot figures/repro_large.png
python run_two_experiments.py
```

Default `--total-timesteps` is **150000**.
