#!/usr/bin/env python3

import argparse
import json
import os
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# More frequent updates + slightly higher LR — usually reaches high success rate by ~100–150k env steps
ROLLOUT_LEN = 1024
MINIBATCH = 128
PPO_EPOCHS = 5
GAE_LAM = 0.95
CLIP = 0.2
VF_COEF = 0.5
LR = 4.5e-4
RND_LR = 1.2e-3
MAX_GRAD_NORM = 0.5
INT_ADV_COEF = 5.5  # intrinsic advantage scale vs extrinsic (ext coeff = 1)


def _reachable(wall, goal, h, w):
    start = (1, 1)
    q = deque([start])
    seen = {start}
    while q:
        y, x = q.popleft()
        if (y, x) == goal:
            return True
        for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not wall[ny, nx] and (ny, nx) not in seen:
                seen.add((ny, nx))
                q.append((ny, nx))
    return False


def _wall_with_obstacles(size, rng_seed, density):
    rng = np.random.RandomState(rng_seed)
    h = w = size
    wall = np.zeros((h, w), dtype=np.bool_)
    wall[0, :] = wall[-1, :] = True
    wall[:, 0] = wall[:, -1] = True
    goal = (h - 2, w - 2)
    start = (1, 1)
    inner = [(y, x) for y in range(1, h - 1) for x in range(1, w - 1) if (y, x) not in (start, goal)]
    rng.shuffle(inner)
    obstacles = []
    for y, x in inner:
        if rng.rand() < density:
            wall[y, x] = True
            obstacles.append((y, x))
    if not _reachable(wall, goal, h, w):
        rng.shuffle(obstacles)
        for y, x in obstacles:
            wall[y, x] = False
            if _reachable(wall, goal, h, w):
                break
    return wall


class SparseGridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, size=23, view=1, max_steps=600, maze_seed=0, obstacle_density=0.14):
        super().__init__()
        self.size = size
        self.view = view
        self.max_steps = max_steps
        self._steps = 0
        h = w = size
        self.wall = _wall_with_obstacles(size, maze_seed, obstacle_density)
        self.goal = (h - 2, w - 2)
        flat = (2 * view + 1) ** 2 * 2
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(flat,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self.pos = (1, 1)

    def _get_obs(self):
        h = w = self.size
        v = self.view
        py, px = self.pos
        ch0, ch1 = [], []
        for dy in range(-v, v + 1):
            for dx in range(-v, v + 1):
                ny, nx = py + dy, px + dx
                if ny < 0 or nx < 0 or ny >= h or nx >= w:
                    ch0.append(1.0)
                    ch1.append(0.0)
                else:
                    is_wall = float(self.wall[ny, nx])
                    goal_here = 1.0 if (not self.wall[ny, nx] and (ny, nx) == self.goal) else 0.0
                    ch0.append(is_wall)
                    ch1.append(goal_here)
        return np.array(ch0 + ch1, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        self.pos = (1, 1)
        return self._get_obs(), {}

    def step(self, action):
        h = w = self.size
        dy, dx = [(-1, 0), (0, 1), (1, 0), (0, -1)][int(action)]
        ny, nx = self.pos[0] + dy, self.pos[1] + dx
        if 0 <= ny < h and 0 <= nx < w and not self.wall[ny, nx]:
            self.pos = (ny, nx)
        self._steps += 1
        terminated = self.pos == self.goal
        truncated = self._steps >= self.max_steps
        reward = 1.0 if terminated else 0.0
        return self._get_obs(), reward, terminated, truncated, {}


def make_env(env_id):
    if env_id == "SparseGridLarge-v0":
        return SparseGridEnv(size=29, view=1, max_steps=800, maze_seed=1337, obstacle_density=0.075)
    if env_id == "SparseGridMedium-v0":
        return SparseGridEnv(size=23, view=1, max_steps=550, maze_seed=42, obstacle_density=0.072)
    return SparseGridEnv()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def layer_init(layer, std=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
        )
        self.policy = layer_init(nn.Linear(hidden, n_actions), std=0.01)
        self.value_ext = layer_init(nn.Linear(hidden, 1), std=1.0)
        self.value_int = layer_init(nn.Linear(hidden, 1), std=1.0)

    def forward(self, x):
        h = self.trunk(x)
        return self.policy(h), self.value_ext(h).squeeze(-1), self.value_int(h).squeeze(-1)

    def act(self, obs):
        logits, v_ext, v_int = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, v_ext, v_int, dist.entropy()


class RND(nn.Module):
    def __init__(self, obs_dim, feat_dim=32, hidden=64):
        super().__init__()
        t = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, feat_dim), std=1.0),
        )
        p = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, feat_dim), std=1.0),
        )
        self.target = t
        self.predictor = p
        for param in self.target.parameters():
            param.requires_grad_(False)

    def forward(self, obs):
        with torch.no_grad():
            tgt = self.target(obs)
        pred = self.predictor(obs)
        return (pred - tgt).pow(2).mean(dim=-1)

    def predictor_loss(self, obs):
        tgt = self.target(obs).detach()
        pred = self.predictor(obs)
        return (pred - tgt).pow(2).mean()


class ObsRunningNorm:
    def __init__(self, dim, clip=5.0, var_floor=1e-4):
        self.dim = dim
        self.clip = clip
        self.var_floor = var_floor
        self._n = 0
        self._mean = np.zeros(dim, dtype=np.float64)
        self._m2 = np.zeros(dim, dtype=np.float64)

    def update_and_normalize(self, obs):
        x = np.asarray(obs, dtype=np.float64).ravel()
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._m2 += delta * delta2
        if self._n < 2:
            std = np.ones(self.dim, dtype=np.float64)
        else:
            var = np.maximum(self._m2 / self._n, self.var_floor)
            std = np.sqrt(var)
        z = np.clip((x - self._mean) / std, -self.clip, self.clip)
        return z.astype(np.float32)


class RunningRMS:
    def __init__(self, eps=1e-5):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x):
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        batch_count = x.size
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot
        self.mean, self.var, self.count = new_mean, m2 / tot, tot


class IntrinsicForwardNormalize:
    def __init__(self, gamma):
        self.gamma = float(gamma)
        self.rewems = None
        self.rms = RunningRMS()

    def normalize_rollout(self, raw_int):
        raw_int = np.asarray(raw_int, dtype=np.float64).ravel()
        collected = []
        for r in raw_int:
            if self.rewems is None:
                self.rewems = float(r)
            else:
                self.rewems = self.gamma * self.rewems + float(r)
            collected.append(self.rewems)
        self.rms.update(np.array(collected, dtype=np.float64))
        std = np.sqrt(self.rms.var) + 1e-8
        return (raw_int / std).astype(np.float64)


def compute_gae(rewards, values, dones, last_value, gamma, lam, episodic):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float64)
    last_gae = 0.0
    next_value = last_value
    for t in reversed(range(T)):
        mask = (1.0 - float(dones[t])) if episodic else 1.0
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    return advantages, advantages + values


def train(
    env_id,
    total_timesteps,
    seed,
    use_rnd,
    out_path,
    rnd_coef=0.28,
    ent_coef=0.032,
    intrinsic_clip=0.45,
    gamma=0.99,
    normalize_obs=True,
    intrinsic_only=False,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    agent = ActorCritic(obs_dim, n_actions).to(device)
    opt_agent = optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

    rnd_module = None
    opt_rnd = None
    obs_norm_rnd = None
    if use_rnd:
        rnd_module = RND(obs_dim).to(device)
        opt_rnd = optim.Adam(rnd_module.parameters(), lr=RND_LR, eps=1e-5)
        obs_norm_rnd = ObsRunningNorm(obs_dim)
    obs_norm_policy = ObsRunningNorm(obs_dim) if normalize_obs else None

    int_forward = IntrinsicForwardNormalize(gamma) if use_rnd else None
    intrinsic_only = bool(intrinsic_only and use_rnd)

    global_step = 0
    episode_return = 0.0
    episode_len = 0
    returns_history = []
    steps_history = []
    best_so_far_history = []
    ep_len_history = []
    running_best_ext = -float("inf")

    raw, _ = env.reset(seed=seed)
    raw_vec = np.asarray(raw, dtype=np.float32).ravel()
    rnd_vec = obs_norm_rnd.update_and_normalize(raw_vec) if obs_norm_rnd else raw_vec.copy()
    obs = obs_norm_policy.update_and_normalize(raw_vec) if obs_norm_policy else raw_vec.copy()

    t0 = time.time()

    while global_step < total_timesteps:
        obs_l, act_l, logp_l, rext_l, done_l, vext_l, vint_l = [], [], [], [], [], [], []
        rnd_obs_l, raw_int_l = [], []

        for _ in range(ROLLOUT_LEN):
            obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action, logp, v_ext, v_int, _ = agent.act(obs_t)
                if rnd_module is not None:
                    rnd_t = torch.as_tensor(rnd_vec, device=device, dtype=torch.float32).unsqueeze(0)
                    raw_int_l.append(float(rnd_module.forward(rnd_t).cpu().numpy().reshape(-1)[0]))
                else:
                    raw_int_l.append(0.0)

            a = int(action.item())
            next_raw, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ext_r = float(reward)

            obs_l.append(obs.copy())
            if rnd_module is not None:
                rnd_obs_l.append(rnd_vec.copy())
            act_l.append(a)
            logp_l.append(float(logp.item()))
            rext_l.append(ext_r)
            done_l.append(done)
            vext_l.append(float(v_ext.item()))
            vint_l.append(float(v_int.item()))

            episode_return += ext_r
            episode_len += 1
            global_step += 1

            if done:
                running_best_ext = max(running_best_ext, episode_return)
                returns_history.append(episode_return)
                best_so_far_history.append(running_best_ext)
                ep_len_history.append(episode_len)
                steps_history.append(global_step)
                episode_return = 0.0
                episode_len = 0
                raw, _ = env.reset()
                raw_vec = np.asarray(raw, dtype=np.float32).ravel()
                rnd_vec = obs_norm_rnd.update_and_normalize(raw_vec) if obs_norm_rnd else raw_vec.copy()
                obs = obs_norm_policy.update_and_normalize(raw_vec) if obs_norm_policy else raw_vec.copy()
            else:
                raw_vec = np.asarray(next_raw, dtype=np.float32).ravel()
                rnd_vec = obs_norm_rnd.update_and_normalize(raw_vec) if obs_norm_rnd else raw_vec.copy()
                obs = obs_norm_policy.update_and_normalize(raw_vec) if obs_norm_policy else raw_vec.copy()

            if global_step >= total_timesteps:
                break

        with torch.no_grad():
            last_obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            _, last_v_ext, last_v_int = agent.forward(last_obs_t)
            last_v_ext = float(last_v_ext.item())
            last_v_int = float(last_v_int.item())

        rews_ext = np.array(rext_l, dtype=np.float64)
        raw_mse = np.array(raw_int_l, dtype=np.float64)
        if use_rnd and int_forward is not None:
            rews_int = rnd_coef * int_forward.normalize_rollout(raw_mse)
            if intrinsic_clip > 0:
                rews_int = np.clip(rews_int, -intrinsic_clip, intrinsic_clip)
        else:
            rews_int = np.zeros_like(raw_mse)

        vals_ext = np.array(vext_l, dtype=np.float64)
        vals_int = np.array(vint_l, dtype=np.float64)
        dones_np = np.array(done_l, dtype=np.bool_)

        adv_ext, ret_ext = compute_gae(rews_ext, vals_ext, dones_np, last_v_ext, gamma, GAE_LAM, True)
        adv_int, ret_int = compute_gae(rews_int, vals_int, dones_np, last_v_int, gamma, GAE_LAM, False)

        if use_rnd:
            if intrinsic_only:
                advantages = adv_int
            else:
                advantages = adv_ext + INT_ADV_COEF * adv_int
        else:
            advantages = adv_ext
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_b = torch.as_tensor(np.stack(obs_l), device=device, dtype=torch.float32)
        act_b = torch.as_tensor(act_l, device=device, dtype=torch.long)
        logp_old = torch.as_tensor(logp_l, device=device, dtype=torch.float32)
        ret_ext_b = torch.as_tensor(ret_ext, device=device, dtype=torch.float32)
        ret_int_b = torch.as_tensor(ret_int, device=device, dtype=torch.float32)
        adv_b = torch.as_tensor(advantages, device=device, dtype=torch.float32)

        n = obs_b.shape[0]
        idx = np.arange(n)

        for _ in range(PPO_EPOCHS):
            np.random.shuffle(idx)
            for s in range(0, n, MINIBATCH):
                mb = idx[s : s + MINIBATCH]
                mb_obs = obs_b[mb]
                mb_act = act_b[mb]
                mb_logp_old = logp_old[mb]
                mb_ret_ext = ret_ext_b[mb]
                mb_ret_int = ret_int_b[mb]
                mb_adv = adv_b[mb]

                logits, v_ext, v_int = agent(mb_obs)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - mb_logp_old)
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP)
                policy_loss = torch.mean(torch.max(pg1, pg2))
                if use_rnd:
                    if intrinsic_only:
                        value_loss = 0.5 * torch.mean((v_int - mb_ret_int) ** 2)
                    else:
                        value_loss = 0.5 * torch.mean((v_ext - mb_ret_ext) ** 2 + (v_int - mb_ret_int) ** 2)
                else:
                    value_loss = 0.5 * torch.mean((v_ext - mb_ret_ext) ** 2)
                loss = policy_loss + VF_COEF * value_loss - ent_coef * entropy

                opt_agent.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                opt_agent.step()

                if rnd_module is not None and opt_rnd is not None:
                    opt_rnd.zero_grad()
                    mb_rnd = torch.as_tensor(
                        np.stack([rnd_obs_l[j] for j in mb]),
                        device=device,
                        dtype=torch.float32,
                    )
                    rnd_module.predictor_loss(mb_rnd).backward()
                    nn.utils.clip_grad_norm_(rnd_module.parameters(), MAX_GRAD_NORM)
                    opt_rnd.step()

    env.close()
    elapsed = time.time() - t0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    meta = {
        "env_id": env_id,
        "total_timesteps": global_step,
        "seed": seed,
        "use_rnd": use_rnd,
        "intrinsic_only": intrinsic_only,
        "rnd_coef": rnd_coef if use_rnd else 0.0,
        "seconds": elapsed,
        "normalize_obs": normalize_obs,
        "gamma": gamma,
        "ent_coef": ent_coef,
        "intrinsic_clip": intrinsic_clip,
    }
    base, _ = os.path.splitext(out_path)
    meta_path = base + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    np.savez(
        out_path,
        steps=np.array(steps_history, dtype=np.int64),
        episode_ext_returns=np.array(returns_history, dtype=np.float64),
        best_ext_so_far=np.array(best_so_far_history, dtype=np.float64),
        episode_lengths=np.array(ep_len_history, dtype=np.int32),
    )
    print(json.dumps(meta, indent=2))
    print("Saved %s and %s" % (out_path, meta_path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="SparseGridLarge-v0")
    p.add_argument("--total-timesteps", type=int, default=150_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/run.npz")
    p.add_argument("--no-rnd", action="store_true")
    p.add_argument("--rnd-coef", type=float, default=0.6)
    p.add_argument("--intrinsic-clip", type=float, default=1.0)
    p.add_argument("--ent-coef", type=float, default=0.025)
    p.add_argument("--no-obs-norm", action="store_true")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument(
        "--intrinsic-only",
        action="store_true",
        help="Policy gradient from intrinsic advantage only (requires RND; ignores --no-rnd)",
    )
    args = p.parse_args()
    use_rnd = not args.no_rnd or args.intrinsic_only
    train(
        env_id=args.env_id,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        use_rnd=use_rnd,
        out_path=args.out,
        rnd_coef=args.rnd_coef,
        ent_coef=args.ent_coef,
        intrinsic_clip=args.intrinsic_clip,
        gamma=args.gamma,
        normalize_obs=not args.no_obs_norm,
        intrinsic_only=args.intrinsic_only,
    )


if __name__ == "__main__":
    main()
