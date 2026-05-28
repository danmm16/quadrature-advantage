"""
ppo_benchmark.py

Proof-of-concept comparison of GAE, quadrature_h7, and sobolev_h4 advantage
estimators in PPO across four environments:

  CartPole-v1          clean discrete control
  CartPole-v1 (noisy)  Gaussian reward noise; tests the noisy-TD regime
  LunarLander-v2       harder discrete control
  HalfCheetah-v4       continuous control (optional: pip install gymnasium[mujoco])

Metrics logged per update:
  episode_reward       smoothed mean reward over last 10 episodes
  adv_var              variance of raw advantage estimates in the rollout
  steps_to_solve       total env steps when smoothed reward first exceeds threshold

Results are written to benchmark_results.csv and a summary is printed at the end.

Usage:
    python ppo_benchmark.py           # full run (~1-2 hours)
    python ppo_benchmark.py --quick   # reduced steps and seeds (~15 min)
"""

import argparse
import csv
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from advantage_estimators import get_estimator, advantage_variance


# ── Configuration ─────────────────────────────────────────────────────────────

FULL_CONFIG = {
    'gamma':          0.99,
    'lam':            0.95,
    'clip_eps':       0.2,
    'lr':             3e-4,
    'n_epochs':       4,
    'minibatch':      64,
    'vf_coef':        0.5,
    'ent_coef':       0.01,
    'max_grad_norm':  0.5,
    'rollout_steps':  128,
    'hidden':         64,
    'quad_H':         7,
    'sob_H':          7,
    'sob_h7_H':       11,
    'noise_std':      1.0,
    'seeds':          [0, 1, 2],
    'envs': {
        # 'CartPole-v1':         {'total_steps': 100_000, 'threshold': 475.0},
        # 'CartPole-v1 (noisy)': {'total_steps': 100_000, 'threshold': 400.0},
        'LunarLander-v3':      {'total_steps': 300_000, 'threshold': 200.0},
        'HalfCheetah-v4':      {'total_steps': 500_000, 'threshold': 3000.0},
    },
    'estimators': ['gae', 'quadrature_h7', 'sobolev_h4', 'sobolev_h7'],
}

QUICK_CONFIG = {
    **FULL_CONFIG,
    'seeds': [0, 1],
    'envs': {
        'CartPole-v1':         {'total_steps': 30_000,  'threshold': 475.0},
        'CartPole-v1 (noisy)': {'total_steps': 30_000,  'threshold': 400.0},
        'LunarLander-v3':      {'total_steps': 100_000, 'threshold': 200.0},
    },
}


# ── Environment utilities ─────────────────────────────────────────────────────

class NoisyRewardWrapper(gym.Wrapper):
    """
    Adds zero-mean Gaussian noise to rewards.
    For CartPole (reward always 1.0), noise_std=1.0 gives SNR=1.
    This creates highly noisy TD residuals -- the primary test case for
    whether Sobolev smoothing provides practical benefit.
    """
    def __init__(self, env, noise_std=1.0, seed=None):
        super().__init__(env)
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        noisy_reward = reward + self.rng.normal(0.0, self.noise_std)
        return obs, noisy_reward, terminated, truncated, info


def make_env(env_name: str, seed: int, noise_std: float = 1.0):
    """
    Create and seed an environment. Returns (env, obs_dim, act_dim, continuous).
    Returns None for env if the environment cannot be created (e.g. missing mujoco).
    """
    base_name = env_name.replace(' (noisy)', '')
    noisy = '(noisy)' in env_name

    try:
        env = gym.make(base_name)
    except Exception as e:
        print(f"  Skipping {env_name}: {e}")
        return None, None, None, None

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    if noisy:
        env = NoisyRewardWrapper(env, noise_std=noise_std, seed=seed)

    obs_dim   = env.observation_space.shape[0]
    continuous = isinstance(env.action_space, gym.spaces.Box)
    act_dim   = (env.action_space.shape[0] if continuous
                 else env.action_space.n)

    return env, obs_dim, act_dim, continuous


# ── Neural network ────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Shared-body actor-critic MLP.
    Discrete action spaces use Categorical; continuous use diagonal Normal.
    Orthogonal initialization following CleanRL convention.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64,
                 continuous: bool = False):
        super().__init__()
        self.continuous = continuous

        self.shared = nn.Sequential(
            self._layer(obs_dim, hidden),  nn.Tanh(),
            self._layer(hidden,  hidden),  nn.Tanh(),
        )
        self.actor  = self._layer(hidden, act_dim, scale=0.01)
        self.critic = self._layer(hidden, 1,       scale=1.0)

        if continuous:
            self.log_std = nn.Parameter(torch.zeros(act_dim))

    @staticmethod
    def _layer(in_f, out_f, scale=np.sqrt(2)):
        layer = nn.Linear(in_f, out_f)
        nn.init.orthogonal_(layer.weight, gain=scale)
        nn.init.zeros_(layer.bias)
        return layer

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self.shared(x)).squeeze(-1)

    def get_action_and_value(self, x: torch.Tensor, action=None):
        features = self.shared(x)
        value    = self.critic(features).squeeze(-1)

        if self.continuous:
            mean = self.actor(features)
            std  = self.log_std.exp().expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            entropy  = dist.entropy().sum(-1)
        else:
            logits = self.actor(features)
            dist   = torch.distributions.Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy  = dist.entropy()

        return action, log_prob, entropy, value


# ── PPO training ──────────────────────────────────────────────────────────────

def collect_rollout(env, model, rollout_steps: int, obs_start: np.ndarray,
                    device: torch.device, continuous: bool):
    """
    Collect a rollout of `rollout_steps` transitions.
    Returns buffer dict and the obs/done at the end of the rollout
    (needed to bootstrap the value for advantage computation).
    """
    act_shape = (env.action_space.shape if continuous else ())
    buf = {
        'obs':       np.zeros((rollout_steps, env.observation_space.shape[0]),
                              dtype=np.float32),
        'actions': np.zeros((rollout_steps,) + act_shape,
                    dtype=np.float32 if continuous else np.int64),
        'log_probs': np.zeros(rollout_steps, dtype=np.float32),
        'values':    np.zeros(rollout_steps, dtype=np.float32),
        'rewards':   np.zeros(rollout_steps, dtype=np.float32),
        'dones':     np.zeros(rollout_steps, dtype=np.float32),
    }

    obs = obs_start
    for t in range(rollout_steps):
        buf['obs'][t] = obs
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, log_prob, _, value = model.get_action_and_value(obs_t)
        buf['actions'][t]   = action.cpu().numpy().squeeze(0)
        buf['log_probs'][t] = log_prob.cpu().item()
        buf['values'][t]    = value.cpu().item()

        action_np = buf['actions'][t]
        if continuous:
            action_np = action_np.reshape(env.action_space.shape)
        obs, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated
        buf['rewards'][t] = reward
        buf['dones'][t]   = float(done)

        if done:
            obs, _ = env.reset()

    return buf, obs, done


def ppo_update(model, optimizer, buf, advantages, returns,
               cfg: dict, device: torch.device, continuous: bool):
    """
    Run n_epochs of minibatch PPO updates on the collected rollout.
    Returns mean policy loss, value loss, and entropy.
    """
    T = len(buf['rewards'])
    obs_t      = torch.tensor(buf['obs'],      dtype=torch.float32, device=device)
    actions_t = torch.tensor(buf['actions'],
                         dtype=torch.float32 if continuous else torch.int64,
                         device=device)
    old_lp_t   = torch.tensor(buf['log_probs'],dtype=torch.float32, device=device)
    adv_t      = torch.tensor(advantages,       dtype=torch.float32, device=device)
    returns_t  = torch.tensor(returns,          dtype=torch.float32, device=device)

    # Normalize advantages
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    losses_pi, losses_vf, losses_ent = [], [], []

    for _ in range(cfg['n_epochs']):
        perm = torch.randperm(T, device=device)
        for start in range(0, T, cfg['minibatch']):
            idx = perm[start:start + cfg['minibatch']]

            _, new_lp, entropy, new_val = model.get_action_and_value(
                obs_t[idx], actions_t[idx])

            ratio    = (new_lp - old_lp_t[idx]).exp()
            adv_mb   = adv_t[idx]
            loss_pi  = -torch.min(
                ratio * adv_mb,
                ratio.clamp(1 - cfg['clip_eps'], 1 + cfg['clip_eps']) * adv_mb
            ).mean()
            loss_vf  = 0.5 * (new_val - returns_t[idx]).pow(2).mean()
            loss_ent = entropy.mean()

            loss = loss_pi + cfg['vf_coef'] * loss_vf - cfg['ent_coef'] * loss_ent
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
            optimizer.step()

            losses_pi.append(loss_pi.item())
            losses_vf.append(loss_vf.item())
            losses_ent.append(loss_ent.item())

    return (np.mean(losses_pi), np.mean(losses_vf), np.mean(losses_ent))


def train(env_name: str, estimator_name: str, seed: int, cfg: dict,
          writer: csv.DictWriter, env_cfg: dict) -> dict:
    """
    Full PPO training run for one (environment, estimator, seed) combination.
    Logs metrics to writer at each update. Returns a summary dict.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env, obs_dim, act_dim, continuous = make_env(
        env_name, seed, cfg['noise_std'])
    if env is None:
        return {'skipped': True}

    device = torch.device('cpu')
    model  = ActorCritic(obs_dim, act_dim, cfg['hidden'], continuous).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], eps=1e-5)

    estimator_fn = get_estimator(estimator_name)
    estimator_kwargs = {}
    if estimator_name == 'quadrature_h7':
        estimator_kwargs['H'] = cfg['quad_H']
    elif estimator_name == 'sobolev_h4':
        estimator_kwargs['H'] = cfg['sob_H']
    elif estimator_name == 'sobolev_h7':
        estimator_kwargs['H'] = cfg['sob_h7_H']    

    total_steps   = env_cfg['total_steps']
    threshold     = env_cfg['threshold']
    rollout_steps = cfg['rollout_steps']
    n_updates     = total_steps // rollout_steps

    reward_window  = deque(maxlen=10)
    episode_reward = 0.0
    steps_to_solve = None

    obs, _ = env.reset(seed=seed)
    global_step = 0
    t0 = time.time()

    for update in range(n_updates):
        # Linear lr annealing (CleanRL convention)
        frac = 1.0 - update / n_updates
        optimizer.param_groups[0]['lr'] = frac * cfg['lr']

        # Collect rollout
        buf, obs_end, done_end = collect_rollout(
            env, model, rollout_steps, obs, device, continuous)
        global_step += rollout_steps

        # Track episode rewards
        ep_r = 0.0
        for t in range(rollout_steps):
            ep_r += buf['rewards'][t]
            if buf['dones'][t]:
                reward_window.append(ep_r)
                ep_r = 0.0

        smoothed_reward = float(np.mean(reward_window)) if reward_window else 0.0

        if steps_to_solve is None and len(reward_window) >= 5:
            if smoothed_reward >= threshold:
                steps_to_solve = global_step

        # Bootstrap value for advantage computation
        with torch.no_grad():
            obs_t  = torch.tensor(obs_end, dtype=torch.float32, device=device).unsqueeze(0)
            next_v = model.get_value(obs_t).cpu().item()

        # Compute advantages
        advantages, returns = estimator_fn(
            buf['rewards'], buf['values'], buf['dones'], next_v,
            cfg['gamma'], cfg['lam'], **estimator_kwargs)

        adv_var = float(np.var(advantages))

        # PPO update
        pi_loss, vf_loss, ent = ppo_update(
            model, optimizer, buf, advantages, returns, cfg, device, continuous)

        obs = obs_end

        # Log
        row = {
            'env':            env_name,
            'estimator':      estimator_name,
            'seed':           seed,
            'global_step':    global_step,
            'smoothed_reward': smoothed_reward,
            'adv_var':        adv_var,
            'pi_loss':        pi_loss,
            'vf_loss':        vf_loss,
            'entropy':        ent,
        }
        writer.writerow(row)

        if update % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{estimator_name:>15s}] step={global_step:>7d} "
                  f"reward={smoothed_reward:>8.2f}  adv_var={adv_var:>8.4f}  "
                  f"({elapsed:.0f}s)")

    env.close()
    final_reward = float(np.mean(reward_window)) if reward_window else 0.0
    return {
        'env':          env_name,
        'estimator':    estimator_name,
        'seed':         seed,
        'final_reward': final_reward,
        'steps_to_solve': steps_to_solve if steps_to_solve else total_steps,
        'solved':       steps_to_solve is not None,
    }


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(cfg: dict, out_file: str = 'benchmark_results.csv'):
    """
    Run the full benchmark: all (env, estimator, seed) combinations.
    Skips HalfCheetah if mujoco is not installed.
    Writes per-update metrics to out_file and prints a summary table.
    """
    fieldnames = ['env', 'estimator', 'seed', 'global_step',
                  'smoothed_reward', 'adv_var', 'pi_loss', 'vf_loss', 'entropy']

    summaries = []
    with open(out_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for env_name, env_cfg in cfg['envs'].items():
            for estimator_name in cfg['estimators']:
                for seed in cfg['seeds']:
                    print(f"\n{'='*60}")
                    print(f"  {env_name}  |  {estimator_name}  |  seed={seed}")
                    print(f"{'='*60}")
                    summary = train(env_name, estimator_name, seed,
                                    cfg, writer, env_cfg)
                    if not summary.get('skipped'):
                        summaries.append(summary)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"{'Environment':<25} {'Estimator':<17} "
          f"{'Mean final reward':>18} {'Solved':>8} {'Mean steps to solve':>20}")
    print(f"{'-'*70}")

    # Group by (env, estimator)
    from collections import defaultdict
    groups = defaultdict(list)
    for s in summaries:
        groups[(s['env'], s['estimator'])].append(s)

    for env_name in cfg['envs']:
        for estimator_name in cfg['estimators']:
            key = (env_name, estimator_name)
            if key not in groups:
                continue
            runs = groups[key]
            mean_reward  = np.mean([r['final_reward']    for r in runs])
            mean_steps   = np.mean([r['steps_to_solve']  for r in runs])
            n_solved     = sum(1 for r in runs if r['solved'])
            print(f"  {env_name:<23} {estimator_name:<17} "
                  f"{mean_reward:>18.2f} {n_solved:>5}/{len(runs):<2} "
                  f"{mean_steps:>20.0f}")
        print()

    print(f"\nPer-update metrics saved to: {out_file}")
    print("Plot with: python plot_benchmark.py  (if available)")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Reduced steps and seeds for fast iteration (~15 min)')
    parser.add_argument('--out', type=str, default='benchmark_results.csv',
                        help='Output CSV filename')
    args = parser.parse_args()

    cfg = QUICK_CONFIG if args.quick else FULL_CONFIG

    print("Quadrature vs GAE Advantage Estimator Benchmark")
    print(f"Config: {'QUICK' if args.quick else 'FULL'}")
    print(f"Environments: {list(cfg['envs'].keys())}")
    print(f"Estimators:   {cfg['estimators']}")
    print(f"Seeds:        {cfg['seeds']}")
    print(f"Output:       {args.out}")
    total = (len(cfg['envs']) * len(cfg['estimators']) * len(cfg['seeds']))
    print(f"Total runs:   {total}")

    # Check dependencies
    print("\nChecking dependencies...")
    deps_ok = True
    for pkg in ['torch', 'gymnasium']:
        try:
            __import__(pkg)
            print(f"  {pkg}: ok")
        except ImportError:
            print(f"  {pkg}: MISSING -- pip install {pkg}")
            deps_ok = False
    if not deps_ok:
        sys.exit(1)

    run_benchmark(cfg, out_file=args.out)