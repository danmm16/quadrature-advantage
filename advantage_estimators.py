"""
advantage_estimators.py

Common interface for GAE, quadrature_h7, and sobolev_h4 advantage estimators.
Each estimator takes a rollout buffer and returns (advantages, returns).

Episode boundary handling: if a done flag appears within a quadrature window,
the window is truncated at that boundary and the remaining steps use GAE fallback.
This is conservative but correct -- it never mixes advantage information across
episode boundaries.
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quadrature_h7 import quadrature_weights as _get_quad_weights
from sobolev_h4 import simpson_sobolev_weights as _get_sob_weights
from sobolev_h7 import simpson_sobolev_h7_weights as _get_sob_h7_weights


# ── Core estimators ───────────────────────────────────────────────────────────

def gae(
    rewards:    np.ndarray,
    values:     np.ndarray,
    dones:      np.ndarray,
    next_value: float,
    gamma:      float = 0.99,
    lam:        float = 0.95,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standard Generalized Advantage Estimation (Schulman et al., 2016).
    O(h^1) accuracy; geometric-series weighting of TD residuals.

    Returns:
        advantages: shape (T,)
        returns:    shape (T,)  (advantages + values, for value loss target)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float64)
    acc = 0.0
    for t in reversed(range(T)):
        nv = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv * (1.0 - dones[t]) - values[t]
        acc = delta + gamma * lam * (1.0 - dones[t]) * acc
        advantages[t] = acc
    return advantages, advantages + values


def _gae_fallback(deltas, dones, t, H, gamma, lam):
    """
    GAE applied to a partial window starting at t, truncated at the first
    done boundary. Used as the boundary fallback for quadrature methods.
    """
    T = len(deltas)
    acc = 0.0
    for k in range(min(H, T - t)):
        if k > 0 and dones[t + k - 1]:
            break
        acc += (gamma * lam) ** k * deltas[t + k]
    return acc


def _window_end(dones, t, H, T):
    """
    Returns the exclusive end index of the largest window starting at t
    that does not cross an episode boundary. Maximum size is H.
    A done at position t+k means position t+k+1 starts a new episode,
    so the usable window is [t, t+k+1).
    """
    end = min(t + H, T)
    for k in range(end - t - 1):    # check positions t through t+H-2
        if dones[t + k]:
            return t + k + 1        # can include delta[t+k], stop before t+k+1
    return end


def quadrature_h7(
    rewards:    np.ndarray,
    values:     np.ndarray,
    dones:      np.ndarray,
    next_value: float,
    gamma:      float = 0.99,
    lam:        float = 0.95,
    H:          int   = 7,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    O(h^7) exponentially-fitted quadrature advantage estimation.
    Uses Lagrange basis integration; exact for polynomials up to degree H-1.
    Falls back to GAE at episode boundaries and trajectory end.

    Returns:
        advantages: shape (T,)
        returns:    shape (T,)
    """
    T   = len(rewards)
    nv  = np.append(values[1:], next_value)
    deltas = rewards + gamma * nv * (1.0 - dones) - values

    w = np.array(_get_quad_weights(gamma, lam, H=H), dtype=np.float64)
    advantages = np.zeros(T, dtype=np.float64)

    for t in range(T):
        end = _window_end(dones, t, H, T)
        if end - t == H:
            advantages[t] = np.dot(w, deltas[t:t + H])
        else:
            advantages[t] = _gae_fallback(deltas, dones, t, H, gamma, lam)

    return advantages, advantages + values


def sobolev_h4(
    rewards:    np.ndarray,
    values:     np.ndarray,
    dones:      np.ndarray,
    next_value: float,
    gamma:      float = 0.99,
    lam:        float = 0.95,
    H:          int   = 5,
    mu0:        float = 1.0,
    mu1:        float = 1.0,
    mu2:        float = 1.0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    O(h^4) Simpson-Sobolev regularized advantage estimation.
    Minimizes the Simpson H^2 Sobolev norm of the weight sequence subject
    to 4 moment conditions. Falls back to GAE at episode boundaries.

    Returns:
        advantages: shape (T,)
        returns:    shape (T,)
    """
    T   = len(rewards)
    nv  = np.append(values[1:], next_value)
    deltas = rewards + gamma * nv * (1.0 - dones) - values

    w = np.array(_get_sob_weights(gamma, lam, H=H, mu0=mu0, mu1=mu1, mu2=mu2),
                 dtype=np.float64)
    advantages = np.zeros(T, dtype=np.float64)

    for t in range(T):
        end = _window_end(dones, t, H, T)
        if end - t == H:
            advantages[t] = np.dot(w, deltas[t:t + H])
        else:
            advantages[t] = _gae_fallback(deltas, dones, t, H, gamma, lam)

    return advantages, advantages + values


def sobolev_h7(
    rewards:    np.ndarray,
    values:     np.ndarray,
    dones:      np.ndarray,
    next_value: float,
    gamma:      float = 0.99,
    lam:        float = 0.95,
    H:          int   = 11,
    mu0:        float = 1.0,
    mu1:        float = 1.0,
    mu2:        float = 1.0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    O(h^7) Simpson-Sobolev regularized advantage estimation.
    H=11 recommended (peak smoothing improvement at 4 free parameters).
    Falls back to GAE at episode boundaries.
    """
    T   = len(rewards)
    nv  = np.append(values[1:], next_value)
    deltas = rewards + gamma * nv * (1.0 - dones) - values

    w = np.array(_get_sob_h7_weights(gamma, lam, H=H, mu0=mu0, mu1=mu1, mu2=mu2),
                 dtype=np.float64)
    advantages = np.zeros(T, dtype=np.float64)

    for t in range(T):
        end = _window_end(dones, t, H, T)
        if end - t == H:
            advantages[t] = np.dot(w, deltas[t:t + H])
        else:
            advantages[t] = _gae_fallback(deltas, dones, t, H, gamma, lam)

    return advantages, advantages + values


# ── Registry ──────────────────────────────────────────────────────────────────

ESTIMATORS = {
    'gae':           gae,
    'quadrature_h7': quadrature_h7,
    'sobolev_h4':    sobolev_h4,
    'sobolev_h7':    sobolev_h7,
}


def get_estimator(name: str):
    """Returns the advantage estimator function by name."""
    if name not in ESTIMATORS:
        raise ValueError(f"Unknown estimator '{name}'. Choose from: {list(ESTIMATORS)}")
    return ESTIMATORS[name]


# ── Variance analysis ─────────────────────────────────────────────────────────

def advantage_variance(
    rewards:    np.ndarray,
    values:     np.ndarray,
    dones:      np.ndarray,
    next_value: float,
    gamma:      float = 0.99,
    lam:        float = 0.95,
    H_quad:     int   = 7,
    H_sob:      int   = 5,
) -> dict:
    """
    Compute advantage variance for all three estimators on the same rollout.
    Returns a dict of variances -- useful for the noisy-TD regime analysis.
    """
    results = {}
    for name, fn in ESTIMATORS.items():
        kwargs = {}
        if name == 'quadrature_h7':
            kwargs['H'] = H_quad
        elif name == 'sobolev_h4':
            kwargs['H'] = H_sob
        advs, _ = fn(rewards, values, dones, next_value, gamma, lam, **kwargs)
        results[name] = float(np.var(advs))
    return results