# Quadrature-Weighted Advantage Estimation

Exponentially-fitted quadrature weights for computing Generalized Advantage Estimation (GAE) in reinforcement learning, with a Simpson-Sobolev regularization variant that selects smooth weights from the family of O(h^4)-accurate solutions.

---

## Motivation

Generalized Advantage Estimation (Schulman et al., 2016) computes advantage estimates as an exponentially-weighted sum of TD residuals:

$$A_t = \sum_{k=0}^{\infty} (\gamma\lambda)^k \delta_{t+k}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

This is a geometric series -- the RL equivalent of a first-order exponential smoother, or a forward-Euler integration scheme applied to the discounted return integral. For smooth TD residual functions, first-order accuracy leaves substantial approximation error on the table.

The insight motivating this work: the discounted advantage can be written as a continuous integral

$$A = \int_0^H e^{-\alpha t}\, \delta(t)\, dt, \qquad \alpha = -\log(\gamma\lambda)$$

and numerical quadrature provides a principled, higher-order alternative. Given a window of $H$ observed TD residuals, we ask: what weights $w_0, \dots, w_{H-1}$ give the best approximation to this integral?

The two implementations here answer that question in different ways.

---

## Files

### `simpson_h4.py` -- O(h^7) exponentially-fitted quadrature

Computes weights that are exact for any TD residual function that is a polynomial of degree 6 or lower. This is the maximum achievable accuracy for a 7-node rule.

**Approach -- Lagrange basis integration**: rather than solving a Vandermonde linear system (which is ill-conditioned for 7 nodes, with condition number ~10^9), weights are computed by directly integrating each Lagrange basis function against the exponential kernel:

$$w_k = \int_0^6 L_k(t)\, e^{-\alpha t}\, dt = \frac{1}{d_k} \sum_{j=0}^{6} c_j^{(k)}\, M_j(\alpha)$$

where $c_j^{(k)}$ are the polynomial coefficients of $P_k(t) = \prod_{j \neq k}(t - j)$ (computed exactly in integer arithmetic), $d_k = \prod_{j \neq k}(k - j)$ is an exact integer denominator, and $M_j(\alpha) = \int_0^6 t^j e^{-\alpha t}\,dt$ are the moments. This formulation has no matrix inversion and no conditioning problem.

**Moment computation**: the integration-by-parts recurrence for $M_j$ suffers catastrophic cancellation when $\alpha N$ is small (the two terms being subtracted are nearly equal and large). For $|\alpha N| < 1$ -- which covers all standard RL parameters with $\gamma, \lambda \in [0.9, 1.0]$ -- moments are computed instead via a Taylor series expansion that converges in roughly 10 terms with no cancellation. The recurrence is only used when $|\alpha N| \geq 1$, where it is numerically stable.

As $\gamma\lambda \to 1$ (no discount), the weights converge to the standard 7-point Newton-Cotes (closed) rule:

$$\mathbf{w} \to \frac{1}{140}[41,\ 216,\ 27,\ 272,\ 27,\ 216,\ 41]$$

This is a useful numerical check: the limiting weights are known analytically and provide a validation target independent of the moment computation.

### `sobolev_h4.py` -- O(h^4) with Simpson-Sobolev regularization

Imposes only 4 moment conditions (exactness up to degree 3), leaving $H - 4$ free parameters. These are determined by minimizing the Simpson-approximated $H^2$ Sobolev norm of the weight sequence:

$$J(\mathbf{w}) = \mu_0\, \mathbf{w}^\top S\, \mathbf{w} + \mu_1\, (L_1\mathbf{w})^\top S_1 (L_1\mathbf{w}) + \mu_2\, (L_2\mathbf{w})^\top S_2 (L_2\mathbf{w})$$

where $S$, $S_1$, $S_2$ are diagonal matrices of composite Simpson weights for sequences of length $H$, $H-1$, $H-2$, and $L_1$, $L_2$ are first and second finite difference operators.

The three penalty terms control:
- $\mu_0$: magnitude of the weights (prevents large canceling weights)
- $\mu_1$: variation between adjacent weights (encourages smooth decay)
- $\mu_2$: curvature of the weight sequence (prevents sharp peaks)

The solution is given in closed form by a Lagrange multiplier argument:

$$\mathbf{w}^* = R^{-1} V_4^\top (V_4 R^{-1} V_4^\top)^{-1} \mathbf{M}_4$$

where $R = \mu_0 S + \mu_1 L_1^\top S_1 L_1 + \mu_2 L_2^\top S_2 L_2$ is symmetric positive definite for any $\mu_i \geq 0$ with at least one strictly positive.

The regularization is motivated by two observations. First, a weight sequence that oscillates or has sharp peaks amplifies high-frequency noise in $\delta_t$ -- which is common in RL, where TD residuals are noisy. Second, the optimal weights should inherit the smoothness of the discount kernel $e^{-\alpha k}$; the Sobolev penalty formalizes this intuition.

The minimum stencil is $H = 5$ (one free parameter). $H = 7$ gives three free parameters and stronger smoothing. For typical RL trajectory lengths, $H = 5$ is the default.

**Note**: `compute_moments` is duplicated verbatim between `simpson_h4.py` and `sobolev_h4.py`. Factoring into a shared `utils.py` is a natural next step if the project grows.

### `sobolev_test.py` -- test suite for `sobolev_h4.py`

| Test | What it checks |
|---|---|
| `test_polynomial_exactness` | All 4 moment conditions satisfied; relative error < 1e-6 for H=5 and H=7 |
| `test_sobolev_beats_unconstrained` | Sobolev solution has strictly smaller $J(\mathbf{w})$ than the minimum-Euclidean-norm pseudoinverse solution |
| `test_monotone_decay` | All weights positive under standard RL parameters; weight profile printed |
| `test_limiting_case` | As $\alpha \to 0$, weights sum to $H - 1$ to within 1e-2; runs H=5 and H=7 |

Tests in `simpson_h4.py` cover: limiting case convergence to Newton-Cotes, polynomial exactness to relative error < 1e-6 for all 7 degrees, and a direct comparison against truncated GAE on a cubic polynomial.

---

## Benchmark

On a cubic polynomial TD residual ($\delta(t) = 1 + 0.3t - 0.05t^2 + 0.002t^3$) with $\gamma=0.99$, $\lambda=0.95$:

| Method | Error vs true integral |
|---|---|
| `simpson_h4` (O(h^7)) | 7.80e-08 |
| GAE (truncated at H=7) | 9.66e-01 |

The quadrature rule is exact for polynomials up to degree 6 by construction; the remaining error is floating-point rounding. GAE's error reflects its first-order nature -- it is not designed to approximate the integral accurately over a fixed window.

---

## Installation

```bash
pip install jax numpy
# GPU:
pip install "jax[cuda12]"
# TPU:
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

---

## Usage

```python
from sobolev_h4 import simpson_sobolev_weights
from simpson_h4 import quadrature_weights, quadrature_advantage
import jax.numpy as jnp

# O(h^4) Sobolev-regularized weights -- recommended for noisy TD residuals
w_ss = simpson_sobolev_weights(gamma=0.99, lam=0.95, H=5, mu0=1.0, mu1=1.0, mu2=1.0)

# O(h^7) weights -- maximum accuracy for smooth TD residuals
w_h7 = quadrature_weights(gamma=0.99, lam=0.95, H=7)

# Apply to a window of TD residuals
delta_window = jnp.array([0.3, 0.1, -0.2, 0.4, 0.0])   # shape (H,)
advantage_estimate = jnp.dot(w_ss, delta_window)

# Slide across a full trajectory (requires len(deltas) >= 7)
deltas = jnp.ones((20,))
advantages = quadrature_advantage(deltas, gamma=0.99, lam=0.95)  # shape (14,)
```

Run the tests:

```bash
python simpson_h4.py
python sobolev_test.py
```

---

## Relationship to Standard GAE

Standard GAE uses geometric-series weights:

$$w_k^{\text{GAE}} = (\gamma\lambda)^k$$

which correspond to a forward-Euler approximation of the discounted integral -- O(h^1) accurate. The implementations here replace this with:

| Method | Accuracy | Stencil | Notes |
|---|---|---|---|
| GAE | O(h^1) | unbounded | Standard; no stencil width |
| `sobolev_h4` | O(h^4) | H >= 5 | Smooth weights; noise-robust |
| `simpson_h4` | O(h^7) | 7 points | Maximum accuracy for H=7 |

For polynomial-like TD residuals (smooth reward and value function), the higher-order rules are exact or nearly exact. For noisy TD residuals -- common in early training -- the Sobolev regularization may reduce variance relative to the unconstrained O(h^7) rule. The tradeoff is controlled by $\mu_0, \mu_1, \mu_2$.

---

## Limitations and Open Questions

The $\mu_i$ hyperparameters have no canonical values derived from RL theory. The right values likely depend on the noise level of the TD residuals, which varies across environments and training stages. An adaptive scheme that estimates noise level and adjusts $\mu_i$ accordingly is a natural extension.

The window-based approach introduces a boundary effect: the last $H - 1$ timesteps of a trajectory cannot produce a full-window estimate. In the current implementation these are filled using standard GAE as a fallback. A tapered stencil at the boundary is an alternative not explored here.

Whether the improved approximation accuracy translates to measurable sample efficiency improvement in PPO is the empirical question this work is intended to set up.

---

## Background

The formulation draws on two bodies of work:

**Exponentially-fitted numerical methods**: quadrature rules that exactly integrate $e^{-\alpha t}$ times a polynomial, adapted to the specific decay rate of the integrand. See Iserles, "A First Course in the Numerical Analysis of Differential Equations" (Ch. 8) for the general theory.

**Sobolev-penalized approximation**: the penalty structure is adapted from Simpson-Sobolev
regularization for neural network training developed in the author's dissertation (SMU, 2026),
where the same three-term penalty -- on function value, first derivative, and second derivative
-- is applied to the output of a neural network over a discretized domain, integrated via
Simpson's rule on a 7-point stencil.

---

## Citation

If you use or build on this work:

```bibtex
@misc{margolis2025qwae,
  author = {Margolis, Daniel},
  title  = {Quadrature-Weighted Advantage Estimation},
  year   = {2025},
  url    = {https://github.com/danmm16/quadrature-advantage}
}
```