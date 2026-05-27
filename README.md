# Quadrature-Weighted Advantage Estimation

Exponentially-fitted quadrature weights for computing Generalized Advantage Estimation (GAE) in reinforcement learning, with Simpson-Sobolev regularization variants that select smooth weights from families of $O(h^4)$- and $O(h^7)$-accurate solutions.

---

## Motivation

Generalized Advantage Estimation (Schulman et al., 2016) computes advantage estimates as an exponentially-weighted sum of TD residuals:

$$A_t = \sum_{k=0}^{\infty} (\gamma\lambda)^k \delta_{t+k}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

This is a geometric series -- the RL equivalent of a first-order exponential smoother, or a forward-Euler integration scheme applied to the discounted return integral. For smooth TD residual functions, first-order accuracy leaves substantial approximation error on the table.

The insight motivating this work: the discounted advantage can be written as a continuous integral

$$A = \int_0^H e^{-\alpha t}\, \delta(t)\, dt, \qquad \alpha = -\log(\gamma\lambda)$$

and numerical quadrature provides a principled, higher-order alternative. Given a window of $H$ observed TD residuals, we ask: what weights $w_0, \dots, w_{H-1}$ give the best approximation to this integral?

The implementations here answer that question at two accuracy orders, with and without Sobolev regularization.

---

## Files

### `quadrature_h7.py` -- $O(h^7)$ exponentially-fitted quadrature

Computes weights that are exact for any TD residual function that is a polynomial of degree 6 or lower. This is the maximum achievable accuracy for a 7-node rule.

**Approach -- Lagrange basis integration**: rather than solving a Vandermonde linear system (ill-conditioned for 7 nodes, condition number ~10^9), weights are computed by directly integrating each Lagrange basis function against the exponential kernel:

$$w_k = \int_0^6 L_k(t)\, e^{-\alpha t}\, dt = \frac{1}{d_k} \sum_{j=0}^{6} c_j^{(k)}\, M_j(\alpha)$$

where $c_j^{(k)}$ are the polynomial coefficients of $P_k(t) = \prod_{j \neq k}(t - j)$ (computed exactly in integer arithmetic), $d_k = \prod_{j \neq k}(k - j)$ is an exact integer denominator, and $M_j(\alpha) = \int_0^6 t^j e^{-\alpha t}\,dt$ are the moments. This formulation has no matrix inversion and no conditioning problem.

**Moment computation**: the integration-by-parts recurrence for $M_j$ suffers catastrophic cancellation when $\alpha N$ is small. For $|\alpha N| < 1$ -- which covers all standard RL parameters with $\gamma, \lambda \in [0.9, 1.0]$ -- moments are computed via a Taylor series expansion that converges in roughly 10 terms with no cancellation. The recurrence is used only when $|\alpha N| \geq 1$, where it is numerically stable.

As $\gamma\lambda \to 1$ (no discount), the weights converge to the standard 7-point Newton-Cotes (closed) rule:

$$\mathbf{w} \to \frac{1}{140}[41,\ 216,\ 27,\ 272,\ 27,\ 216,\ 41]$$

### `sobolev_h4.py` -- $O(h^4)$ with Simpson-Sobolev regularization

Imposes only 4 moment conditions (exactness up to degree 3), leaving $H - 4$ free parameters determined by minimizing the Simpson-approximated $H^2$ Sobolev norm of the weight sequence:

$$J(\mathbf{w}) = \mu_0\, \mathbf{w}^\top S\, \mathbf{w} + \mu_1\, (L_1\mathbf{w})^\top S_1 (L_1\mathbf{w}) + \mu_2\, (L_2\mathbf{w})^\top S_2 (L_2\mathbf{w})$$

where $S$, $S_1$, $S_2$ are diagonal matrices of composite Simpson weights for sequences of length $H$, $H-1$, $H-2$, and $L_1$, $L_2$ are first and second finite difference operators.

The three penalty terms control:
- $\mu_0$: magnitude of the weights (prevents large canceling weights)
- $\mu_1$: variation between adjacent weights (encourages smooth decay)
- $\mu_2$: curvature of the weight sequence (prevents sharp peaks)

The closed-form solution via Lagrange multipliers is:

$$\mathbf{w}^* = R^{-1} V_4^\top (V_4 R^{-1} V_4^\top)^{-1} \mathbf{M}_4$$

where $R = \mu_0 S + \mu_1 L_1^\top S_1 L_1 + \mu_2 L_2^\top S_2 L_2$ is symmetric positive definite.

Minimum stencil: $H = 5$ (one free parameter). $H = 7$ gives three free parameters and stronger smoothing.

### `sobolev_h7.py` -- O(h^7) with Simpson-Sobolev regularization

The same Sobolev framework applied at $O(h^7)$ accuracy: imposes 7 moment conditions (exactness up to degree 6), leaving $H - 7$ free parameters. Requires $H \geq 8$.

This file provides the fair methodological comparison against `quadrature_h7.py`: same accuracy order, same exponential fitting, with and without Sobolev smoothing.

Tested across $H=8$ to $H=16$ ($1$ to $9$ free parameters). Two stencil sizes are of particular interest:

**H=8 (minimum stencil, 1 free parameter)**: the Sobolev penalty has almost no room to act. Improvement over the pseudoinverse baseline is only 0.10%. Weights are comparable to `quadrature_h7.py` with a slight smoothing effect.

**H=11 (recommended, 4 free parameters)**: the Sobolev penalty has meaningful influence. Improvement over baseline peaks at 1.89% -- the largest improvement across the full $H=8$ to $H=16$ sweep. Beyond H=11 the improvement declines as the wider window gives the pseudoinverse more room to find a smooth solution on its own.

### Test files

| File | Tests |
|---|---|
| `quadrature_h7.py` (self-tests) | Newton-Cotes limiting case; polynomial exactness to relative error < 1e-6 for all 7 degrees; direct comparison against GAE |
| `sobolev_test.py` | Polynomial exactness ($O(h^4)$, $H=5$ and $H=7$); Sobolev optimality; positive weights; limiting case sum |
| `sobolev_h7_test.py` | Polynomial exactness ($O(h^7)$, $H=8$ through $H=16$); Sobolev optimality with improvement percentage; weight profiles; limiting case sum; cross-method accuracy comparison |

---

## Results

### Quadrature vs GAE on a cubic polynomial

With $\gamma=0.99$, $\lambda=0.95$, $\delta(t) = 1 + 0.3t - 0.05t^2 + 0.002t^3$:

| Method | Relative error vs true integral |
|---|---|
| `quadrature_h7` ($O(h^7)$, $H=7$) | 7.80e-08 |
| `sobolev_h4` ($O(h^4)$, $H=7$) | 3.38e-08 |
| GAE (truncated at $H=7$) | 9.66e-01 |

Both quadrature methods are at floating-point noise. GAE error of 0.966 on a cubic polynomial reflects its first-order nature.

### Cross-method accuracy comparison ($O(h^7)$ methods)

All errors are relative to each method's true integral over its own window. $\gamma=0.99$, $\lambda=0.95$:

| Function | `quadrature_h7` ($H=7$) | `sobolev_h7` ($H=8$) | `sobolev_h7` ($H=11$) | GAE ($H=7$) |
|---|---|---|---|---|
| Constant | 1.25e-08 | 1.74e-08 | 9.53e-09 | 1.69e-01 |
| Linear | 6.18e-09 | 1.86e-08 | 5.04e-09 | 1.44e-01 |
| Cubic | 5.71e-09 | 2.21e-08 | 2.03e-09 | 3.32e-01 |
| Degree 6 | 1.02e-08 | 2.41e-08 | 1.51e-09 | 6.43e-01 |
| Exponential | 1.44e-08 | 1.70e-08 | 1.02e-08 | 1.82e-01 |
| Mixed polynomial | 1.11e-08 | 1.71e-08 | 9.59e-09 | 1.38e-01 |

All three quadrature methods are effectively equivalent in accuracy -- all at machine epsilon. The gap versus GAE comes from accuracy order, not stencil width.

### Sobolev smoothing effect across H=8 to H=16

Improvement of $J(\mathbf{w})$ relative to the minimum-Euclidean-norm pseudoinverse baseline:

| $H$ | Free params | Improvement |
|---|---|---|
| 8 | 1 | 0.10% |
| 9 | 2 | 1.33% |
| 10 | 3 | 1.35% |
| 11 | 4 | **1.89%** |
| 12 | 5 | 1.13% |
| 13 | 6 | 0.89% |
| 14 | 7 | 0.76% |
| 15 | 8 | 0.72% |
| 16 | 9 | 0.71% |

$H=11$ is the empirically optimal stencil for this penalty structure at these RL parameters. The improvement peaks at four free parameters before diminishing returns set in as the wider window reduces the advantage the Sobolev penalty has over the unconstrained solution.

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
from sobolev_h7 import simpson_sobolev_h7_weights
from quadrature_h7 import quadrature_weights, quadrature_advantage
import jax.numpy as jnp

# O(h^4) Sobolev weights -- recommended for noisy TD residuals
w_h4 = simpson_sobolev_weights(gamma=0.99, lam=0.95, H=5, mu0=1.0, mu1=1.0, mu2=1.0)

# O(h^7) unconstrained weights -- maximum accuracy for smooth TD residuals
w_h7 = quadrature_weights(gamma=0.99, lam=0.95, H=7)

# O(h^7) Sobolev weights -- H=11 recommended (peak smoothing at 1.89% improvement)
w_ss7 = simpson_sobolev_h7_weights(gamma=0.99, lam=0.95, H=11, mu0=1.0, mu1=1.0, mu2=1.0)

# Apply to a window of TD residuals
delta_window = jnp.array([0.3, 0.1, -0.2, 0.4, 0.0])   # shape (H,)
advantage_estimate = jnp.dot(w_h4, delta_window)

# Slide across a full trajectory (requires len(deltas) >= 7)
deltas = jnp.ones((20,))
advantages = quadrature_advantage(deltas, gamma=0.99, lam=0.95)  # shape (14,)
```

Run the tests:

```bash
python quadrature_h7.py
python sobolev_test.py
python sobolev_h7_test.py
```

---

## Relationship to Standard GAE

Standard GAE uses geometric-series weights:

$$w_k^{\text{GAE}} = (\gamma\lambda)^k$$

which correspond to a forward-Euler approximation of the discounted integral -- $O(h^1)$ accurate. The implementations here replace this with:

| Method | Accuracy | Stencil | Free params | Notes |
|---|---|---|---|---|
| GAE | $O(h^1)$ | unbounded | -- | Standard; no stencil width |
| `sobolev_h4` | $O(h^4)$ | $H \geq 5$ | $H - 4$ | Smooth weights; noise-robust |
| `quadrature_h7` | $O(h^7)$ | $H = 7$ | $0$ | Maximum accuracy; no smoothing |
| `sobolev_h7` | $O(h^7)$ | $16 \geq H \geq 8$ | $H - 7$ | H=11 recommended; peak smoothing |

---

## Limitations and Open Questions

The $\mu_i$ hyperparameters have no canonical values derived from RL theory. The right values likely depend on the noise level of the TD residuals, which varies across environments and training stages. An adaptive scheme that estimates noise level and adjusts $\mu_i$ accordingly is a natural extension.

The window-based approach introduces a boundary effect: the last $H - 1$ timesteps of a trajectory cannot produce a full-window estimate. In the current implementation these are filled using standard GAE as a fallback. A tapered stencil at the boundary is an alternative not explored here.

The Sobolev improvement percentages reported here measure smoothness of the weight sequence, not variance reduction in practice. Whether improved weight smoothness translates to reduced gradient variance and measurable sample efficiency improvement in PPO is the empirical question this work is intended to set up.

---

## Background

The formulation draws on two bodies of work:

**Exponentially-fitted numerical methods**: quadrature rules that exactly integrate $e^{-\alpha t}$ times a polynomial, adapted to the specific decay rate of the integrand. See Iserles, "A First Course in the Numerical Analysis of Differential Equations" (Ch. 8) for the general theory.

**Sobolev-penalized approximation**: the penalty structure is adapted from Simpson-Sobolev regularization for neural network training developed in the author's dissertation (SMU, 2026), where the same three-term penalty -- on function value, first derivative, and second derivative -- is applied to the output of a neural network over a discretized domain, integrated via Simpson's rule on a 7-point stencil.

---

## License

MIT

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