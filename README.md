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

**Approach -- Lagrange basis integration**: rather than solving a Vandermonde linear system (ill-conditioned for 7 nodes, condition number ~$10^9$), weights are computed by directly integrating each Lagrange basis function against the exponential kernel:

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

### `sobolev_h7.py` -- $O(h^7)$ with Simpson-Sobolev regularization

The same Sobolev framework applied at $O(h^7)$ accuracy: imposes 7 moment conditions (exactness up to degree 6), leaving $H - 7$ free parameters. Requires $H \geq 8$.

This file provides the fair methodological comparison against `quadrature_h7.py`: same accuracy order, same exponential fitting, with and without Sobolev smoothing.

Tested across $H=8$ to $H=16$ ($1$ to $9$ free parameters). Two stencil sizes are of particular interest:

**$H=8$ (minimum stencil, 1 free parameter)**: the Sobolev penalty has almost no room to act. Improvement over the pseudoinverse baseline is only 0.10%. Weights are comparable to `quadrature_h7.py` with a slight smoothing effect.

**$H=11$ (recommended, 4 free parameters)**: the Sobolev penalty has meaningful influence. Improvement over baseline peaks at 1.89% -- the largest improvement across the full $H=8$ to $H=16$ sweep. Beyond $H=11$ the improvement declines as the wider window gives the pseudoinverse more room to find a smooth solution on its own.

### `advantage_estimators.py` -- common PPO interface

Wraps GAE, `quadrature_h7`, `sobolev_h4`, and `sobolev_h7` behind a single function signature for use inside a PPO training loop. Handles episode boundary detection -- if a done flag appears within a quadrature window, the window is truncated at that boundary and the remaining steps use GAE as a fallback. Exposes an `ESTIMATORS` registry and an `advantage_variance` utility for comparing all estimators on the same rollout.

### `ppo_benchmark.py` -- PPO training comparison

Full PPO implementation comparing all four estimators across environments. Logs smoothed episode reward and advantage variance per update to `benchmark_results.csv`. Supports LunarLander-v3, HalfCheetah-v4 (requires `gymnasium[mujoco]`), and a noisy-reward CartPole variant for the noisy-TD-residual regime. Run with `--quick` for a faster 2-seed reduced-step version.

### `plot_benchmark.py` -- result visualization

Reads `benchmark_results.csv` and saves four figures to `plots/`: learning curves with shaded ±1 std across seeds, advantage variance over training on a log scale, a grouped bar chart of final reward, and a side-by-side clean vs noisy CartPole advantage variance comparison.

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

### Sobolev smoothing effect across $H=8$ to $H=16$

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

### PPO benchmark -- full results

Four environments, four estimators, 3 seeds each. PPO with rollout length 128, $gamma=0.99$, $ambda=0.95$.

**The headline finding is not that quadrature beats GAE -- it does not, consistently. The headline finding is that the Sobolev penalty is necessary, not optional.** The pure $O(h^7)$ rule without regularization matches or approaches GAE on simple discrete environments and fails catastrophically on continuous control. `sobolev_h4` is the robust method across all environments tested.

#### CartPole-v1 (clean rewards)

All four methods reach similar final reward (~50-75 range). The key signal is advantage variance: `quadrature_h7` and `sobolev_h4` start at roughly 1.2 at step 128 vs GAE's 16.5 -- a 14x reduction from the first rollout. This reflects the higher-order accuracy of the quadrature estimators on the smooth, near-polynomial TD residuals in CartPole.

#### CartPole-v1 (noisy rewards, σ=1.0)

The noise regime separates the methods. `sobolev_h4` and GAE finish at approximately equal final reward (~55). `quadrature_h7` degrades ~40% relative to GAE, finishing around 33. The unregularized $O(h^7)$ weights amplify the reward noise in the TD residuals; the Sobolev penalty suppresses this. `sobolev_h4` achieves the lowest advantage variance throughout and maintains reward parity with GAE -- confirming the theoretical prediction.

#### LunarLander-v3

All methods remain in negative reward territory at 300k steps (LunarLander typically requires more steps to solve). GAE, `quadrature_h7`, and `sobolev_h4` finish comparably at approximately -155 to -175 mean final reward. `sobolev_h7` trends worse over the training run. `sobolev_h4` maintains lower advantage variance throughout -- roughly one order of magnitude below GAE on the log scale.

#### HalfCheetah-v4

The most consequential result. `quadrature_h7` experiences catastrophic variance explosions in early training -- spikes to $10^4$ to $10^5$ on the log scale, roughly 100-1000x above GAE. These are not logging noise; they represent individual rollouts where the unregularized $O(h^7)$ weights, applied to large continuous-control TD residuals, produce degenerate advantage estimates. Training never recovers. Mean final reward for `quadrature_h7` is approximately -80 vs GAE's -10 -- an 8x reward gap caused entirely by the absence of regularization.

`sobolev_h4` does not exhibit this instability. Its variance stays in the 10-20 range throughout, comparable to GAE, and its learning curve tracks GAE closely (final reward approximately -15). The Sobolev penalty's practical role is preventing weight configurations that catastrophically amplify large TD residuals, not only smoothing small ones.

`sobolev_h7` is intermediate -- no catastrophic explosion (the smoothing prevents it), but substantially worse than `sobolev_h4` (final reward approximately -55). The $H=11$ stencil sees more of the trajectory, which works against stability in the high-variance HalfCheetah environment.

#### Summary table

| Environment | GAE | Quadrature h7 | Sobolev h4 | Sobolev h7 |
|---|---|---|---|---|
| CartPole (clean) | ~50 | ~42 | ~58 | ~53 |
| CartPole (noisy) | ~55 | ~33 | ~55 | ~44 |
| LunarLander-v3 | ~-160 | ~-155 | ~-175 | ~-200 |
| HalfCheetah-v4 | ~-10 | ~-80 | ~-15 | ~-55 |

Higher is better. HalfCheetah numbers are mean final reward over last 10% of training, 3 seeds. 500k steps on HalfCheetah is short -- none of the methods approach convergence. The variance result is more robust to this concern than the reward result.

#### Advantage variance summary

Across all environments, `sobolev_h4` produces the lowest or near-lowest advantage variance. On CartPole, the early-training advantage variance is 4-14x lower than GAE. On LunarLander, it is approximately one order of magnitude lower throughout. On HalfCheetah, it stays in the same range as GAE while `quadrature_h7` explodes. The consistent variance reduction across diverse environments is the strongest empirical result in this work.

Run the benchmark to reproduce:

```bash
pip install torch gymnasium
pip install "gymnasium[box2d]"      # LunarLander
pip install "gymnasium[mujoco]"     # HalfCheetah (optional)
python ppo_benchmark.py             # full run (~2 hours)
python ppo_benchmark.py --quick     # 2 seeds, reduced steps (~15 min)
python plot_benchmark.py            # generate plots from results CSV
```

---

## Installation

**Quadrature and Sobolev methods** (JAX only):

```bash
pip install jax numpy
# GPU:
pip install "jax[cuda12]"
# TPU:
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**PPO benchmark** (additional dependencies):

```bash
pip install torch pandas matplotlib gymnasium
pip install "gymnasium[box2d]"      # LunarLander-v3
pip install "gymnasium[mujoco]"     # HalfCheetah-v4 (optional)
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

Run the PPO benchmark:

```bash
python ppo_benchmark.py --quick     # fast (~15 min)
python ppo_benchmark.py             # full (~2 hours)
python plot_benchmark.py            # generate plots
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
| `quadrature_h7` | $O(h^7)$ | $7$ | $0$ | Maximum accuracy; no smoothing |
| `sobolev_h7` | $O(h^7)$ | $16 \geq H \geq 8$ | $H - 7$ | $H=11$ recommended; peak smoothing |

---

## Limitations and Open Questions

The $\mu_i$ hyperparameters have no canonical values derived from RL theory. The right values likely depend on the noise level of the TD residuals, which varies across environments and training stages. An adaptive scheme that estimates noise level and adjusts $\mu_i$ accordingly is a natural extension.

The window-based approach introduces a boundary effect: the last $H - 1$ timesteps of a trajectory cannot produce a full-window estimate. In the current implementation these are filled using standard GAE as a fallback. A tapered stencil at the boundary is an alternative not explored here.

The benchmark runs HalfCheetah-v4 for 500k steps, which is short for this environment -- none of the methods approach convergence. The reward ordering (GAE ≈ `sobolev_h4` >> `sobolev_h7` >> `quadrature_h7`) is likely stable since it is driven by the variance explosion in `quadrature_h7`, but the absolute reward gaps would narrow with longer runs and per-method hyperparameter tuning.

The Sobolev improvement percentages in the mathematical results measure smoothness of the weight sequence, not variance reduction in practice. The benchmark confirms that smoothness translates to practical benefit, but the quantitative relationship between $J(\mathbf{w})$ improvement and advantage variance reduction is not yet characterized.

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
@misc{margolis2026qwae,
  author = {Margolis, Daniel},
  title  = {Quadrature-Weighted Advantage Estimation},
  year   = {2026},
  url    = {https://github.com/danmm16/quadrature-advantage}
}
```