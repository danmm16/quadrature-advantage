import jax.numpy as jnp
import numpy as np


# ── Utilities (duplicated from sobolev_h4.py -- factor into utils.py if project grows) ──

def simpson_weights_for_length(m: int) -> np.ndarray:
    """
    Composite Simpson quadrature weights for m equally-spaced points.
    Composite 1/3 rule for odd m, 3/8 rule for m=4,
    trapezoidal fallback for m=2, composite (1/3 then 3/8) for even m >= 6.
    """
    if m == 2:
        return np.array([1, 1], dtype=np.float64) / 2.0
    elif m == 3:
        return np.array([1, 4, 1], dtype=np.float64) / 3.0
    elif m == 4:
        return np.array([3, 9, 9, 3], dtype=np.float64) / 8.0
    elif m == 5:
        return np.array([1, 4, 2, 4, 1], dtype=np.float64) / 3.0
    elif m % 2 == 1:                   # odd >= 7: composite 1/3
        w = np.ones(m, dtype=np.float64)
        w[1:-1:2] = 4
        w[2:-2:2] = 2
        return w / 3.0
    else:                              # even >= 6: composite 1/3 + 3/8 at end
        w = np.zeros(m, dtype=np.float64)
        n1 = m - 3
        if n1 >= 3 and n1 % 2 == 1:
            w1 = np.ones(n1, dtype=np.float64)
            w1[1:-1:2] = 4
            w1[2:-2:2] = 2
            w[:n1] += w1 / 3.0
        w[m-4:] += np.array([3, 9, 9, 3], dtype=np.float64) / 8.0
        return w


def compute_moments(alpha: float, order: int, N: float = None) -> np.ndarray:
    """
    M_j(alpha) = integral_0^N t^j exp(-alpha t) dt, j = 0..order-1.
    Uses Taylor series for |alpha*N| < 1 (avoids catastrophic cancellation)
    and the integration-by-parts recurrence otherwise.
    """
    if N is None:
        N = float(order - 1)
    moments = np.zeros(order)

    if abs(alpha * N) < 1.0:
        for j in range(order):
            val   = 0.0
            pow_N = N ** (j + 1)
            coeff = 1.0
            for m in range(60):
                contrib = coeff * pow_N / (j + m + 1)
                val    += contrib
                if m > 0 and abs(contrib) < abs(val) * 1e-15:
                    break
                coeff *= -alpha / (m + 1)
                pow_N *= N
            moments[j] = val
    else:
        exp_aN     = np.exp(-alpha * N)
        moments[0] = (1.0 - exp_aN) / alpha
        for j in range(1, order):
            moments[j] = (j / alpha) * moments[j-1] - (N**j / alpha) * exp_aN

    return moments


def difference_matrix(n: int, order: int) -> np.ndarray:
    """Finite difference matrix of given order. Shape: (n - order, n)."""
    D = np.eye(n)
    for _ in range(order):
        D = np.diff(D, axis=0)
    return D


# ── Main ───────────────────────────────────────────────────────────────────────

def simpson_sobolev_h7_weights(
    gamma: float,
    lam:   float,
    H:     int   = 8,
    mu0:   float = 1.0,
    mu1:   float = 1.0,
    mu2:   float = 1.0,
) -> jnp.ndarray:
    """
    O(h^7) advantage quadrature weights with Simpson-Sobolev regularization.

    Imposes 7 moment conditions (exactness up to degree 6), leaving H - 7 free
    parameters determined by minimizing the Simpson-approximated H^2 Sobolev
    norm of the weight sequence:

        J(w) = mu0 * w^T S w
             + mu1 * (L1 w)^T S1 (L1 w)
             + mu2 * (L2 w)^T S2 (L2 w)

    Minimum stencil: H=8 (one free parameter -- Sobolev effect is small).
    Recommended:     H=11 (four free parameters -- meaningful smoothing).

    Args:
        gamma, lam:     RL discount parameters; alpha = -log(gamma * lam)
        H:              stencil size; H >= 8
        mu0, mu1, mu2:  penalties on w, Delta^1 w, Delta^2 w

    Returns:
        weights: shape (H,) JAX array
    """
    assert H >= 8, (
        f"H={H} too small: need H >= 8 for at least one free parameter "
        f"after imposing 7 moment conditions"
    )

    alpha = float(-np.log(np.clip(gamma * lam, 1e-12, 1.0)))
    N     = float(H - 1)

    # 7 moment conditions: exactness up to degree 6
    # Vandermonde on normalized nodes [0, 1] avoids ill-conditioning
    k       = np.arange(H, dtype=np.float64)
    N_sc    = float(H - 1)
    k_norm  = k / N_sc
    V7      = np.vander(k_norm, N=7, increasing=True).T        # (7, H)
    M7      = compute_moments(alpha, order=7, N=N)              # (7,)
    M7_norm = M7 / (N_sc ** np.arange(7, dtype=np.float64))    # rescale RHS

    # Simpson-Sobolev roughness matrix
    S  = np.diag(simpson_weights_for_length(H))
    L1 = difference_matrix(H, order=1)
    L2 = difference_matrix(H, order=2)
    S1 = np.diag(simpson_weights_for_length(H - 1))
    S2 = np.diag(simpson_weights_for_length(H - 2))

    R = (mu0 * S
       + mu1 * (L1.T @ S1 @ L1)
       + mu2 * (L2.T @ S2 @ L2))                               # (H, H) SPD

    # Lagrange multiplier solution: w* = R^{-1} V7^T (V7 R^{-1} V7^T)^{-1} M7_norm
    R_inv = np.linalg.inv(R)
    A     = V7 @ R_inv @ V7.T                                   # (7, 7)
    lam_v = np.linalg.solve(A, M7_norm)                        # (7,)
    w     = R_inv @ V7.T @ lam_v                               # (H,)

    return jnp.array(w, dtype=jnp.float32)