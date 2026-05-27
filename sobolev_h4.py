import jax
import jax.numpy as jnp
import numpy as np

# ── Building blocks ────────────────────────────────────────────────────────────

def simpson_weights_for_length(m: int) -> np.ndarray:
    """
    Composite Simpson quadrature weights for m equally-spaced points.
    Composite 1/3 rule for odd m, 3/8 rule for m=4,
    composite (1/3 then 3/8) for even m >= 6.
    """
    if m == 2:
        return np.array([1, 1], dtype=np.float64) / 2.0   # trapezoidal fallback
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
        n1 = m - 3                     # 1/3 portion covers indices 0..n1
        if n1 >= 3 and n1 % 2 == 1:
            w1 = np.ones(n1, dtype=np.float64)
            w1[1:-1:2] = 4
            w1[2:-2:2] = 2
            w[:n1] += w1 / 3.0
        w[m-4:] += np.array([3, 9, 9, 3], dtype=np.float64) / 8.0
        return w


# NOTE: compute_moments is duplicated in simpson_h4.py.
# If extending this project, consider factoring into a shared utils.py.
def compute_moments(alpha: float, order: int, N: float = None) -> np.ndarray:
    if N is None:
        N = float(order - 1)
    moments = np.zeros(order)

    if abs(alpha * N) < 1.0:
        # Taylor series: M_j = sum_{m>=0} (-alpha)^m / m! * N^{j+m+1} / (j+m+1)
        # Stable for small alpha -- no catastrophic cancellation.
        # Converges in ~10 terms for alpha*N < 1.
        for j in range(order):
            val   = 0.0
            pow_N = N ** (j + 1)   # N^{j+m+1}, starting at m=0
            coeff = 1.0            # (-alpha)^m / m!
            for m in range(60):
                contrib = coeff * pow_N / (j + m + 1)
                val    += contrib
                if m > 0 and abs(contrib) < abs(val) * 1e-15:
                    break
                coeff *= -alpha / (m + 1)
                pow_N *= N
            moments[j] = val
    else:
        # Recurrence is stable when alpha*N >= 1 (exp_aN not close to 1)
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

def simpson_sobolev_weights(
    gamma: float,
    lam:   float,
    H:     int   = 5,
    mu0:   float = 1.0,
    mu1:   float = 1.0,
    mu2:   float = 1.0,
) -> jnp.ndarray:
    """
    O(h^4) advantage quadrature weights with Simpson-Sobolev regularization.

    Minimizes the Simpson-approximated H^2 Sobolev norm of the weight sequence:

        J(w) = mu0 * w^T S w
             + mu1 * (D1 w)^T S1 (D1 w)
             + mu2 * (D2 w)^T S2 (D2 w)

    subject to polynomial exactness up to degree 3 (four moment conditions).

    Args:
        gamma, lam:     RL discount parameters; alpha = -log(gamma * lam)
        H:              stencil size; H=5 is minimal, H=7 gives stronger smoothing
        mu0, mu1, mu2:  penalties on w, Delta^1 w, Delta^2 w

    Returns:
        weights: shape (H,) JAX array
    """
    assert H >= 5, "H >= 5 required: need H - 4 >= 1 free parameters for Sobolev opt"

    alpha = float(-np.log(np.clip(gamma * lam, 1e-12, 1.0)))
    N     = float(H - 1)

    # Polynomial exactness constraints: V4 w = M4
    k      = np.arange(H, dtype=np.float64)
    N_sc   = float(H - 1)
    k_norm = k / N_sc
    V4     = np.vander(k_norm, N=4, increasing=True).T
    M4     = compute_moments(alpha, order=4, N=N)
    M4_norm = M4 / (N_sc ** np.arange(4, dtype=np.float64))

    # Simpson-Sobolev roughness matrix
    S  = np.diag(simpson_weights_for_length(H))         # (H,   H  )
    L1 = difference_matrix(H, order=1)                  # (H-1, H  )
    L2 = difference_matrix(H, order=2)                  # (H-2, H  )
    S1 = np.diag(simpson_weights_for_length(H - 1))     # (H-1, H-1)
    S2 = np.diag(simpson_weights_for_length(H - 2))     # (H-2, H-2)

    R = (mu0 * S
       + mu1 * (L1.T @ S1 @ L1)
       + mu2 * (L2.T @ S2 @ L2))                        # (H, H) SPD

    # Lagrange multiplier solution: w* = R^{-1} V4^T (V4 R^{-1} V4^T)^{-1} M4
    R_inv = np.linalg.inv(R)
    A     = V4 @ R_inv @ V4.T                           # (4, 4)
    lam_v = np.linalg.solve(A, M4_norm)                     # (4,)
    w     = R_inv @ V4.T @ lam_v                        # (H,)

    return jnp.array(w, dtype=jnp.float32)