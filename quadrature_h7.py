import jax
import jax.numpy as jnp
import numpy as np

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


def _lagrange_poly_coeffs(k: int, H: int) -> np.ndarray:
    """
    Coefficients of P_k(t) = prod_{j=0, j!=k}^{H-1} (t - j).
    Degree H-1 polynomial. Returns H coefficients [c_0, c_1, ..., c_{H-1}].
    Computed exactly in integer arithmetic -- no floating-point error.
    """
    coeffs = np.array([1.0])
    for j in range(H):
        if j == k:
            continue
        new_coeffs = np.zeros(len(coeffs) + 1)
        new_coeffs[1:]  += coeffs        # t * poly
        new_coeffs[:-1] -= j * coeffs   # -j * poly
        coeffs = new_coeffs
    return coeffs  # length H


def quadrature_weights(gamma: float, lam: float, H: int = 7) -> jnp.ndarray:
    """
    O(h^H) exponentially-fitted quadrature weights for H nodes.

    Computed via direct integration of Lagrange basis functions:
        w_k = integral_0^{H-1} L_k(t) exp(-alpha t) dt

    This bypasses the ill-conditioned Vandermonde system entirely.
    """
    alpha = -np.log(np.clip(gamma * lam, 1e-12, 1.0))
    N     = float(H - 1)

    # Moments M_j = integral_0^N t^j exp(-alpha t) dt, j = 0..H-1
    M = compute_moments(alpha, H, N=N)

    w = np.zeros(H, dtype=np.float64)
    for k in range(H):
        # Polynomial coefficients of P_k(t) = prod_{j!=k}(t-j)
        poly = _lagrange_poly_coeffs(k, H)          # exact, length H

        # Integral of P_k(t) exp(-alpha t) over [0, N]
        integral = np.dot(poly, M)                  # dot product with moments

        # Denominator: prod_{j!=k}(k-j) -- exact integer product
        denom = np.prod([k - j for j in range(H) if j != k], dtype=np.float64)

        w[k] = integral / denom

    return jnp.array(w, dtype=jnp.float32)


def quadrature_advantage(deltas: jnp.ndarray,
                         gamma: float,
                         lam: float) -> jnp.ndarray:
    """
    Compute advantage estimates via the 7-point O(h^7) quadrature rule.

    Args:
        deltas: TD residuals, shape (T,) where T >= 7
        gamma, lam: discount and GAE lambda parameters
    Returns:
        advantages: shape (T - 6,)
    """
    H = 7
    assert len(deltas) >= H, f"deltas length {len(deltas)} must be >= H={H}"
    w = quadrature_weights(gamma, lam, H)  # precompute once

    def single_window(delta_window):
        return jnp.dot(w, delta_window)

    # Slide a window of size H across the trajectory
    # deltas[t:t+H] for t in range(T - H + 1)
    windows = jax.vmap(
        lambda t: jax.lax.dynamic_slice(deltas, (t,), (H,))
    )(jnp.arange(len(deltas) - H + 1))

    return jax.vmap(single_window)(windows)


# ── Tests ────────────────────────────────────────────────────────────────────

def test_limiting_case():
    """Weights should approach Newton-Cotes as gamma*lam -> 1."""
    nc_weights = jnp.array([41, 216, 27, 272, 27, 216, 41]) / 140.0
    w = quadrature_weights(gamma=0.999999, lam=0.999999, H=7)
    assert jnp.allclose(w, nc_weights, atol=1e-3), f"Limiting case failed: {w}"
    print("Limiting case ✓")

def test_polynomial_exactness(gamma=0.99, lam=0.95, H=7):
    """
    Rule should be exact for delta(t) = t^j, j = 0,...,6.
    True value: integral_0^6 t^j exp(-alpha t) dt = M_j(alpha).
    """
    alpha = -np.log(gamma * lam)
    w = np.array(quadrature_weights(gamma, lam, H=H))
    M = compute_moments(alpha, order=H, N=float(H - 1))

    for j in range(H):
        nodes = np.arange(H, dtype=np.float64)
        approx = np.dot(w, nodes ** j)
        exact = M[j]
        err = abs(approx - exact)
        print(f"j={j}: approx={approx:.8f}, exact={exact:.8f}, err={err:.2e}")
        rel_err = err / abs(exact) if abs(exact) > 1e-12 else err
        print(f"j={j}: approx={approx:.8f}, exact={exact:.8f}, abs_err={err:.2e}, rel_err={rel_err:.2e}")
        assert rel_err < 1e-6, f"Failed exactness for degree {j}: rel_error={rel_err}"

    print("Polynomial exactness ✓")

def test_vs_gae():
    """
    Compare quadrature estimate vs standard GAE on a smooth delta function.
    Both should approximate the true discounted integral; quadrature
    should be closer for polynomial-like TD residuals.
    """
    gamma, lam = 0.99, 0.95
    alpha = -np.log(gamma * lam)

    # True delta: cubic polynomial (quadrature should be EXACT)
    t = np.arange(7, dtype=np.float64)
    delta_cubic = 1.0 + 0.3*t - 0.05*t**2 + 0.002*t**3

    # True integral
    M = compute_moments(alpha, order=7)
    # coefficients of the cubic: 1, 0.3, -0.05, 0.002
    true_A = 1.0*M[0] + 0.3*M[1] - 0.05*M[2] + 0.002*M[3]

    # Quadrature estimate
    w = np.array(quadrature_weights(gamma, lam, H=7))
    quad_A = np.dot(w, delta_cubic)

    # GAE estimate (truncated geometric sum)
    r = gamma * lam
    gae_A = sum(r**k * delta_cubic[k] for k in range(7))

    print(f"True integral: {true_A:.8f}")
    print(f"Quadrature:    {quad_A:.8f}  (err={abs(quad_A-true_A):.2e})")
    print(f"GAE (trunc):   {gae_A:.8f}  (err={abs(gae_A-true_A):.2e})")


if __name__ == "__main__":
    test_limiting_case()
    test_polynomial_exactness()
    test_vs_gae()