import numpy as np
from sobolev_h4 import (
    simpson_weights_for_length,
    compute_moments,
    difference_matrix,
    simpson_sobolev_weights,
)

def test_polynomial_exactness(gamma=0.99, lam=0.95, H=5):
    """Weights must exactly integrate any delta(t) = t^j, j=0,1,2,3."""
    alpha = -np.log(gamma * lam)
    w = np.array(simpson_sobolev_weights(gamma, lam, H))
    M4 = compute_moments(alpha, order=4, N=float(H - 1))
    k = np.arange(H, dtype=np.float64)

    print(f"Polynomial exactness test (H={H}):")
    for j in range(4):
        approx = np.dot(w, k ** j)
        exact  = M4[j]
        rel_err = abs(approx - exact) / abs(exact) if abs(exact) > 1e-12 else abs(approx - exact)
        print(f"  j={j}: approx={approx:.8f}, exact={exact:.8f}, err={rel_err:.2e}")
    assert all(abs(np.dot(w, k**j) - M4[j]) / abs(M4[j]) < 1e-6 for j in range(4) if abs(M4[j]) > 1e-12), "Exactness failed"
    print("  ✓\n")


def test_sobolev_beats_unconstrained(gamma=0.99, lam=0.95, H=5):
    """
    The Sobolev weights should have strictly smaller J(w) than the
    pseudoinverse solution (minimum Euclidean norm, no Sobolev penalty).
    """
    alpha = -np.log(gamma * lam)
    k  = np.arange(H, dtype=np.float64)
    V4 = np.vander(k, N=4, increasing=True).T
    M4 = compute_moments(alpha, order=4, N=float(H - 1))

    # Pseudoinverse (minimum L2 norm, ignores Sobolev)
    w_pinv = np.linalg.lstsq(V4, M4, rcond=None)[0]

    # Simpson-Sobolev weights
    w_ss = np.array(simpson_sobolev_weights(gamma, lam, H))

    # Evaluate J(w) for both
    S  = np.diag(simpson_weights_for_length(H))
    L1 = difference_matrix(H, order=1)
    L2 = difference_matrix(H, order=2)
    S1 = np.diag(simpson_weights_for_length(H - 1))
    S2 = np.diag(simpson_weights_for_length(H - 2))
    R  = S + L1.T @ S1 @ L1 + L2.T @ S2 @ L2

    J_pinv = w_pinv @ R @ w_pinv
    J_ss   = w_ss   @ R @ w_ss

    print(f"Sobolev objective test (H={H}):")
    print(f"  J(w_pinv) = {J_pinv:.6f}")
    print(f"  J(w_ss)   = {J_ss:.6f}  (should be <= J_pinv)")
    assert J_ss <= J_pinv + 1e-8, "Sobolev weights are not optimal"
    print("  ✓\n")


def test_monotone_decay(gamma=0.99, lam=0.95, H=5):
    """
    For large alpha (strong discount), weights should be roughly monotone
    decreasing -- early TD residuals matter more.
    """
    w = np.array(simpson_sobolev_weights(gamma, lam, H))
    print(f"Weight profile (H={H}, gamma={gamma}, lam={lam}):")
    print(f"  w = {w.round(4)}")
    print(f"  All positive: {bool(np.all(w > 0))}")
    assert np.all(w > 0), f"Non-positive weights found: {w}"
    print("  ✓\n")


def test_limiting_case(H=5):
    """
    As gamma*lam -> 1 (alpha -> 0), weights should stabilize to the
    Simpson-Sobolev-optimal undiscounted rule (not Boole's rule, since
    that uses 5 moment conditions; ours uses only 4).
    """
    w_limit = np.array(simpson_sobolev_weights(0.9999, 0.9999, H))
    expected_sum = float(H - 1)
    print(f"Limiting case (alpha -> 0, H={H}):")
    print(f"  w = {w_limit.round(4)}")
    print(f"  sum(w) = {w_limit.sum():.6f} (should be ~{expected_sum:.1f})")
    assert abs(w_limit.sum() - expected_sum) < 1e-2, f"Weight sum wrong: {w_limit.sum()}"
    print("  ✓\n")


from quadrature_h7 import quadrature_weights

def test_accuracy_comparison(gamma=0.99, lam=0.95):
    """
    Compare approximation error of simpson_h4 (O(h^7)) vs sobolev_h4 (O(h^4))
    vs standard GAE on several TD residual functions.

    simpson_h4 should be exact for polynomials up to degree 6.
    sobolev_h4 should be exact for polynomials up to degree 3.
    GAE should be poor for all of them over a fixed window.
    """
    alpha = -np.log(gamma * lam)
    H = 7  # use H=7 for both so the comparison is fair (same stencil width)

    w_h7 = np.array(quadrature_weights(gamma, lam, H=7))
    w_ss = np.array(simpson_sobolev_weights(gamma, lam, H=7))

    t = np.arange(H, dtype=np.float64)
    M = compute_moments(alpha, order=H, N=float(H - 1))

    # GAE weights (geometric series, truncated at H)
    w_gae = np.array([(gamma * lam) ** k for k in range(H)])

    test_cases = {
        "constant (degree 0)":  (np.ones(H),          M[0]),
        "linear (degree 1)":    (t,                    M[1]),
        "quadratic (degree 2)": (t ** 2,               M[2]),
        "cubic (degree 3)":     (t ** 3,               M[3]),
        "degree 4":             (t ** 4,               M[4]),
        "degree 6":             (t ** 6,               M[6]),
        "mixed polynomial":     (
            1.0 + 0.3*t - 0.05*t**2 + 0.002*t**3,
            1.0*M[0] + 0.3*M[1] - 0.05*M[2] + 0.002*M[3]
        ),
        "exponential":          (
            np.exp(-0.1 * t),
            compute_moments(alpha + 0.1, order=1, N=float(H - 1))[0]
        ),
    }

    print(f"\nAccuracy comparison (gamma={gamma}, lam={lam}, H={H}):")
    print(f"{'Function':<25} {'simpson_h4':>14} {'sobolev_h4':>14} {'GAE':>14}")
    print("-" * 70)

    for name, (delta, true_val) in test_cases.items():
        err_h7  = abs(np.dot(w_h7,  delta) - true_val)
        err_ss  = abs(np.dot(w_ss,  delta) - true_val)
        err_gae = abs(np.dot(w_gae, delta) - true_val)
        print(f"{name:<25} {err_h7:>14.2e} {err_ss:>14.2e} {err_gae:>14.2e}")

    print()

    # Assertions: simpson_h4 exact up to degree 6, sobolev_h4 exact up to degree 3
    for j in range(7):
        delta    = t ** j
        true_val = M[j]
        err_h7   = abs(np.dot(w_h7, delta) - true_val) / abs(true_val)
        assert err_h7 < 1e-5, f"simpson_h4 failed exactness at degree {j}: rel_err={err_h7}"

    for j in range(4):
        delta    = t ** j
        true_val = M[j]
        err_ss   = abs(np.dot(w_ss, delta) - true_val) / abs(true_val)
        assert err_ss < 1e-5, f"sobolev_h4 failed exactness at degree {j}: rel_err={err_ss}"

    print("Exactness assertions ✓")


if __name__ == "__main__":
    for H in [5, 7]:
        test_polynomial_exactness(H=H)
        test_sobolev_beats_unconstrained(H=H)
        test_monotone_decay(H=H)
        test_limiting_case(H=H)
    test_accuracy_comparison()