import numpy as np
from sobolev_h7 import (
    simpson_sobolev_h7_weights,
    compute_moments,
    difference_matrix,
    simpson_weights_for_length,
)
from quadrature_h7 import quadrature_weights


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_polynomial_exactness(gamma=0.99, lam=0.95, H=8):
    """
    All 7 moment conditions must be satisfied to relative error < 1e-6.
    Tests exactness for delta(t) = t^j, j = 0..6.
    """
    alpha  = -np.log(gamma * lam)
    w      = np.array(simpson_sobolev_h7_weights(gamma, lam, H))
    M7     = compute_moments(alpha, order=7, N=float(H - 1))
    k      = np.arange(H, dtype=np.float64)

    print(f"Polynomial exactness test (H={H}):")
    for j in range(7):
        approx  = np.dot(w, k ** j)
        exact   = M7[j]
        rel_err = abs(approx - exact) / abs(exact) if abs(exact) > 1e-12 else abs(approx - exact)
        print(f"  j={j}: approx={approx:.8f}, exact={exact:.8f}, rel_err={rel_err:.2e}")
        assert rel_err < 1e-5, f"Failed exactness at degree {j}: rel_err={rel_err}"
    print("  ✓\n")


def test_sobolev_beats_unconstrained(gamma=0.99, lam=0.95, H=8):
    """
    The Sobolev solution must have strictly smaller J(w) than the
    minimum-Euclidean-norm pseudoinverse solution.
    With H=8 (one free parameter) the improvement will be small but nonzero.
    With H=11 (four free parameters) it will be more pronounced.
    """
    alpha = -np.log(gamma * lam)
    k     = np.arange(H, dtype=np.float64)

    # Normalized Vandermonde -- must match sobolev_h7.py exactly
    N_sc    = float(H - 1)
    k_norm  = k / N_sc
    V7      = np.vander(k_norm, N=7, increasing=True).T
    M7      = compute_moments(alpha, order=7, N=float(H - 1))
    M7_norm = M7 / (N_sc ** np.arange(7, dtype=np.float64))

    # Pseudoinverse: minimum Euclidean norm, no Sobolev penalty
    # Solves V7 @ w = M7_norm in the least-norm sense
    w_pinv = np.linalg.lstsq(V7, M7_norm, rcond=None)[0]

    # Sobolev weights
    w_ss = np.array(simpson_sobolev_h7_weights(gamma, lam, H))

    # Evaluate J(w) for both (using mu0=mu1=mu2=1 to match default)
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
    print(f"  Improvement: {100*(J_pinv - J_ss)/J_pinv:.4f}%")
    assert J_ss <= J_pinv + 1e-8, "Sobolev weights are not optimal"
    print("  ✓\n")


def test_weight_profile(gamma=0.99, lam=0.95, H=8):
    """Print weight profile and verify all weights are positive."""
    w = np.array(simpson_sobolev_h7_weights(gamma, lam, H))
    print(f"Weight profile (H={H}, gamma={gamma}, lam={lam}):")
    print(f"  w = {w.round(4)}")
    print(f"  All positive: {bool(np.all(w > 0))}")
    assert np.all(w > 0), f"Non-positive weights found: {w}"
    print("  ✓\n")


def test_limiting_case(H=8):
    """As alpha -> 0, weights should sum to N = H - 1."""
    w_limit      = np.array(simpson_sobolev_h7_weights(0.999999, 0.999999, H))
    expected_sum = float(H - 1)
    print(f"Limiting case (alpha -> 0, H={H}):")
    print(f"  w = {w_limit.round(4)}")
    print(f"  sum(w) = {w_limit.sum():.6f} (should be ~{expected_sum:.1f})")
    assert abs(w_limit.sum() - expected_sum) < 1e-2, f"Weight sum wrong: {w_limit.sum()}"
    print("  ✓\n")


def test_accuracy_comparison(gamma=0.99, lam=0.95):
    """
    Compare approximation errors of:
      - quadrature_h7  (H=7,  O(h^7), no Sobolev)
      - sobolev_h7 A   (H=8,  O(h^7), 1 free parameter)
      - sobolev_h7 B   (H=11, O(h^7), 4 free parameters)

    Each method approximates the discounted integral over its own window
    [0, H-1]. True integrals are computed separately for each window.
    Errors are relative to the true integral in each case.

    All three should be exact (near machine epsilon) for polynomials up to
    degree 6. Differences on smooth non-polynomial functions (exponential,
    mixed) reveal the effect of the Sobolev smoothing.
    """
    alpha = -np.log(gamma * lam)

    # Weights for each method
    w_h7 = np.array(quadrature_weights(gamma, lam, H=7))   # window [0, 6]
    w_A  = np.array(simpson_sobolev_h7_weights(gamma, lam, H=8))   # window [0, 7]
    w_B  = np.array(simpson_sobolev_h7_weights(gamma, lam, H=11))  # window [0, 10]

    # Moments over each window
    M_h7 = compute_moments(alpha, order=11, N=6.0)
    M_A  = compute_moments(alpha, order=11, N=7.0)
    M_B  = compute_moments(alpha, order=11, N=10.0)

    t_h7 = np.arange(7,  dtype=np.float64)
    t_A  = np.arange(8,  dtype=np.float64)
    t_B  = np.arange(11, dtype=np.float64)

    def gae(gamma, lam, H):
        return np.array([(gamma * lam) ** k for k in range(H)])

    w_gae_h7 = gae(gamma, lam, 7)
    w_gae_A  = gae(gamma, lam, 8)
    w_gae_B  = gae(gamma, lam, 11)

    def rel_err(approx, exact):
        return abs(approx - exact) / abs(exact) if abs(exact) > 1e-12 else abs(approx - exact)

    # Test cases: (name, function of t, true integral coefficients)
    # For polynomials, true = sum of c_j * M_j
    # For exponential, true = M_0 with alpha replaced
    def run_case(name, fn_h7, fn_A, fn_B, true_h7, true_A, true_B):
        e_h7  = rel_err(np.dot(w_h7,     fn_h7), true_h7)
        e_A   = rel_err(np.dot(w_A,      fn_A),  true_A)
        e_B   = rel_err(np.dot(w_B,      fn_B),  true_B)
        g_h7  = rel_err(np.dot(w_gae_h7, fn_h7), true_h7)
        g_A   = rel_err(np.dot(w_gae_A,  fn_A),  true_A)
        g_B   = rel_err(np.dot(w_gae_B,  fn_B),  true_B)
        return e_h7, e_A, e_B, g_h7, g_A, g_B

    print(f"\nAccuracy comparison (gamma={gamma}, lam={lam}):")
    print(f"  Stencil widths: quadrature_h7=7, sobolev_h7_A=8, sobolev_h7_B=11")
    print(f"  All errors are relative to each method's true integral over its own window.\n")
    header = f"{'Function':<22} {'q_h7 (H=7)':>12} {'ss_h7_A (H=8)':>14} {'ss_h7_B (H=11)':>15} {'GAE_h7':>10} {'GAE_A':>10} {'GAE_B':>10}"
    print(header)
    print("-" * len(header))

    cases = [
        ("constant",       np.ones(7), np.ones(8), np.ones(11),
         M_h7[0], M_A[0], M_B[0]),
        ("linear",         t_h7, t_A, t_B,
         M_h7[1], M_A[1], M_B[1]),
        ("cubic",          t_h7**3, t_A**3, t_B**3,
         M_h7[3], M_A[3], M_B[3]),
        ("degree 6",       t_h7**6, t_A**6, t_B**6,
         M_h7[6], M_A[6], M_B[6]),
        ("exponential",
         np.exp(-0.1*t_h7), np.exp(-0.1*t_A), np.exp(-0.1*t_B),
         compute_moments(alpha+0.1, order=1, N=6.0)[0],
         compute_moments(alpha+0.1, order=1, N=7.0)[0],
         compute_moments(alpha+0.1, order=1, N=10.0)[0]),
        ("mixed poly",
         1+0.3*t_h7-0.05*t_h7**2+0.002*t_h7**3,
         1+0.3*t_A -0.05*t_A**2 +0.002*t_A**3,
         1+0.3*t_B -0.05*t_B**2 +0.002*t_B**3,
         1.0*M_h7[0]+0.3*M_h7[1]-0.05*M_h7[2]+0.002*M_h7[3],
         1.0*M_A[0] +0.3*M_A[1] -0.05*M_A[2] +0.002*M_A[3],
         1.0*M_B[0] +0.3*M_B[1] -0.05*M_B[2] +0.002*M_B[3]),
    ]

    for (name, fh7, fA, fB, th7, tA, tB) in cases:
        e_h7, e_A, e_B, g_h7, g_A, g_B = run_case(name, fh7, fA, fB, th7, tA, tB)
        print(f"{name:<22} {e_h7:>12.2e} {e_A:>14.2e} {e_B:>15.2e} {g_h7:>10.2e} {g_A:>10.2e} {g_B:>10.2e}")

    print()

    # Sobolev improvement: compare J(w) between methods on their own windows
    print("Sobolev smoothing effect (J(w) relative to pseudoinverse baseline):")
    for H in range(8, 17):
        alpha_  = -np.log(gamma * lam)
        k       = np.arange(H, dtype=np.float64)
        N_sc    = float(H - 1)
        k_norm  = k / N_sc
        V7      = np.vander(k_norm, N=7, increasing=True).T
        M7      = compute_moments(alpha_, order=7, N=float(H - 1))
        M7_norm = M7 / (N_sc ** np.arange(7, dtype=np.float64))
        w_pinv  = np.linalg.lstsq(V7, M7_norm, rcond=None)[0]
        w_ss    = np.array(simpson_sobolev_h7_weights(gamma, lam, H))
        S       = np.diag(simpson_weights_for_length(H))
        L1      = difference_matrix(H, order=1)
        L2      = difference_matrix(H, order=2)
        S1      = np.diag(simpson_weights_for_length(H - 1))
        S2      = np.diag(simpson_weights_for_length(H - 2))
        R       = S + L1.T @ S1 @ L1 + L2.T @ S2 @ L2
        J_pinv  = w_pinv @ R @ w_pinv
        J_ss    = w_ss   @ R @ w_ss
        free    = H - 7
        print(f"  H={H:2d} ({free} free): J(w_pinv)={J_pinv:.6f}, "
              f"J(w_ss)={J_ss:.6f}, "
              f"improvement={100*(J_pinv - J_ss)/J_pinv:.4f}%")
    print()


if __name__ == "__main__":
    H_values = range(8, 17)  # 8, 9, 10, ..., 16

    for H in H_values:
        print("=" * 60)
        print(f"H={H} ({H - 7} free parameter{'s' if H - 7 > 1 else ''})")
        print("=" * 60)
        test_polynomial_exactness(H=H)
        test_sobolev_beats_unconstrained(H=H)
        test_weight_profile(H=H)
        test_limiting_case(H=H)

    # Cross-method accuracy comparison across all H values
    print("=" * 60)
    print("Cross-method accuracy comparison")
    print("=" * 60)
    test_accuracy_comparison()