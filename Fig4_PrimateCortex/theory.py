import numpy as np
from scipy.linalg import solve_continuous_lyapunov

def auto_theory_from_OU(H, q00, idx_list=None):
    """
    Auto-covariance theory for OU: dx = H x dt + dβ, with E[dβ dβ^T] = Q dt,
    noise only on node 0: Q[0,0] = q00.

    Returns:
        tau_i  : "area-weighted" time constant (ratio of moments) for each i
        peak_i : C_ii(0) = P_inf[ii]
        amp_i  : area over t>=0: ∫_0^∞ C_ii(t) dt = [-H^{-1} P_inf]_{ii}
        P_inf  : stationary covariance
    """
    H = np.asarray(H, dtype=float)
    d = H.shape[0]
    assert H.shape == (d, d)
    Q = np.zeros((d, d), dtype=float)
    Q[0, 0] = float(q00)
    P_inf = solve_continuous_lyapunov(H, -Q)
    if idx_list is None:
        idx_list = np.arange(d)
    idx_list = np.asarray(idx_list, dtype=int)
    A = np.linalg.solve(-H, P_inf)
    U = np.linalg.solve(H, P_inf)
    B = np.linalg.solve(H, U)
    peak = np.real(np.diag(P_inf))[idx_list]
    amp = np.real(np.diag(A))[idx_list]
    tau = amp / peak
    return (tau, peak, amp, P_inf)

def cross_cov_theory_two_methods(H, m=0, q00=1.0, use_complex_ok=False):
    """
    Cross-covariance metrics for noise injected only at node m (default 0).

    Returns:
        out = {
          "P_inf": P_inf,
          "lyap": {"amp": (N,), "t_peak": (N,), "peak": (N,)},
          "eig" : {"amp": (N,), "t_peak": (N,), "peak": (N,)},
        }

    Notes:
    - Lyap method uses M1/M2/M3 matrices via linear solves.
    - Eigen method uses your ORIGINAL nested sums (p,q loops).
    - q00 plays the role of "strength" in your generate_theory() code.
    """
    H = np.asarray(H, dtype=float)
    N = H.shape[0]
    assert H.shape == (N, N)
    Q = np.zeros((N, N), dtype=float)
    Q[m, m] = float(q00)
    P_inf = solve_continuous_lyapunov(H, -Q)
    A1 = np.linalg.solve(H, P_inf)
    A2 = np.linalg.solve(H, A1)
    A3 = np.linalg.solve(H, A2)
    HT = H.T
    B1 = np.linalg.solve(HT, P_inf.T).T
    B2 = np.linalg.solve(HT, B1.T).T
    B3 = np.linalg.solve(HT, B2.T).T
    M1_mat = -(A1 + B1)
    M2_mat = A2 - B2
    M3_mat = -(A3 + B3)
    M1 = M1_mat[m, :]
    M2 = M2_mat[m, :]
    M3 = M3_mat[m, :]
    t_peak_lyap = np.real(M2 / M1)
    denom = 4 * M3 * M1 - 2 * M2 ** 2
    denom = np.where(np.real(denom) < 0, denom + 0j, denom)
    peak_lyap = np.real(M1 ** 2 / np.sqrt(denom))
    amp_lyap = np.real(-M1)
    eigen_val, eigen_matrix = np.linalg.eig(H)
    eigen_matrix_inv = np.linalg.inv(eigen_matrix)
    thr_peak_eig = np.zeros(N, dtype=float)
    thr_t_eig = np.zeros(N, dtype=float)
    thr_amp_eig = np.zeros(N, dtype=float)
    m = 0
    strength = float(q00)
    for node_j in range(N):
        MM_first = 0
        MM_second = 0
        MM_third = 0
        for p in range(N):
            for q in range(N):
                MM_first += eigen_matrix[node_j][q] * eigen_matrix_inv[q][m] * (eigen_matrix[m][p] * eigen_matrix_inv[p][m]) * (1 / (eigen_val[p] * eigen_val[q]))
                MM_second += eigen_matrix[node_j][q] * eigen_matrix_inv[q][m] * (eigen_matrix[m][p] * eigen_matrix_inv[p][m]) * (1 / (eigen_val[p] * eigen_val[q]) * (1 / eigen_val[p] - 1 / eigen_val[q]))
                MM_third += eigen_matrix[node_j][q] * eigen_matrix_inv[q][m] * (eigen_matrix[m][p] * eigen_matrix_inv[p][m]) * ((1 / eigen_val[p]) ** 3 + (1 / eigen_val[q]) ** 3) * (1 / (eigen_val[p] + eigen_val[q]))
        thr_peak_eig[node_j] = np.real(strength * MM_first ** 2 / (4 * MM_third * MM_first - 2 * MM_second ** 2) ** 0.5)
        thr_amp_eig[node_j] = np.real(strength * MM_first)
        thr_t_eig[node_j] = np.real(MM_second / thr_amp_eig[node_j])
    tau_theory_list = np.zeros(N, dtype=float)
    for area_idx in range(N):
        tau_dom = 0
        tau_num = 0
        for j in range(N):
            tau_num += eigen_matrix[area_idx, j] * eigen_matrix_inv[j, m] / np.power(eigen_val[j], 2)
            tau_dom += eigen_matrix[area_idx, j] * eigen_matrix_inv[j, m] / eigen_val[j]
        tau_theory_list[area_idx] = -tau_num / tau_dom
    out = {'P_inf': P_inf, 'lyap': {'amp': amp_lyap, 't_peak': t_peak_lyap, 'peak': peak_lyap}, 'eig': {'amp': thr_amp_eig, 't_peak': thr_t_eig, 'peak': thr_peak_eig}, 'low': {'t_peak': tau_theory_list}}
    if not use_complex_ok:
        for key in ['lyap', 'eig']:
            for f in ['amp', 't_peak', 'peak']:
                out[key][f] = np.real_if_close(out[key][f], tol=1000000.0)
    return out