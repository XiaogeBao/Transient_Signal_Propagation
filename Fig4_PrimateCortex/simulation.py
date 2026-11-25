import numpy as np

from scipy.fft import rfft, irfft, next_fast_len
from scipy.integrate import simpson
from scipy.optimize import curve_fit

def _single_exp(t, tau):
    return np.exp(-t / tau)

def simulate_linear_sde_node0(
    H, T, dt,
    q00,
    burn_in=0.3,
    seed=0,
    x0=None,
    stride=1,
    noise_type="white",   # "white" or "ou"
    tau_c=10.0,           # OU correlation time (same unit as dt)
):
    """
    Simulate linear dynamics with noise injected only at node 0.

    white:
        dx = H x dt + sigma0 dW,   sigma0^2 = q00

    ou (colored forcing):
        dx = H x dt + e0 * eta dt
        d eta = -(1/tau_c) eta dt + sqrt(2*q00/tau_c) dW
      => stationary Var(eta) = q00, Corr time = tau_c
    """
    H = np.asarray(H, dtype=float)
    d = H.shape[0]
    assert H.shape == (d, d)

    n = int(np.floor(T / dt)) + 1
    k0 = int(burn_in*n) if 0 < burn_in < 1 else max(0,int(burn_in/dt))
    if k0 >= n:
        raise ValueError("Burn-in exceeds simulation length")
    samples = np.empty(((n-1-k0)//stride+1, d))
    state = np.zeros(d)
    if x0 is not None:
        state = np.asarray(x0, dtype=float)

    rng = np.random.default_rng(seed)
    sqrt_dt = np.sqrt(dt)

    noise_type = str(noise_type).lower()
    if noise_type not in ("white", "ou"):
        raise ValueError("noise_type must be 'white' or 'ou'.")

    sigma0 = np.sqrt(float(q00))  # white noise std

    eta = 0.0
    if noise_type == "ou":
        if tau_c is None or tau_c <= 0:
            raise ValueError("For OU noise, tau_c must be > 0.")
        tau_c = float(tau_c)
        ou_sigma = np.sqrt(2.0 * float(q00) / tau_c)

    if k0 == 0:
        samples[0] = state
    for k in range(n - 1):
        state = state + (H @ state) * dt

        if noise_type == "white":
            state[0] += sigma0 * rng.standard_normal() * sqrt_dt
        else:
            eta += (-eta / tau_c) * dt + ou_sigma * rng.standard_normal() * sqrt_dt
            state[0] += eta * dt

        j = k+1-k0
        if j >= 0 and j % stride == 0:
            samples[j//stride] = state
    return samples, dt*stride

def lag_cov_auto_cross_fft(
    x,
    dt,
    target_dt=1.0,
    max_lag=10000.0,
    fit_threshold=0.05,
    fit_max_lag=5000.0,
):
    x = np.asarray(x, dtype=float)
    nT, d = x.shape

    # downsample by stride
    step = max(1, int(round(target_dt / dt)))
    x_ds = x[::step]
    dt_ds = dt * step
    N = x_ds.shape[0]
    if N < 8:
        raise ValueError("Downsampled series too short. Increase T or decrease target_dt.")

    x_ds = x_ds - x_ds.mean(axis=0, keepdims=True)

    L = int(round(max_lag / dt_ds))
    L = min(L, N - 1)
    center = N - 1
    idx0 = center - L
    idx1 = center + L + 1
    lag_samp = np.arange(-L, L + 1)
    lag_ms = lag_samp * dt_ds

    t_pos = np.arange(0, L + 1) * dt_ds

    nfft = next_fast_len(2 * N - 1)

    source_fft = rfft(x_ds[:,0], n=nfft)
    auto_pos = np.empty((d,L+1))
    cross_win = np.empty((d,2*L+1))
    for j in range(d):
        fft_j = rfft(x_ds[:,j], n=nfft)
        rev_j = rfft(x_ds[::-1,j], n=nfft)
        auto_pos[j] = irfft(fft_j*rev_j, n=nfft)[center:center+L+1]/N
        cross_win[j] = irfft(source_fft*rev_j, n=nfft)[idx0:idx1]/N

    auto_peak = np.full(d, np.nan)
    auto_area = np.full(d, np.nan)
    auto_tau  = np.full(d, np.nan)

    cross_peak = np.full(d, np.nan)
    cross_area = np.full(d, np.nan)
    cross_t_peak = np.full(d, np.nan)

    for i in range(d):
        y = auto_pos[i]
        auto_peak[i] = y[0]
        auto_area[i] = simpson(y, x=t_pos)

        if y[0] <= 0:
            continue
        yn = y / y[0]

        if fit_max_lag is not None:
            keep = t_pos <= fit_max_lag
            tt = t_pos[keep]
            yy = yn[keep]
        else:
            tt, yy = t_pos, yn

        below = np.where(yy <= fit_threshold)[0]
        end = below[0] if below.size > 0 else len(yy)
        tt2 = tt[:end]
        yy2 = yy[:end]
        if len(tt2) < 3:
            continue

        try:
            p0 = max(np.median(tt2), dt_ds)
            (tau_hat,), _ = curve_fit(_single_exp, tt2, yy2, p0=[p0], bounds=(1e-12, np.inf))
            auto_tau[i] = float(tau_hat)
        except Exception:
            pass

    for j in range(d):
        y = cross_win[j]
        cross_area[j] = simpson(y, x=lag_ms)
        k = int(np.argmax(y))
        cross_peak[j] = y[k]
        cross_t_peak[j] = lag_ms[k]

    metrics = dict(
        dt_ds=dt_ds,
        step=step,
        N_ds=N,
        max_lag=max_lag,
        auto_peak=auto_peak,
        auto_area=auto_area,
        auto_tau=auto_tau,
        cross_peak=cross_peak,
        cross_area=cross_area,
        cross_t_peak=cross_t_peak,
    )

    return lag_ms, auto_pos, cross_win, metrics

def run_mc(
    H,
    n_runs=1,
    T=200000.0,
    dt=0.05,
    q00=1.0,
    burn_in=0.3,
    target_dt=1.0,
    max_lag=10000.0,
    fit_threshold=0.05,
    fit_max_lag=5000.0,
    seed0=0,
    noise_type="white",
    tau_c=10.0,
):
    H = np.asarray(H, dtype=float)
    d = H.shape[0]

    lag_ms_ref = None
    auto_sum = None
    cross_sum = None

    auto_peak_sum = np.zeros(d)
    auto_area_sum = np.zeros(d)
    auto_tau_sum  = np.zeros(d)

    cross_peak_sum = np.zeros(d)
    cross_area_sum = np.zeros(d)
    cross_t_peak_sum = np.zeros(d)

    valid_auto_tau = np.zeros(d, dtype=int)

    for r in range(n_runs):
        xs, stored_dt = simulate_linear_sde_node0(
            H, T=T, dt=dt, q00=q00, burn_in=burn_in, seed=seed0 + r, stride=max(1,round(target_dt/dt)),
            noise_type=noise_type, tau_c=tau_c
        )

        lag_ms, auto_pos, cross_win, met = lag_cov_auto_cross_fft(
            xs, dt=stored_dt, target_dt=target_dt, max_lag=max_lag,
            fit_threshold=fit_threshold, fit_max_lag=fit_max_lag
        )

        if lag_ms_ref is None:
            lag_ms_ref = lag_ms
            auto_sum = np.zeros_like(auto_pos)
            cross_sum = np.zeros_like(cross_win)

        auto_sum += auto_pos
        cross_sum += cross_win

        auto_peak_sum += met["auto_peak"]
        auto_area_sum += met["auto_area"]
        cross_peak_sum += met["cross_peak"]
        cross_area_sum += met["cross_area"]
        cross_t_peak_sum += met["cross_t_peak"]

        tau = met["auto_tau"]
        ok = np.isfinite(tau)
        auto_tau_sum[ok] += tau[ok]
        valid_auto_tau[ok] += 1

        print(f"[MC] finished {r+1}/{n_runs}")

    auto_avg = auto_sum / n_runs
    cross_avg = cross_sum / n_runs

    auto_peak_avg = auto_peak_sum / n_runs
    auto_area_avg = auto_area_sum / n_runs
    auto_tau_avg = np.full(d, np.nan)
    ok = valid_auto_tau > 0
    auto_tau_avg[ok] = auto_tau_sum[ok] / valid_auto_tau[ok]

    cross_peak_avg = cross_peak_sum / n_runs
    cross_area_avg = cross_area_sum / n_runs
    cross_t_peak_avg = cross_t_peak_sum / n_runs

    return dict(
        lag_ms=lag_ms_ref,
        auto_cov_pos_avg=auto_avg,
        cross_cov_win_avg=cross_avg,
        metrics_avg=dict(
            auto_peak=auto_peak_avg,
            auto_area=auto_area_avg,
            auto_tau=auto_tau_avg,
            cross_peak=cross_peak_avg,
            cross_area=cross_area_avg,
            cross_t_peak=cross_t_peak_avg,
        ),
        params=dict(
            n_runs=n_runs, T=T, dt=dt, q00=q00, burn_in=burn_in,
            target_dt=target_dt, max_lag=max_lag,
            fit_threshold=fit_threshold, fit_max_lag=fit_max_lag,
            seed0=seed0, noise_type=noise_type, tau_c=tau_c,
        ),
    )
