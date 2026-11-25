"""Localized white-noise response in dx/dt = Hx + e0 u(t)."""
import sys

import numpy as np
from scipy.linalg import expm, solve
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

SEED = 20261125
N = 100
DT = 0.1
SAMPLES = 4096
RECORDS = 64
TRIALS = 10


def adjacency(kind, n, rng):
    """Rows are source nodes. Connect separate components before transposing."""
    a = np.zeros((n, n))
    idx = np.arange(n)
    if kind in ("undirectedchain", "directedchain"):
        a[idx[:-1], idx[1:]] = 1
        return a + a.T if kind == "undirectedchain" else a
    if kind == "random":
        a = (rng.random((n, n)) < 0.02).astype(float)
        np.fill_diagonal(a, 0)
    elif kind in ("regular", "small-world"):
        for d in range(1, 3 if kind == "regular" else 2):
            a[idx, (idx + d) % n] = a[(idx + d) % n, idx] = 1
        if kind == "small-world":
            for i in idx:
                j = (i + 1) % n
                candidates = idx[(idx != i) & (a[i] == 0)]
                if rng.random() < .5 and candidates.size:
                    k = rng.choice(candidates)
                    a[i,j] = a[j,i] = 0
                    a[i,k] = a[k,i] = 1
    elif kind == "scale-free":
        a[0,1] = a[1,0] = 1
        for i in range(2,n):
            degree = a[:i,:i].sum(axis=1)
            j = rng.choice(i, p=degree/degree.sum())
            a[i,j] = a[j,i] = 1
    elif kind == "geometric":
        xy = rng.random((n,2))
        a = (np.linalg.norm(xy[:,None]-xy, axis=2) < .2).astype(float)
        np.fill_diagonal(a,0)
    count, labels = connected_components(a, directed=True, connection="strong")
    if count > 1:
        reps = [np.flatnonzero(labels == k)[0] for k in range(count)]
        a[reps, np.roll(reps,-1)] = 1
    return a


def measure(a, rng):
    n = len(a)
    h = a.T - np.diag(1 + a.sum(axis=0))
    source = np.eye(n)[0]
    gain = solve(-h, source)
    first = solve(-h, gain)
    tau = first / gain
    variance = 2*solve(-h, first)/gain - tau**2
    transition = expm(h*DT)
    # Response to a unit impulse integrated over one input bin.
    impulse = solve(h, (transition-np.eye(n)) @ source)
    kernel = np.empty((SAMPLES,n))
    step = csr_matrix(transition)
    for k in range(SAMPLES):
        kernel[k] = impulse
        impulse = step @ impulse
    spectrum = np.fft.rfft(kernel, axis=0)
    # Each row is a fresh periodic Gaussian input. Averaging its periodogram
    # before applying the transfer functions equals averaging output covariances.
    inputs = rng.normal(size=(RECORDS,SAMPLES))/np.sqrt(DT)
    inputs -= inputs.mean(axis=1, keepdims=True)
    power = np.mean(abs(np.fft.rfft(inputs,axis=1))**2,axis=0)/SAMPLES
    maxlag = min(SAMPLES//2-1, int((max(tau)+20)/DT))
    lags = DT*np.arange(-maxlag,maxlag+1)
    peak, lag, area = [np.empty((n,n)) for _ in range(3)]
    fit, crossing = np.empty(n), np.empty(n)
    for i in range(n):
        # C[i,j](s) = E[x_i(t) x_j(t+s)].
        cov = np.fft.irfft((power*spectrum[:,i].conj())[:,None]*spectrum,axis=0)
        window = np.concatenate((cov[-maxlag:],cov[:maxlag+1]))
        best = np.argmax(window,axis=0)
        peak[i] = window[best,np.arange(n)]
        lag[i] = lags[best]
        area[i] = np.trapezoid(window,dx=DT,axis=0) if hasattr(np,'trapezoid') else np.trapz(window,dx=DT,axis=0)
        ac = cov[:maxlag+1,i]/cov[0,i]
        hits = np.flatnonzero(ac <= np.exp(-1))
        if not hits.size:
            raise ValueError("Autocorrelation exceeds lag window; increase SAMPLES or the lag window.")
        crossing[i] = hits[0]*DT
        end = np.flatnonzero(ac < .1)
        y = ac[:max(2,int(end[0]))] if end.size else ac
        t = np.arange(len(y))*DT
        fit[i] = minimize_scalar(lambda v: np.sum((y-np.exp(-t/v))**2),
                                 bounds=(DT/20,10*t[-1]),method="bounded").x
    # Integral correlation time = gain^2/(2 Pii). Integrate the squared
    # continuous impulse response using midpoint samples (also stable for a chain).
    vector = expm(h*(DT/2)) @ source
    diag = np.zeros(n)
    for _ in range(SAMPLES):
        diag += DT*vector**2
        vector = step @ vector
    autocorr_time = gain**2/(2*diag)
    predicted_lag = tau[None,:]-tau[:,None]
    predicted_peak = np.outer(gain,gain)/np.sqrt(2*(variance[:,None]+variance))
    return autocorr_time, fit, crossing, predicted_lag, lag, predicted_peak, peak, area


def run(panels):
    from scaling_stats import save_by_distance
    graphs = ["undirectedchain","regular","random","small-world","scale-free","geometric","directedchain"]
    for index, kind in enumerate(graphs):
        comparison = kind != "directedchain" and bool(panels & {"noiseauto","noisecov"})
        scaling = "noise-scaling" in panels and kind in ("random","directedchain")
        if not (comparison or scaling):
            continue
        rows, distances, peaks, times, areas = {}, [], [], [], []
        trials = 1 if kind in ("undirectedchain","regular","directedchain") else TRIALS
        for trial in range(trials):
            rng = np.random.default_rng([SEED,index,trial])
            a = adjacency(kind,N,rng)
            d = shortest_path(a,directed=True,unweighted=True)
            theory, fit, crossing, tl, sl, tp, sp, area = measure(a,rng)
            def collect(key,x,y):
                rows.setdefault(key,[]).append(np.stack((x,y)))
            if comparison and "noiseauto" in panels:
                keep = np.ones(N,dtype=bool) if kind=="undirectedchain" else d[0]<=10
                collect("noiseauto_exptime",theory[keep],crossing[keep])
                collect("noiseauto_exptimeconstant",theory[keep],fit[keep])
            if comparison and "noisecov" in panels:
                if kind=="undirectedchain":
                    collect("noisecov_peaktime",tl,sl)
                    collect("noisecov_peak",tp,sp)
                else:
                    mask = (d.T[0,:,None]<10)&(d.T[0,None,:]<10)&(d<10)&(d.T<10)
                    mask[0,:]=mask[:,0]=True
                    mask &= (tl>1e-8)&(sl>1e-8)&(tp>1e-8)&(sp>1e-8)
                    collect("noisecov_peaktime",tl[mask],sl[mask])
                    collect("noisecov_peak",tp[mask],sp[mask])
            if scaling:
                distances.append(d[0]); peaks.append(sp[0]); times.append(sl[0]); areas.append(area[0])
        for key,values in rows.items():
            graph = "ER_random" if kind=="random" and key.startswith("noisecov") else kind
            np.save(f"pic_two/{key}_{graph}_equal.npy",np.concatenate(values,axis=1))
        if scaling:
            metrics = {"peak_response":peaks,"time_constant":times,"amplification":areas}
            if kind=="random":
                save_by_distance("noise",distances,metrics)
            else:
                for key,values in metrics.items():
                    np.save(f"scaling/noise_{key}.npy",values[0])
        print(f"Generated {kind} ({trials} realizations)",flush=True)

if __name__=="__main__":
    run(set(sys.argv[1:]))
