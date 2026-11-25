"""ER constant-input comparison and distance scaling on the original Euler grid."""
import numpy as np
from scipy.optimize import curve_fit
from scipy.sparse.csgraph import connected_components, shortest_path
from scaling_stats import save_by_distance


def draw_graph(rng, n, probability):
    edges = rng.random((n, n)) < probability
    np.fill_diagonal(edges, False)
    count, component = connected_components(edges, directed=True, connection="strong")
    if count > 1:
        representatives = [np.flatnonzero(component == k)[0] for k in range(count)]
        edges[representatives, np.roll(representatives, -1)] = True
    return edges


def exponential(t, tau):
    return np.exp(-t / tau)


def measure(edges, t, dt):
    n = len(edges)
    h = edges.T.astype(float) - np.diag(1 + edges.sum(axis=0))
    gain = np.linalg.solve(-h, np.eye(n)[0])
    moment = np.linalg.solve(-h, gain)
    # For step input the residual after k Euler steps equals (I + dt H)^k gain.
    # Compute blocks on the same time grid; each column is one initial residual.
    transition = np.eye(n) + dt * h
    width = 250
    block = np.linalg.matrix_power(transition, width)
    starts = np.empty((n, (len(t) + width - 1) // width))
    starts[:, 0] = gain
    for j in range(1, starts.shape[1]):
        starts[:, j] = block @ starts[:, j - 1]
    residual = np.empty((n, starts.shape[1], width))
    for k in range(width):
        residual[:, :, k] = starts
        starts = transition @ starts
    residual = residual.reshape(n, -1)[:, :len(t)]
    tau = np.empty(n)
    crossing = np.empty(n)
    for j, y in enumerate(residual):
        normalized = (y-y[-1])/(y[0]-y[-1])
        tau[j] = curve_fit(exponential, t, normalized)[0][0]
        crossing[j] = t[np.flatnonzero(normalized <= np.exp(-1))[0]-1]
    distances = shortest_path(edges, directed=True, indices=0, unweighted=True)
    return distances, gain, moment, moment/gain, tau, crossing


def main():
    rng = np.random.default_rng(20261125)
    t = np.linspace(0, 100, 100000)
    distances, gains, moments, times = [], [], [], []
    fitted, crossed = [], []
    for trial in range(100):
        print(f"ER constant trial {trial + 1}/100", flush=True)
        d, gain, moment, theory, tau, crossing = measure(draw_graph(rng, 100, .02), t, .001)
        distances.append(d)
        gains.append(gain)
        moments.append(moment)
        times.append(tau)
        if trial < 10:
            fitted.append(np.stack((theory, tau)))
            crossed.append(np.stack((theory, crossing)))
    save_by_distance("constant", distances, {"amplification": gains, "peak_response": moments,
                                            "time_constant": times})
    np.save("pic_two/constant_exptimeconstant_random_equal.npy", np.concatenate(fitted, axis=1))
    np.save("pic_two/constant_exptime_random_equal.npy", np.concatenate(crossed, axis=1))


if __name__ == "__main__":
    main()
