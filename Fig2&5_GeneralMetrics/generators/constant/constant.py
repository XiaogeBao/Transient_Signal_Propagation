"""Constant-input chain and regular-ring metrics used in both figures."""
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


def exponential(t, tau):
    return np.exp(-t / tau)


def generate(topology, n=100, duration=100, samples=100000):
    nodes = np.arange(n)
    adjacency = np.zeros((n, n))
    if topology == "directedchain":
        adjacency[nodes[1:], nodes[:-1]] = 1
    elif topology == "regular":
        for shift in (-2, -1, 1, 2):
            adjacency[nodes, (nodes + shift) % n] = 1
    else:
        raise ValueError(topology)
    h = adjacency - np.diag(1 + adjacency.sum(axis=1))
    source = np.eye(n)[0]
    gain = np.linalg.solve(-h, source)
    amplification = np.linalg.solve(-h, gain)
    theory_tau = amplification / gain

    # Retain the original Euler step and the original stored time labels.
    t = np.linspace(0, duration, samples)
    dt = duration / samples
    response = np.zeros((samples, n))
    for k in range(1, samples):
        previous = response[k - 1]
        response[k] = previous + dt * (h @ previous + source)

    eta = 1 - np.exp(-1)
    measured_tau = np.empty(n)
    measured_time = np.empty(n)
    for node in nodes:
        trace = response[:, node]
        if abs(trace[-1] - trace[-2]) >= 0.001 * trace[-1]:
            raise ValueError(f"Response at node {node} has not settled by t={duration}")
        residual = trace[-1] - trace
        normalized = (residual - residual.min()) / np.ptp(residual)
        measured_tau[node] = curve_fit(exponential, t, normalized)[0][0]
        level = eta * trace.max()
        crossing = np.flatnonzero((trace[:-1] <= level) & (trace[1:] >= level))
        measured_time[node] = t[crossing[0]]

    prefix = Path("pic_two")
    np.save(prefix / f"constant_exptimeconstant_{topology}_equal.npy", [theory_tau, measured_tau])
    np.save(prefix / f"constant_exptime_{topology}_equal.npy",
            [-np.log1p(-eta) * theory_tau, measured_time])
    if topology == "directedchain":
        for name, values in (("amplification", amplification), ("peak_response", gain),
                             ("time_constant", measured_tau)):
            np.save(Path("scaling") / f"constant_{name}.npy", values)


if __name__ == "__main__":
    for directory in ("pic_two", "scaling"):
        Path(directory).mkdir(exist_ok=True)
    for topology in ("directedchain", "regular"):
        generate(topology)
