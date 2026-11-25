import numpy as np
import networkx as nx
import time
from numpy import linalg as LA
from collections import defaultdict
from scipy import integrate

class ODEEuler:

    def __init__(self, H, I, initial_v):
        self.H = H
        self.I = I
        self.initial_v = initial_v

    def H_function(self, t, v):
        return np.dot(self.H, v) + self.I(t)

    def I_function(self, t):
        return np.array([1, 0, ...])

    def solve(self, t_span, t_eval=None):
        values = np.zeros((len(t_eval), len(self.initial_v)))
        values[0] = self.initial_v
        dt = (t_span[1] - t_span[0]) / len(t_eval)
        for i in range(1, len(t_eval)):
            t = t_eval[i - 1]
            v = values[i - 1]
            values[i] = v + dt * self.H_function(t, v)
        return (t_eval, values)

def generate_input(input_type, params, dim):

    def make_array(value, default=0):
        return np.array(value) if isinstance(value, list) else np.full(dim, value)
    if input_type == 'constant':
        c = make_array(params.get('c', 0))

        def I(t):
            return c
    elif input_type == 'square':
        s = make_array(params.get('s', 0))
        start_time = params.get('start_time', 0)
        end_time = params.get('end_time', 1)

        def I(t):
            return s if start_time <= t <= end_time else np.zeros_like(s)
    elif input_type == 'noise':
        miu = make_array(params.get('miu', 0), default=None)
        sigma = make_array(params.get('sigma', 0), default=None)

        def I(t):
            noise = np.array([np.random.normal(m, s) if m is not None else 0 for m, s in zip(miu, sigma)])
            return noise
    elif input_type == 'pulse':
        p = make_array(params.get('p', 0))
        p_time = params.get('p_time', 0)

        def I(t):
            return p if np.isclose(t, p_time, atol=0.01) else np.zeros_like(p)
    elif input_type == 'zero':

        def I(t):
            return np.zeros(dim)
    else:
        raise ValueError("Invalid input type. Choose 'constant', 'pulse', 'noise', 'zero', or 'square'.")
    return I

def make_strongly_connected(G):
    if nx.is_strongly_connected(G):
        return G
    sccs = list(nx.strongly_connected_components(G))
    scc_subgraphs = [G.subgraph(scc).copy() for scc in sccs]
    for i in range(len(scc_subgraphs) - 1):
        source = next(iter(scc_subgraphs[i].nodes))
        target = next(iter(scc_subgraphs[i + 1].nodes))
        G.add_edge(source, target)
    last_source = next(iter(scc_subgraphs[-1].nodes))
    first_target = next(iter(scc_subgraphs[0].nodes))
    G.add_edge(last_source, first_target)
    return G

def generate_graph_matrix(graph_type, size, params):
    if graph_type == 'directed_chain':
        length = params.get('length', 1)
        G = nx.path_graph(length, create_using=nx.DiGraph())
    elif graph_type == 'ER_random':
        p = params.get('probability', 0.5)
        G = nx.erdos_renyi_graph(size, p, directed=True)
        G = make_strongly_connected(G)
    elif graph_type == 'small-world':
        k = params.get('nearest_neighbors', 2)
        p = params.get('probability', 0.5)
        G = nx.watts_strogatz_graph(size, k, p)
    elif graph_type == 'scale-free':
        G = nx.barabasi_albert_graph(size, params.get('attachments', 1))
    elif graph_type == 'geometric':
        radius = params.get('radius', 0.1)
        G = nx.random_geometric_graph(size, radius)
    else:
        raise ValueError('Invalid graph type')
    return (G, nx.to_numpy_array(G))

def generate_H(topology, diag_matrix):
    N = np.shape(topology)[0]
    D = np.zeros((N, N))
    for i in range(N):
        D[i][i] = sum(topology[i])
    return topology - D - diag_matrix

def generate_simulation(input_type, params, dim):

    def make_array(value, default=0):
        return np.array(value) if isinstance(value, list) else np.full(dim, value)
    if input_type == 'square':

        def judge_stable(y_plot):
            stable = np.abs(y_plot[-2] - y_plot[-1]) / np.max(y_plot)
            if stable < 0.001:
                return 1
            else:
                print(stable)
                return 0
        t = make_array(params.get('t', 0))
        y = make_array(params.get('y', 0))
        eta = params.get('eta', 0)
        peak, amp = [[] for i in range(2)]
        for i in np.arange(np.shape(y)[1]):
            y_plot = y[:, i]
            if judge_stable(y_plot):
                peak.append(np.max(y_plot))
                area = integrate.trapezoid(y_plot, t)
                amp.append(area)
            else:
                print('time is not long enough')
        return (peak, amp)

def generate_theory(input_type, params, dim):

    def make_array(value, default=0):
        return np.array(value) if isinstance(value, list) else value
    if input_type == 'square':
        H = make_array(params.get('H', 0))
        strength = make_array(params.get('strength', 0))
        m = make_array(params.get('m', 0))
        matrix = make_array(params.get('matrix', 0))
        eta = make_array(params.get('eta', 0))
        ts = make_array(params.get('ts', 0))
        peak_theory, amp_theory, path = [[] for i in range(3)]
        H_inv = np.linalg.inv(H)
        H_inv_two = np.linalg.inv(np.linalg.matrix_power(H, 2))
        for area_idx in range(dim):
            tau = -H_inv_two[area_idx][m] / H_inv[area_idx][m]
            peak_theory.append(-H_inv[area_idx][m] * (1.0 - np.exp(-ts / tau)))
            amp_theory.append(-H_inv[area_idx][m] * ts)
            G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
            d = nx.dijkstra_path_length(G, m, area_idx, weight=1)
            path.append(d)
        return (peak_theory, amp_theory, path)

def chain_ts(N, weight, t_span, t_evel, start_time, end_time, p):
    G, matrix = generate_graph_matrix('ER_random', N, {'probability': p})
    matrix = matrix.T
    matrix = weight * matrix
    diag_matrix = np.zeros((N, N))
    beta = 1.0
    for i in range(N):
        diag_matrix[i][i] = beta
    H = generate_H(matrix, diag_matrix)
    eigenvalues, eigenvectors = LA.eig(H)
    m = 0
    strength = 1
    eta = 1.0 - 1.0 / np.exp(1)
    initial_v = [0 for i in range(N)]
    input_type = 'square'
    params = {'s': [0 if i != 0 else strength for i in range(N)], 'start_time': start_time, 'end_time': end_time}
    ts = end_time - start_time
    I = generate_input(input_type, params, N)
    solver_euler = ODEEuler(H, I, initial_v)
    t_eval, solution_euler = solver_euler.solve(t_span, t_evel)
    simu_params = {'t': list(t_eval), 'y': list(solution_euler), 'eta': eta, 'ts': ts}
    theory_params = {'H': list(H), 'strength': strength, 'm': m, 'matrix': list(matrix.T), 'eta': eta, 'ts': ts}

    def make_array(value, default=0):
        return np.array(value) if isinstance(value, list) else value
    t = make_array(simu_params.get('t', 0))
    y = make_array(simu_params.get('y', 0))
    peak, amp = generate_simulation(input_type, simu_params, N)
    peak_theory, amp_theory, path = generate_theory(input_type, theory_params, N)
    return (amp, amp_theory, peak, peak_theory, path)

def compute_stats(path, y_values):
    value_dict = defaultdict(list)
    for p, y in zip(path, y_values):
        value_dict[p].append(y)
    sorted_paths = sorted(value_dict.keys())
    means = []
    stds = []
    for p in sorted_paths:
        values = value_dict[p]
        means.append(np.mean(values))
        stds.append(np.std(values))
    return (sorted_paths, means, stds)

def record_simulations():
    N = 100
    p = 0.02
    data_save = []
    weight = 1.0
    t_span = [0, 150]
    t_evel = np.linspace(0, 150, 200000)
    start_time = 0
    end_time = 10
    round_num = 100
    amp_list, amp_theory_list, peak_list, peak_theory_list, path_list = [[] for _ in range(5)]
    for j in range(round_num):
        t1 = time.time()
        amp, amp_theory, peak, peak_theory, path = chain_ts(N, weight, t_span, t_evel, start_time, end_time, p)
        amp_list.append(amp)
        amp_theory_list.append(amp_theory)
        peak_list.append(peak)
        peak_theory_list.append(peak_theory)
        path_list.append(path)
        t2 = time.time()
        print(j, t2 - t1)
    total_path_list = [item for sublist in path_list for item in sublist]
    total_amp_list = [item for sublist in amp_list for item in sublist]
    total_peak_list = [item for sublist in peak_list for item in sublist]
    paths, means, stds = compute_stats(total_path_list, total_amp_list)
    np.save('scaling/square_amplification_ER.npy', [paths, means, stds])
    paths, means, stds = compute_stats(total_path_list, total_peak_list)
    np.save('scaling/square_peak_response_ER.npy', [paths, means, stds])
    np.save('pic_two/square_peak_random_equal.npy', np.stack([np.concatenate([np.ravel(row) for row in trials]) for trials in (peak_theory_list, peak_list)]))
    print('save success!')

if __name__ == "__main__":
    record_simulations()
