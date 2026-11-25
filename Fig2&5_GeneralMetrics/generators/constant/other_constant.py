import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import math
from collections import Counter
import pickle

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
    elif graph_type == 'small-world':
        k = params.get('nearest_neighbors', 2)
        p = params.get('probability', 0.5)
        G = nx.watts_strogatz_graph(size, k, p)
    elif graph_type == 'scale-free':
        G = nx.barabasi_albert_graph(size, params.get('attachments', 1))
    elif graph_type == 'geometric':
        radius = params.get('radius', 0.1)
        G = nx.random_geometric_graph(size, radius)
    elif graph_type == 'regular':
        k = params.get('nearest_neighbors', 2)
        G = nx.watts_strogatz_graph(size, k, 0)
    else:
        raise ValueError('Invalid graph type')
    G = G.to_directed()
    G = make_strongly_connected(G)
    return (G, nx.to_numpy_array(G))

def generate_H(topology, diag_matrix):
    N = np.shape(topology)[0]
    D = np.zeros((N, N))
    for i in range(N):
        D[i][i] = sum(topology[i])
    L = topology - D - diag_matrix
    for i in range(N):
        if np.abs(L[i, i]) < np.sum(np.abs(L[i, :])) - np.abs(L[i, i]):
            print('no laplacian')
            return False
    return L

def generate_simulation(input_type, params, dim):

    def single_exp(x, a):
        return np.exp(-x / a)

    def sigmoid(x, a):
        return 2.0 / (1.0 + np.exp(x / a))

    def make_array(value, default=0):
        return np.array(value) if isinstance(value, list) else np.full(dim, value)
    if input_type == 'constant':

        def judge_stable(y_plot):
            stable = np.abs(y_plot[-2] - y_plot[-1]) / y_plot[-1]
            if stable < 0.001:
                return 1
            else:
                print(stable)
                return 0
        t = make_array(params.get('t', 0))
        y = make_array(params.get('y', 0))
        eta = params.get('eta', 0)
        threshold_time, decay_time, sigmoid_time = [[] for i in range(3)]
        for i in np.arange(np.shape(y)[1]):
            y_plot = y[:, i]
            if judge_stable(y_plot):
                fit_curve = y_plot[-1] - y_plot
                fit_curve = (fit_curve - np.min(fit_curve)) / (np.max(fit_curve) - np.min(fit_curve))
                t_plot = t[:len(fit_curve)]
                r_single, _ = curve_fit(single_exp, t_plot, fit_curve)
                plt.plot(t_plot, fit_curve, color='gray', alpha=0.5)
                decay_time.append(r_single[0])
                r_sigmoid, _ = curve_fit(sigmoid, t_plot, fit_curve)
                sigmoid_time.append(r_sigmoid[0])
                delta_y = y[:, i] - y[0, i]
                y_val = np.max(delta_y) * eta
                time_scale_up = [t[j] for j in range(0, len(delta_y) - 1) if delta_y[j] <= y_val and delta_y[j + 1] >= y_val][0]
                threshold_time.append(time_scale_up)
            else:
                print('time is not long enough')
        plt.title('exp fitting curve')
        plt.close("all")
        if 1:
            for i in range(3):
                y_plot = y[:, i] - y[0, i]
                plt.plot(t, y_plot / np.max(y_plot))
            plt.xlim([0, 6])
            plt.close("all")
        return (decay_time, threshold_time, sigmoid_time)

def generate_theory(input_type, params, dim):

    def make_array(value, default=0):
        return np.array(value) if isinstance(value, list) else value
    if input_type == 'constant':
        H = make_array(params.get('H', 0))
        strength = make_array(params.get('strength', 0))
        m = make_array(params.get('m', 0))
        matrix = make_array(params.get('matrix', 0))
        eta = make_array(params.get('eta', 0))
        threshold, decay, decay_high, path = [[] for i in range(4)]
        peak, amp, sigmoid = [[] for i in range(3)]
        H_inv = np.linalg.inv(H)
        H_inv_two = np.linalg.inv(np.linalg.matrix_power(H, 2))
        H_inv_three = np.linalg.inv(np.linalg.matrix_power(H, 3))
        for area_idx in range(dim):
            peak.append(-H_inv[area_idx][m])
            amp.append(H_inv_two[area_idx][m])
            decay.append(-H_inv_two[area_idx][m] / H_inv[area_idx][m])
            decay_high.append(-H_inv_three[area_idx][m] / H_inv_two[area_idx][m])
            threshold.append(np.log(1.0 - eta) * H_inv_two[area_idx][m] / H_inv[area_idx][m])
            sigmoid.append(np.log((1.0 - eta) / (1.0 + eta)) * (H_inv_two[area_idx][m] / H_inv[area_idx][m]) / (2 * np.log(2)))
            G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
            d = nx.dijkstra_path_length(G, m, area_idx, weight=1)
            path.append(d)
        print('time constant:', decay)
        return (threshold, decay, decay_high, path, sigmoid, peak, amp)

def chain_simulation_dotting(N, weight, t_span, t_evel, net_name, para_set):
    simu_data, theory_data, matrix_list = [[] for i in range(3)]
    for i in range(100):
        print('round:', i)
        G, matrix = generate_graph_matrix(net_name, N, para_set)
        matrix = matrix.T
        matrix = weight * matrix
        diag_matrix = np.zeros((N, N))
        beta = 1.0
        for i in range(N):
            diag_matrix[i][i] = beta
        H = generate_H(matrix, diag_matrix)
        m = 0
        strength = 1
        eta = 1.0 - 1.0 / np.exp(1)
        initial_v = [0 for i in range(N)]
        input_type = 'constant'
        params = {'c': [0 if i != 0 else strength for i in range(N)]}
        I = generate_input(input_type, params, N)
        I_values = np.array([I(ti) for ti in t_evel])
        solver_euler = ODEEuler(H, I, initial_v)
        t_eval, solution_euler = solver_euler.solve(t_span, t_evel)
        simu_params = {'t': list(t_eval), 'y': list(solution_euler), 'eta': eta}
        theory_params = {'H': list(H), 'strength': strength, 'm': m, 'matrix': list(matrix.T), 'eta': eta}

        def make_array(value, default=0):
            return np.array(value) if isinstance(value, list) else value
        t = make_array(simu_params.get('t', 0))
        y = make_array(simu_params.get('y', 0))
        dim = N
        decay_time, threshold_time, sigmoid_time = generate_simulation(input_type, simu_params, N)
        threshold, decay, decay_high, path, sigmoid, peak, amp = generate_theory(input_type, theory_params, N)
        simu_data.append([decay_time, threshold_time, sigmoid_time])
        theory_data.append([threshold, decay, decay_high, path, sigmoid, peak, amp])
        matrix_list.append(matrix)
    random_data_dot_save = {'simu': simu_data, 'theo': theory_data, 'matrix': matrix_list}
    with open(net_name + '_constant_dot.pkl', 'wb') as file:
        pickle.dump(random_data_dot_save, file)

def dotted_plotting(theory_data, simu_data, G_list, net_name):
    t_tau, s_tau, s_tau_sig = [[] for i in range(3)]
    t_time, t_time_sig, s_time = [[] for i in range(3)]
    for i in range(100):
        t_tau += list(theory_data[0][1])
        s_tau += list(simu_data[0][0])
        s_tau_sig += list(simu_data[0][2])
        t_time += list(theory_data[0][0])
        t_time_sig += list(theory_data[0][4])
        s_time += list(simu_data[0][1])
    in_degree_distributions = []
    out_degree_distributions = []
    for G in G_list:
        out_degrees = np.sum(G, axis=0)
        in_degrees = np.sum(G, axis=1)
        in_degree_distribution = dict(Counter(in_degrees))
        out_degree_distribution = dict(Counter(out_degrees))
        in_degree_distributions.append(in_degree_distribution)
        out_degree_distributions.append(out_degree_distribution)

    def calculate_average_and_errors(distributions):
        all_degrees = set(sum([list(dist.keys()) for dist in distributions], []))
        avg_distribution = {}
        std_errors = {}
        for degree in all_degrees:
            values = [dist.get(degree, 0) for dist in distributions]
            avg_distribution[degree] = np.mean(values)
            std_errors[degree] = np.std(values)
        total = sum(avg_distribution.values())
        avg_distribution = {k: v / total for k, v in avg_distribution.items()}
        std_errors = {k: v / total for k, v in std_errors.items()}
        return (avg_distribution, std_errors)
    avg_in_distribution, in_errors = calculate_average_and_errors(in_degree_distributions)
    avg_out_distribution, out_errors = calculate_average_and_errors(out_degree_distributions)
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    degrees = sorted(avg_in_distribution.keys())
    avg_values = [avg_in_distribution[degree] for degree in degrees]
    std_values = [in_errors[degree] for degree in degrees]
    plt.bar(degrees, avg_values, yerr=std_values, capsize=5, color='gray')
    plt.xlabel('$indegree$', fontsize=15)
    plt.ylabel('$probability$', fontsize=15)
    plt.xticks(np.arange(1, np.max(degrees) + 1, 2), rotation=45)
    plt.subplot(1, 2, 2)
    degrees = sorted(avg_out_distribution.keys())
    avg_values = [avg_out_distribution[degree] for degree in degrees]
    std_values = [in_errors[degree] for degree in degrees]
    plt.bar(degrees, avg_values, yerr=std_values, capsize=5, color='gray')
    plt.xticks(np.arange(1, np.max(degrees) + 1, 2), rotation=45)
    plt.xlabel('$outdegree$', fontsize=15)
    plt.tight_layout()
    plt.close("all")
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    degrees = np.array(sorted(avg_in_distribution.keys()))
    avg_values = np.array([avg_in_distribution[degree] for degree in degrees])
    std_values = [in_errors[degree] for degree in degrees]
    plt.plot(degrees, avg_values, color='black')
    plt.fill_between(degrees, avg_values - std_values, avg_values + std_values, color='gray', alpha=0.8)
    plt.xlabel('In-Degree', fontsize=15)
    plt.ylabel('Probability', fontsize=15)
    plt.xlim([1, np.max(degrees) + 1])
    plt.xticks(np.arange(1, np.max(degrees) + 1, 7))
    plt.subplot(1, 2, 2)
    degrees = np.array(sorted(avg_out_distribution.keys()))
    avg_values = np.array([avg_out_distribution[degree] for degree in degrees])
    std_values = [out_errors[degree] for degree in degrees]
    plt.plot(degrees, avg_values, color='black')
    plt.fill_between(degrees, avg_values - std_values, avg_values + std_values, color='gray', alpha=0.8)
    plt.xlabel('Out-Degree', fontsize=15)
    plt.xlim([1, np.max(degrees) + 1])
    plt.xticks(np.arange(1, np.max(degrees) + 1, 7))
    plt.tight_layout()
    plt.close("all")

    def compute_distribution(degrees):
        degree_distribution = Counter(degrees)
        total = sum(degree_distribution.values())
        degree_distribution = {k: v / total for k, v in degree_distribution.items()}
        return degree_distribution
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    for G in G_list:
        in_degrees = np.sum(G, axis=1)
        in_degree_distribution = compute_distribution(in_degrees)
        degrees = np.array(list(in_degree_distribution.keys()))
        probabilities = np.array([in_degree_distribution[degree] for degree in degrees])
        plt.scatter(degrees, probabilities, color='gray', s=2, edgecolors='none')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('In-Degree', fontsize=15)
    plt.ylabel('Probability', fontsize=15)
    plt.subplot(1, 2, 2)
    for G in G_list:
        out_degrees = np.sum(G, axis=0)
        out_degree_distribution = compute_distribution(out_degrees)
        degrees = np.array(list(out_degree_distribution.keys()))
        probabilities = np.array([out_degree_distribution[degree] for degree in degrees])
        plt.scatter(degrees, probabilities, color='gray', s=2, edgecolors='none')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Out-Degree', fontsize=15)
    plt.tight_layout()
    plt.close("all")
    amp_max = 100
    time_max = 100
    decay_max = 100
    print('initial:', len(t_tau), len(s_tau), len(s_tau_sig))
    for i in reversed(range(len(t_tau))):
        if any((x is None or math.isnan(x) or x > amp_max for x in [t_tau[i], s_tau[i], s_tau_sig[i]])):
            del t_tau[i]
            del s_tau[i]
            del s_tau_sig[i]
    print('after:', len(t_tau), len(s_tau), len(s_tau_sig))
    plt.figure(figsize=(4, 4))
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('$theory$', fontsize=15)
    plt.ylabel('$simulation$', fontsize=15)
    a, b = (t_tau, s_tau)
    modelone = LinearRegression()
    modelone.fit(np.array(a).reshape(-1, 1), np.array(b).reshape(-1, 1))
    predicts = modelone.predict(np.array(a).reshape(-1, 1))
    R2 = modelone.score(np.array(a).reshape(-1, 1), np.array(b).reshape(-1, 1))
    R2 = round(R2, 4)
    ax.scatter(a[::10], b[::10], color='gray', alpha=0.5, label='$\\tau_i$', s=20)
    print('direct coef_:%.3f' % modelone.coef_.item())
    print('direct r^2:%.3f' % R2)
    c, d = (1 / (2 * np.log(2)) * np.array(t_tau), s_tau_sig)
    modeltwo = LinearRegression()
    modeltwo.fit(np.array(c).reshape(-1, 1), np.array(d).reshape(-1, 1))
    predicts = modeltwo.predict(np.array(c).reshape(-1, 1))
    R2 = modeltwo.score(np.array(c).reshape(-1, 1), np.array(d).reshape(-1, 1))
    R2 = round(R2, 4)
    ax.scatter(c[::10], d[::10], color='red', alpha=0.3, label='$\\tau_i^s$', s=20)
    print('amend coef_:%.3f' % modeltwo.coef_.item())
    print('amend r^2:%.3f' % R2)
    combined_list = list(a) + list(b) + list(c) + list(d)
    ax.plot(np.linspace(min(combined_list), max(combined_list), 1000), np.linspace(min(combined_list), max(combined_list), 1000), color='black', linewidth=2, linestyle='dashed', alpha=0.8)
    plt.legend(frameon=False, prop={'size': 15})
    plt.tight_layout()
    plt.close("all")
    print('initial:', len(t_time), len(t_time_sig), len(s_time))
    for i in reversed(range(len(t_time))):
        if any((x is None or math.isnan(x) or x < 0 or (x > time_max) for x in [t_time[i], t_time_sig[i], s_time[i]])):
            del t_time[i]
            del t_time_sig[i]
            del s_time[i]
    print('after:', len(t_time), len(t_time_sig), len(s_time))
    plt.figure(figsize=(4, 4))
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('$theory$', fontsize=15)
    plt.ylabel('$simulation$', fontsize=15)
    a, b = (t_time, s_time)
    modelone = LinearRegression()
    modelone.fit(np.array(a).reshape(-1, 1), np.array(b).reshape(-1, 1))
    predicts = modelone.predict(np.array(a).reshape(-1, 1))
    R2 = modelone.score(np.array(a).reshape(-1, 1), np.array(b).reshape(-1, 1))
    R2 = round(R2, 4)
    ax.scatter(a[::10], b[::10], color='gray', alpha=0.5, label='$\\bar{t}_i$', s=20)
    print('direct coef_:%.3f' % modelone.coef_.item())
    print('direct r^2:%.3f' % R2)
    c, d = (t_time_sig, s_time)
    modeltwo = LinearRegression()
    modeltwo.fit(np.array(c).reshape(-1, 1), np.array(d).reshape(-1, 1))
    predicts = modeltwo.predict(np.array(c).reshape(-1, 1))
    R2 = modeltwo.score(np.array(c).reshape(-1, 1), np.array(d).reshape(-1, 1))
    R2 = round(R2, 4)
    ax.scatter(c[::10], d[::10], color='red', alpha=0.3, label='$\\bar{t}_i^s$', s=20)
    print('amend coef_:%.3f' % modeltwo.coef_.item())
    print('amend r^2:%.3f' % R2)
    combined_list = list(a) + list(b) + list(c) + list(d)
    ax.plot(np.linspace(min(combined_list), max(combined_list), 1000), np.linspace(min(combined_list), max(combined_list), 1000), color='black', linewidth=2, linestyle='dashed', alpha=0.8)
    plt.legend(frameon=False, prop={'size': 15})
    plt.tight_layout()
    plt.close("all")
    return (t_tau, s_tau, t_time, s_time, 1 / (2 * np.log(2)) * np.array(t_tau), s_tau_sig, t_time_sig, s_time)

if __name__ == "__main__":
    weight = 1.0
    t_span = [0, 100]
    t_evel = np.linspace(0, 100, 100000)
    N = 100
    net_name = 'small-world'
    para_set = {'nearest_neighbors': 2, 'probability': 0.5}
    chain_simulation_dotting(N, weight, t_span, t_evel, net_name, para_set)
    net_name = 'small-world'
    with open(net_name + '_constant_dot.pkl', 'rb') as file:
        loaded_data_dot = pickle.load(file)
    theory_data = loaded_data_dot['theo']
    simu_data = loaded_data_dot['simu']
    G_list = loaded_data_dot['matrix']
    thr_timeconstant_exp, sim_timeconstant_exp, thr_time_exp, sim_time_exp, thr_timeconstant_sig, sim_timeconstant_sig, thr_time_sig, sim_time_sig = dotted_plotting(theory_data, simu_data, G_list, net_name)
    np.save('pic_two/constant_exptimeconstant_' + net_name + '_equal.npy', [thr_timeconstant_exp, sim_timeconstant_exp])
    np.save('pic_two/constant_exptime_' + net_name + '_equal.npy', [thr_time_exp, sim_time_exp])
    print('save success!')
    weight = 1.0
    t_span = [0, 100]
    t_evel = np.linspace(0, 100, 100000)
    N = 100
    net_name = 'scale-free'
    para_set = {'attachments': 1}
    chain_simulation_dotting(N, weight, t_span, t_evel, net_name, para_set)
    net_name = 'scale-free'
    with open(net_name + '_constant_dot.pkl', 'rb') as file:
        loaded_data_dot = pickle.load(file)
    theory_data = loaded_data_dot['theo']
    simu_data = loaded_data_dot['simu']
    G_list = loaded_data_dot['matrix']
    thr_timeconstant_exp, sim_timeconstant_exp, thr_time_exp, sim_time_exp, thr_timeconstant_sig, sim_timeconstant_sig, thr_time_sig, sim_time_sig = dotted_plotting(theory_data, simu_data, G_list, net_name)
    np.save('pic_two/constant_exptimeconstant_' + net_name + '_equal.npy', [thr_timeconstant_exp, sim_timeconstant_exp])
    np.save('pic_two/constant_exptime_' + net_name + '_equal.npy', [thr_time_exp, sim_time_exp])
    print('save success!')
    weight = 1.0
    t_span = [0, 100]
    t_evel = np.linspace(0, 100, 100000)
    N = 100
    net_name = 'geometric'
    para_set = {'radius': 0.2}
    chain_simulation_dotting(N, weight, t_span, t_evel, net_name, para_set)
    net_name = 'geometric'
    with open(net_name + '_constant_dot.pkl', 'rb') as file:
        loaded_data_dot = pickle.load(file)
    theory_data = loaded_data_dot['theo']
    simu_data = loaded_data_dot['simu']
    G_list = loaded_data_dot['matrix']
    thr_timeconstant_exp, sim_timeconstant_exp, thr_time_exp, sim_time_exp, thr_timeconstant_sig, sim_timeconstant_sig, thr_time_sig, sim_time_sig = dotted_plotting(theory_data, simu_data, G_list, net_name)
    np.save('pic_two/constant_exptimeconstant_' + net_name + '_equal.npy', [thr_timeconstant_exp, sim_timeconstant_exp])
    np.save('pic_two/constant_exptime_' + net_name + '_equal.npy', [thr_time_exp, sim_time_exp])
    print('save success!')
