import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import math
from collections import Counter
import pickle
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
        G = make_strongly_connected(G)
    else:
        raise ValueError('Invalid graph type')
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

    def make_array(value, default=0):
        return np.array(value) if isinstance(value, list) else np.full(dim, value)
    if input_type == 'pulse' or input_type == 'zero':

        def judge_stable(y_plot):
            stable = np.abs(y_plot[-1] - y_plot[-2]) / np.max(y_plot)
            if stable < 0.001:
                return 1
            else:
                print('unstable')
                return 0

        def truncate_list(a):
            return a[:next((i for i, value in enumerate(a) if value < a[0] * 0.01), len(a))]
        t = make_array(params.get('t', 0))
        y = make_array(params.get('y', 0))
        peak = np.max(y, axis=0)
        peak_time = [t[i] for i in np.argmax(y, axis=0)]
        decay_time = []
        amp = []
        for i in np.arange(np.shape(y)[1]):
            y_plot = y[:, i]
            if judge_stable(y_plot):
                area = integrate.trapezoid(y_plot, t)
                amp.append(area)
                plt.plot(t, y_plot, color='gray', alpha=0.5)
                fit_curve = y_plot[np.argmax(y_plot):]
                fit_curve = (fit_curve - np.min(fit_curve)) / (np.max(fit_curve) - np.min(fit_curve))
                fit_curve = truncate_list(fit_curve)
                t_plot = t[:len(fit_curve)]
                r_single, _ = curve_fit(single_exp, t_plot, fit_curve)
                decay_time.append(r_single[0])
            else:
                print('time is not long enough')
        plt.close("all")
        return (peak, peak_time, decay_time, amp)

def generate_theory(input_type, params, dim):

    def make_array(value, default=0):
        return np.array(value) if isinstance(value, list) else value
    if input_type == 'pulse' or input_type == 'zero':
        H = make_array(params.get('H', 0))
        strength = make_array(params.get('strength', 0))
        m = make_array(params.get('m', 0))
        matrix = make_array(params.get('matrix', 0))
        initial_v = [0 if i != m else strength for i in range(dim)]
        try:
            eigen_val, eigen_matrix = np.linalg.eig(H)
            eigen_matrix_inv = np.linalg.inv(eigen_matrix)
        except:
            eigen_matrix_inv = np.zeros((np.shape(H)[0], np.shape(H)[0]))
        if 0:
            real_parts = eigen_val.real
            imaginary_parts = eigen_val.imag
            plt.figure(figsize=(8, 6))
            plt.scatter(real_parts, imaginary_parts, color='red', marker='x')
            plt.title('Eigenvalues in the Complex Plane')
            plt.xlabel('Real Part')
            plt.ylabel('Imaginary Part')
            plt.axhline(0, color='black', linewidth=0.5)
            plt.axvline(0, color='black', linewidth=0.5)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.close("all")
        eig_max = np.max(np.real(eigen_val))
        tau = -1.0 / eig_max
        H_inv = np.linalg.inv(H)
        H_inv_two = np.linalg.inv(np.linalg.matrix_power(H, 2))
        H_inv_three = np.linalg.inv(np.linalg.matrix_power(H, 3))
        ti_amend, Hi_amend_long, Hi_amend_short, ti_direct, Hi_direct, decay_matrix, path = [[] for i in range(7)]
        amp_list = []
        for area_idx in range(dim):
            G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
            try:
                d = nx.dijkstra_path_length(G, m, area_idx, weight=1)
                path.append(int(d))
            except:
                path.append(-1)
                print('not link')
            ti_direct.append(-H_inv_two[area_idx][m] / H_inv[area_idx][m])
            Hi_direct.append(strength * np.power(H_inv[area_idx][m], 2) / np.power(2 * H_inv_three[area_idx][m] * H_inv[area_idx][m] - np.power(H_inv_two[area_idx][m], 2), 0.5))
            ti_amend.append(np.real(ti_direct[area_idx] + 1 / np.max(eigen_val)))
            Hi_amend_long.append(Hi_direct[area_idx] / np.power(2 * math.pi, 0.5))
            Hi_amend_short.append(Hi_direct[area_idx] * np.power(d + 1, 0.5) * np.power(d, d) / (np.exp(d) * math.factorial(int(d))))
        for area_idx in range(dim):
            try:
                d = nx.dijkstra_path_length(G, m, area_idx, weight=1)
            except:
                d = -1
                print('not link')
            x_p = Hi_amend_short[area_idx]
            HI = -np.dot(H_inv, np.array(initial_v))
            decay_matrix.append((1 - 1 / np.exp(1)) * HI[area_idx] / x_p)
            amp_list.append(-H_inv[area_idx][m])
        return (ti_amend, Hi_amend_long, Hi_amend_short, ti_direct, Hi_direct, decay_matrix, path, amp_list)

def chain_simulation_dotting(N, p, weight, t_span, t_evel):
    simu_data, theory_data, matrix_list = [[] for i in range(3)]
    from scaling_stats import save_by_distance
    distances, amplitudes, peaks, times = [], [], [], []
    for trial in range(100):
        print('round:', trial)
        G, matrix = generate_graph_matrix('ER_random', N, {'probability': p})
        matrix = matrix.T
        matrix = weight * matrix
        diag_matrix = np.zeros((N, N))
        beta = 1.0
        for i in range(N):
            diag_matrix[i][i] = beta
        H = generate_H(matrix, diag_matrix)
        m = 0
        strength = 1
        initial_v = [0 if i != m else strength for i in range(N)]
        input_type = 'zero'
        params = {}
        I = generate_input(input_type, params, N)
        I_values = np.array([I(ti) for ti in t_evel])
        solver_euler = ODEEuler(H, I, initial_v)
        t_eval, solution_euler = solver_euler.solve(t_span, t_evel)
        simu_params = {'t': list(t_eval), 'y': list(solution_euler)}
        theory_params = {'H': list(H), 'strength': strength, 'm': m, 'matrix': list(matrix.T), 't': list(t_eval), 'y': list(solution_euler)}

        def make_array(value, default=0):
            return np.array(value) if isinstance(value, list) else value
        t = make_array(simu_params.get('t', 0))
        y = make_array(simu_params.get('y', 0))
        dim = N
        peak, peak_time, decay_time, amp = generate_simulation(input_type, simu_params, dim)
        ti_amend, Hi_amend_long, Hi_amend_short, ti_direct, Hi_direct, decay_matrix, path, amp_list = generate_theory(input_type, theory_params, dim)
        if 0:
            print('s peak:', np.min(peak), np.max(peak))
            print('s peak time:', np.min(peak_time), np.max(peak_time))
            print('s time:', np.min(decay_time), np.max(decay_time))
            print('t peak:', np.min(Hi_direct), np.max(Hi_direct))
            print('t peak time:', np.min(ti_direct), np.max(ti_direct))
            print('t time:', np.min(decay_matrix), np.max(decay_matrix))
        distances.append(path)
        amplitudes.append(amp)
        peaks.append(peak)
        times.append(peak_time)
        if trial < 10:
            simu_data.append([peak, peak_time, decay_time])
            theory_data.append([ti_amend, Hi_amend_long, Hi_amend_short, ti_direct, Hi_direct, decay_matrix, path])
            matrix_list.append(matrix)
    save_by_distance("pulse", distances, {"amplification": amplitudes, "peak_response": peaks, "time_constant": times})
    random_data_dot_save = {'simu': simu_data, 'theo': theory_data, 'matrix': matrix_list}
    with open('random_pulse_dot' + str(int(p * 100)) + '.pkl', 'wb') as file:
        pickle.dump(random_data_dot_save, file)

def dotted_plotting(theory_data, simu_data, G_list, p):
    t_initial_time, t_amend_time, s_time = [[] for i in range(3)]
    t_initial_amp, t_amend_amp, s_amp = [[] for i in range(3)]
    t_decay, s_decay = [[] for i in range(2)]
    for i in range(100):
        t_initial_amp += list(theory_data[0][4])
        t_amend_amp += list(theory_data[0][1])
        s_amp += list(simu_data[0][0])
        t_amend_time += list(theory_data[0][0])
        t_initial_time += list(theory_data[0][3])
        s_time += list(simu_data[0][1])
        t_decay += list(theory_data[0][5])
        s_decay += list(simu_data[0][2])
    print('s time:', s_time)
    print('t time:', t_initial_time)
    print('t amend time:', t_amend_time)
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
        errors = []
        for degree in all_degrees:
            values = [dist.get(degree, 0) for dist in distributions]
            avg_distribution[degree] = np.mean(values)
            errors.append((np.min(values), np.max(values)))
        total = sum(avg_distribution.values())
        avg_distribution = {k: v / total for k, v in avg_distribution.items()}
        errors = [(e[0] / total, e[1] / total) for e in errors]
        return (avg_distribution, errors)
    avg_in_distribution, in_errors = calculate_average_and_errors(in_degree_distributions)
    avg_out_distribution, out_errors = calculate_average_and_errors(out_degree_distributions)
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    degrees = sorted(avg_in_distribution.keys())
    avg_values = [avg_in_distribution[degree] for degree in degrees]
    in_error_values = [(avg_in_distribution[degree] - in_errors[i][0], in_errors[i][1] - avg_in_distribution[degree]) for i, degree in enumerate(degrees)]
    plt.bar(degrees, avg_values, yerr=np.array(in_error_values).T, capsize=5, color='gray')
    plt.xlabel('$indegree$', fontsize=15)
    plt.ylabel('$probability$', fontsize=15)
    plt.xticks(np.arange(1, np.max(degrees) + 1, 2), rotation=45)
    plt.subplot(1, 2, 2)
    degrees = sorted(avg_out_distribution.keys())
    avg_values = [avg_out_distribution[degree] for degree in degrees]
    out_error_values = [(avg_out_distribution[degree] - out_errors[i][0], out_errors[i][1] - avg_out_distribution[degree]) for i, degree in enumerate(degrees)]
    plt.bar(degrees, avg_values, yerr=np.array(out_error_values).T, capsize=5, color='gray')
    plt.xticks(np.arange(1, np.max(degrees) + 1, 2), rotation=45)
    plt.xlabel('$outdegree$', fontsize=15)
    plt.tight_layout()
    plt.close("all")
    amp_max = 100
    time_max = 100
    decay_max = 100
    print('initial:', len(t_initial_amp), len(t_amend_amp), len(s_amp))
    for i in reversed(range(len(t_initial_amp))):
        if any((x is None or math.isnan(x) or x > amp_max for x in [t_initial_amp[i], t_amend_amp[i], s_amp[i]])):
            del t_initial_amp[i]
            del t_amend_amp[i]
            del s_amp[i]
    print('after:', len(t_initial_amp), len(t_amend_amp), len(s_amp))
    plt.figure(figsize=(4, 4))
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('$theory$', fontsize=15)
    plt.ylabel('$simulation$', fontsize=15)
    a, b = (t_initial_amp, s_amp)
    modelone = LinearRegression()
    modelone.fit(np.array(a).reshape(-1, 1), np.array(b).reshape(-1, 1))
    predicts = modelone.predict(np.array(a).reshape(-1, 1))
    R2 = modelone.score(np.array(a).reshape(-1, 1), np.array(b).reshape(-1, 1))
    R2 = round(R2, 4)
    ax.scatter(a, b, color='red', alpha=0.5, label='$P_i^{pulse}$', s=20)
    print('direct coef_:%.3f' % modelone.coef_.item())
    print('direct r^2:%.3f' % R2)
    c, d = (t_amend_amp, s_amp)
    modeltwo = LinearRegression()
    modeltwo.fit(np.array(c).reshape(-1, 1), np.array(d).reshape(-1, 1))
    predicts = modeltwo.predict(np.array(c).reshape(-1, 1))
    R2 = modeltwo.score(np.array(c).reshape(-1, 1), np.array(d).reshape(-1, 1))
    R2 = round(R2, 4)
    ax.scatter(c, d, color='gray', alpha=0.5, label='$P_i^{\\prime(pulse)}$', s=20)
    print('amend coef_:%.3f' % modeltwo.coef_.item())
    print('amend r^2:%.3f' % R2)
    combined_list = list(a) + list(b) + list(c) + list(d)
    ax.plot(np.linspace(min(combined_list), max(combined_list), 1000), np.linspace(min(combined_list), max(combined_list), 1000), color='black', linewidth=2, linestyle='dashed', alpha=0.8)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.legend(frameon=False, prop={'size': 15})
    plt.tight_layout()
    plt.close("all")
    print('initial:', len(t_initial_time), len(t_amend_time), len(s_time))
    for i in reversed(range(len(t_initial_time))):
        if any((x is None or math.isnan(x) or x < 0 or (x > time_max) for x in [t_initial_time[i], t_amend_time[i], s_time[i]])):
            del t_initial_time[i]
            del t_amend_time[i]
            del s_time[i]
    print('after:', len(t_initial_time), len(t_amend_time), len(s_time))
    plt.figure(figsize=(4, 4))
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('$theory$', fontsize=15)
    plt.ylabel('$simulation$', fontsize=15)
    print(t_initial_time)
    a, b = (t_initial_time, s_time)
    modelone = LinearRegression()
    modelone.fit(np.array(a).reshape(-1, 1), np.array(b).reshape(-1, 1))
    predicts = modelone.predict(np.array(a).reshape(-1, 1))
    R2 = modelone.score(np.array(a).reshape(-1, 1), np.array(b).reshape(-1, 1))
    R2 = round(R2, 4)
    ax.scatter(a, b, color='red', alpha=0.5, label='$\\langle t\\rangle_i$', s=20)
    print('direct coef_:%.3f' % modelone.coef_.item())
    print('direct r^2:%.3f' % R2)
    c, d = (t_amend_time, s_time)
    modeltwo = LinearRegression()
    modeltwo.fit(np.array(c).reshape(-1, 1), np.array(d).reshape(-1, 1))
    predicts = modeltwo.predict(np.array(c).reshape(-1, 1))
    R2 = modeltwo.score(np.array(c).reshape(-1, 1), np.array(d).reshape(-1, 1))
    R2 = round(R2, 4)
    ax.scatter(c, d, color='gray', alpha=0.5, label='$\\langle t\\rangle^{\\prime}_i$', s=20)
    print('amend coef_:%.3f' % modeltwo.coef_.item())
    print('amend r^2:%.3f' % R2)
    combined_list = list(a) + list(b) + list(c) + list(d)
    ax.plot(np.linspace(min(combined_list), max(combined_list), 1000), np.linspace(min(combined_list), max(combined_list), 1000), color='black', linewidth=2, linestyle='dashed', alpha=0.8)
    plt.legend(frameon=False, prop={'size': 15})
    plt.tight_layout()
    plt.close("all")
    print('initial:', len(t_decay), len(s_decay))
    for i in reversed(range(len(t_decay))):
        if any((x is None or math.isnan(x) or x < 0 or (x > decay_max) for x in [t_decay[i], s_decay[i]])):
            del t_decay[i]
            del s_decay[i]
    print('after:', len(t_decay), len(s_decay))
    plt.figure(figsize=(4, 4))
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('$theory$', fontsize=15)
    plt.ylabel('$simulation$', fontsize=15)
    a, b = (t_decay, s_decay)
    modelone = LinearRegression()
    modelone.fit(np.array(a).reshape(-1, 1), np.array(b).reshape(-1, 1))
    predicts = modelone.predict(np.array(a).reshape(-1, 1))
    R2 = modelone.score(np.array(a).reshape(-1, 1), np.array(b).reshape(-1, 1))
    R2 = round(R2, 4)
    ax.scatter(a[::5], b[::5], color='gray', alpha=1.0, label='$\\tau_i^{decay}$', s=20)
    print('direct coef_:%.3f' % modelone.coef_.item())
    print('direct r^2:%.3f' % R2)
    combined_list = list(a) + list(b)
    ax.plot(np.linspace(min(combined_list), max(combined_list), 1000), np.linspace(min(combined_list), max(combined_list), 1000), color='black', linewidth=2, linestyle='dashed', alpha=0.8)
    plt.legend(frameon=False, prop={'size': 15})
    plt.tight_layout()
    plt.close("all")
    return (t_decay, s_decay, t_amend_time, s_time, t_amend_amp, s_amp)

if __name__ == "__main__":
    weight = 1.0
    t_span = [0, 100]
    t_evel = np.linspace(0, 100, 100000)
    N = 100
    p = 0.02
    chain_simulation_dotting(N, p, weight, t_span, t_evel)
    p = 0.02
    with open('random_pulse_dot' + str(int(p * 100)) + '.pkl', 'rb') as file:
        loaded_data_dot = pickle.load(file)
    theory_data = loaded_data_dot['theo']
    simu_data = loaded_data_dot['simu']
    G_list = loaded_data_dot['matrix']
    thr_decaytime, sim_decaytime, thr_peaktime, sim_peak_time, thr_peak, sim_peak = dotted_plotting(theory_data, simu_data, G_list, p)
    np.save('pic_two/pulse_decaytime_random_equal.npy', [thr_decaytime, sim_decaytime])
    np.save('pic_two/pulse_peaktime_random_equal.npy', [thr_peaktime, sim_peak_time])
    np.save('pic_two/pulse_peak_random_equal.npy', [thr_peak, sim_peak])
    print('save success!')
