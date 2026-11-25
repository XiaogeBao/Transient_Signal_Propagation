import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.linear_model import LinearRegression
import matplotlib.colors as colors
from scipy import integrate
from sklearn.metrics import r2_score

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
    return (G, nx.to_numpy_array(G))

def plot_directed_weighted_graph(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=30, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=5, edge_color='darkblue', width=1)
    plt.axis('off')
    plt.close("all")

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
                plt.plot(y_plot)
                area = integrate.trapezoid(y_plot, t)
                amp.append(area)
            else:
                print('time is not long enough')
        plt.title('evolution')
        plt.close("all")
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

def generate_pic(input_type, simu_params, theory_params, dim):

    def make_array(value, default=0):
        return np.array(value) if isinstance(value, list) else value
    t = make_array(simu_params.get('t', 0))
    y = make_array(simu_params.get('y', 0))
    if input_type == 'square':
        peak, amp = generate_simulation(input_type, simu_params, dim)
        peak_theory, amp_theory, path = generate_theory(input_type, theory_params, dim)
        plt.figure(figsize=(4, 3))
        ax = plt.axes()
        colors = ['#FF4500', '#FF6347', '#FF7F50', '#FFA07A', '#FFDAB9']
        for i, color in zip(np.arange(1, 6), colors):
            plt.plot(t, y[:, i], label='d=' + str(i), color=color)
        plt.legend(frameon=False, prop={'size': 12})
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('$t$', fontsize=15)
        plt.ylabel('$\\Delta x_i^{square}$', fontsize=15)
        plt.tight_layout()
        plt.close("all")

        def linear_regre(x, y):
            x = np.array(x)
            x_reshaped = x.reshape(-1, 1)
            model = LinearRegression()
            model.fit(x_reshaped, y)
            y_pred = model.predict(x_reshaped)
            slope = model.coef_[0]
            r_squared = r2_score(y, y_pred)
            return (slope, r_squared)
        plt.figure(figsize=(4, 3))
        ax = plt.axes()
        plt.plot(path[1:11], peak[1:11], label='$P_i^{square}(t_s) \\ sim. $', linestyle='--', color='blue', linewidth=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.yscale('log')
        plt.xlabel('$distance \\ d$', fontsize=15)
        plt.ylabel('$peak \\ response$', fontsize=15)
        plt.xticks(path[1:11], np.arange(1, 1 + len(path[:10])))
        plt.legend(frameon=False, prop={'size': 12})
        plt.tight_layout()
        plt.close("all")
        slope, r_square = linear_regre(path[1:11], np.log(peak[1:11]))
        print('peak:', slope, r_square)
        print('peak', -np.log(1 + 1 / 1))
        plt.figure(figsize=(4, 3))
        ax = plt.axes()
        plt.plot(path[1:11], amp[1:11], label='$A_i^{square}(t_s) \\ sim. $', linestyle='--', color='blue', linewidth=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.yscale('log')
        plt.xlabel('$distance \\ d$', fontsize=15)
        plt.ylabel('$response \\ amplification$', fontsize=15)
        plt.xticks(path[1:11], np.arange(1, 1 + len(path[:10])))
        plt.legend(frameon=False, prop={'size': 12})
        plt.tight_layout()
        plt.close("all")
        slope, r_square = linear_regre(path[1:11], np.log(amp[1:11]))
        print('peak:', slope, r_square)
        print('peak', -np.log(1 + 1 / 1))
        plt.figure(figsize=(4, 3))
        ax = plt.axes()
        for i, color in zip(np.arange(0, 10), colors):
            if i == 0:
                plt.plot(t, y[:, i], label='input node', color='#B41D23', linewidth=3)
            elif i == 1:
                plt.plot(t, y[:, i], label='other nodes', color='gray', linewidth=3)
            else:
                plt.plot(t, y[:, i], color='gray', linewidth=3)
        plt.xlabel('$t$', fontsize=15)
        plt.ylabel('$\\Delta x_i^{square}$', fontsize=15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlim([0, 20])
        plt.tight_layout()
        plt.close("all")
        return (amp, amp_theory, peak, peak_theory, path)

def find_first_decrease(y):
    for i in range(len(y) - 1):
        if y[i] > y[i + 1]:
            return i
    return None

def chain_square(N, weight, t_span, t_evel, start_time, end_time):
    G, matrix = generate_graph_matrix('directed_chain', N, {'length': N})
    plot_directed_weighted_graph(G)
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
    input_type = 'square'
    params = {'s': [0 if i != 0 else strength for i in range(N)], 'start_time': start_time, 'end_time': end_time}
    ts = end_time - start_time
    I = generate_input(input_type, params, N)
    I_values = np.array([I(ti) for ti in t_evel])
    solver_euler = ODEEuler(H, I, initial_v)
    t_eval, solution_euler = solver_euler.solve(t_span, t_evel)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), gridspec_kw={'height_ratios': [1, 3]})
    y_plot = list(solution_euler[:, 1])
    I_plot = I_values[:, 0]
    ax1.plot(I_plot, color='black', linewidth=3.0)
    ax1.set_ylabel('Input')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis('off')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.plot(y_plot, color='black', linewidth=3.0)
    ax2.set_ylabel('Output')
    plt.legend(frameon=False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.axis('off')
    plt.tight_layout()
    plt.close("all")
    fig, ax2 = plt.subplots(1, 1, figsize=(4, 3))
    y_plot = list(solution_euler[:, 1])[:23000]
    print('length:', len(y_plot))
    I_plot = I_values[:, 0]
    t_plot = np.arange(len(y_plot))
    ax2.plot(t_plot, y_plot, color='#B41D23', linewidth=3.0)
    plt.legend(frameon=False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.fill_between(t_plot, y_plot, 0, color='#CACEE5', alpha=1.0)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlim([0, len(y_plot) + 8000])
    ax2.set_ylim([0, 1.2 * np.max(y_plot)])
    y_element = find_first_decrease(y_plot)
    max_y = y_plot[y_element]
    ax2.plot([0, y_element], [max_y, max_y], color='#406CB4', linestyle=':')
    ax2.plot([y_element, y_element], [0, max_y], color='black', linestyle=':')
    ax2.annotate('', xy=(len(y_plot) + 5000, 0), xytext=(0, 0), arrowprops=dict(facecolor='black', arrowstyle='->'))
    ax2.annotate('', xy=(0, 1.2 * np.max(y_plot)), xytext=(0, 0), arrowprops=dict(facecolor='black', arrowstyle='->'))
    ax2.text(len(y_plot) + 6000, 0, '$t$', fontsize=15, verticalalignment='center', horizontalalignment='left')
    ax2.text(0, 1.2 * np.max(y_plot) + 0.03, '$\\Delta x_i^{square}$', fontsize=15, verticalalignment='bottom', horizontalalignment='left')
    plt.tight_layout()
    plt.close("all")
    simu_params = {'t': list(t_eval), 'y': list(solution_euler), 'eta': eta, 'ts': ts}
    theory_params = {'H': list(H), 'strength': strength, 'm': m, 'matrix': list(matrix.T), 'eta': eta, 'ts': ts}
    amp, amp_theory, peak, peak_theory, path = generate_pic(input_type, simu_params, theory_params, N)
    np.save('pic_two/square_peak_directedchain_equal.npy', [peak_theory, peak])
    np.save('scaling/square_amplification.npy', amp)
    np.save('scaling/square_peak_response.npy', peak)

    print('save success!')
    return (amp, amp_theory, peak, peak_theory, path)

def chain_regular(N, weight, t_span, t_evel, start_time, end_time):
    net_name = 'regular'
    G, matrix = generate_graph_matrix(net_name, N, {'nearest_neighbors': 4})
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
    input_type = 'square'
    params = {'s': [0 if i != 0 else strength for i in range(N)], 'start_time': start_time, 'end_time': end_time}
    ts = end_time - start_time
    I = generate_input(input_type, params, N)
    I_values = np.array([I(ti) for ti in t_evel])
    solver_euler = ODEEuler(H, I, initial_v)
    t_eval, solution_euler = solver_euler.solve(t_span, t_evel)
    simu_params = {'t': list(t_eval), 'y': list(solution_euler), 'eta': eta, 'ts': ts}
    theory_params = {'H': list(H), 'strength': strength, 'm': m, 'matrix': list(matrix.T), 'eta': eta, 'ts': ts}
    amp, amp_theory, peak, peak_theory, path = generate_pic(input_type, simu_params, theory_params, N)
    np.save('pic_two/square_peak_regular_equal.npy', [peak_theory, peak])
    print('save success!')

if __name__ == "__main__":
    weight = 1.0
    t_span = [0, 150]
    t_evel = np.linspace(0, 150, 200000)
    N = 100
    start_time = 0
    end_time = 10
    chain_square(N, weight, t_span, t_evel, start_time, end_time)
    weight = 1.0
    t_span = [0, 150]
    t_evel = np.linspace(0, 150, 200000)
    N = 100
    start_time = 0
    end_time = 10
    chain_regular(N, weight, t_span, t_evel, start_time, end_time)
