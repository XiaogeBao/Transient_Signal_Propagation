import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import math
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
    elif graph_type == 'bi_chain':
        length = params.get('length', 1)
        G = nx.path_graph(length)
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

    def single_exp(x, a):
        return np.exp(-x / a)

    def make_array(value, default=0):
        return np.array(value) if isinstance(value, list) else np.full(dim, value)
    if input_type == 'pulse' or input_type == 'zero':

        def judge_stable(y_plot):
            stable = np.abs(y_plot[-1]) / np.max(y_plot)
            if stable < 0.001:
                return 1
            else:
                print(stable)
                return 0

        def truncate_list(a):
            return a[:next((i for i, value in enumerate(a) if value < a[0] * 0.01), len(a))]
        t = make_array(params.get('t', 0))
        y = make_array(params.get('y', 0))
        peak = np.max(y, axis=0)
        peak_time = [t[i] for i in np.argmax(y, axis=0)]
        decay_time, amp = [[] for i in range(2)]
        for i in np.arange(np.shape(y)[1]):
            y_plot = y[:, i]
            if judge_stable(y_plot):
                area = integrate.trapezoid(y_plot, t)
                amp.append(area)
                fit_curve = y_plot[np.argmax(y_plot):]
                fit_curve = (fit_curve - np.min(fit_curve)) / (np.max(fit_curve) - np.min(fit_curve))
                fit_curve = truncate_list(fit_curve)
                t_plot = t[:len(fit_curve)]
                r_single, _ = curve_fit(single_exp, t_plot, fit_curve)
                plt.plot(t_plot, fit_curve, color='gray', alpha=0.5)
                decay_time.append(r_single[0])
            else:
                print('time is not long enough')
        plt.title('fitting decay curve')
        plt.close("all")
        for i in np.arange(np.shape(y)[1]):
            y_plot = y[:, i]
            fit_curve = y_plot
            fit_curve = (fit_curve - np.min(fit_curve)) / (np.max(fit_curve) - np.min(fit_curve))
            t_plot = t[:len(fit_curve)]
            plt.plot(t_plot[:10000], fit_curve[:10000], color='gray', alpha=0.5)
        plt.title('initial curve')
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
        eig_max = np.max(np.real(eigen_val))
        tau = -1.0 / eig_max
        H_inv = np.linalg.inv(H)
        H_inv_two = np.linalg.inv(np.linalg.matrix_power(H, 2))
        H_inv_three = np.linalg.inv(np.linalg.matrix_power(H, 3))
        ti_amend, Hi_amend_long, Hi_amend_short, ti_direct, Hi_direct, path = [[] for i in range(6)]
        amp_list, decay_matrix = [[] for i in range(2)]
        for area_idx in range(dim):
            G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
            d = nx.dijkstra_path_length(G, m, area_idx, weight=1)
            path.append(d)
            ti_direct.append(-H_inv_two[area_idx][m] / H_inv[area_idx][m])
            Hi_direct.append(strength * np.power(H_inv[area_idx][m], 2) / np.power(2 * H_inv_three[area_idx][m] * H_inv[area_idx][m] - np.power(H_inv_two[area_idx][m], 2), 0.5))
            amp_list.append(-H_inv[area_idx][m])
            ti_amend.append(ti_direct[area_idx] + 1 / np.max(eigen_val))
            Hi_amend_long.append(Hi_direct[area_idx] / np.power(2 * math.pi, 0.5))
            Hi_amend_short.append(Hi_direct[area_idx] * np.power(d + 1, 0.5) * np.power(d, d) / (np.exp(d) * math.factorial(int(d))))
            x_p = Hi_amend_short[area_idx]
            HI = -np.dot(H_inv, np.array(initial_v))
            decay_matrix.append((1 - 1 / np.exp(1)) * HI[area_idx] / x_p)
        return (ti_direct, Hi_direct, ti_amend, Hi_amend_long, Hi_amend_short, decay_matrix, amp_list, path)

def generate_pic(input_type, simu_params, theory_params, dim):

    def make_array(value, default=0):
        return np.array(value) if isinstance(value, list) else value
    t = make_array(simu_params.get('t', 0))
    y = make_array(simu_params.get('y', 0))
    if input_type == 'pulse' or input_type == 'zero':
        peak, peak_time, decay_time, amp = generate_simulation(input_type, simu_params, dim)
        ti_direct, Hi_direct, ti_amend, Hi_amend_long, Hi_amend_short, decay_matrix, amp, path = generate_theory(input_type, theory_params, dim)
        np.save('scaling/pulse_amplification.npy', amp)
        np.save('scaling/pulse_peak_response.npy', peak)
        np.save('scaling/pulse_time_constant.npy', peak_time)

        pulse_peak = np.save('pulse_peak_direct_chain.npy', np.array(peak))
        pulse_amp = np.save('pulse_amp_direct_chain.npy', np.array(amp))
        plt.figure(figsize=(4, 3))
        ax = plt.axes()
        colors = ['#C6A855', '#C59323', '#C1571D', '#77171C']
        i = 0
        for i in np.arange(0, 4):
            if i == 0:
                plt.plot(t, y[:, i], label='input node', color=colors[3 - i])
            elif i == 1:
                plt.plot(t, y[:, i], label='other nodes', color=colors[3 - i])
            else:
                plt.plot(t, y[:, i], color=colors[3 - i])
            i += 1
        ax.set_xticks([])
        ax.set_yticks([1], ['$1$'], fontsize=15)
        ax.set_xlim([0, 4])
        ax.set_ylim([0, 1])
        plt.xlabel('$t$', fontsize=15)
        plt.ylabel('$\\Delta x_i^{pulse}$', fontsize=15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.close("all")
        plt.figure(figsize=(4, 3))
        ax = plt.axes()
        plt.plot(path[:10], peak_time[:10], label='$\\langle t\\rangle_i \\ sim.$', marker='o', color='black')
        plt.plot(path[:10], ti_direct[:10], label='$\\langle t\\rangle_i \\ theory$', linestyle='--', color='red', linewidth=3)
        plt.plot(path[:10], ti_amend[:10], label='$\\langle t\\rangle^{\\prime}_i \\ theory$', linestyle='--', color='black', linewidth=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('$distance \\ d$', fontsize=15)
        plt.ylabel('$time$', fontsize=15)
        plt.xticks(path[:10], np.arange(len(path[:10])))
        plt.legend(frameon=False, prop={'size': 12})
        plt.tight_layout()
        plt.close("all")
        plt.figure(figsize=(4, 3))
        ax = plt.axes()
        plt.plot(path[:10], peak_time[:10], label='$\\langle t\\rangle_i \\ sim.$', linestyle='--', color='black', linewidth=3)
        plt.plot(path[:10], ti_direct[:10], label='$\\langle t\\rangle_i \\ theory$', color='red', marker='o')
        plt.plot(path[:10], ti_amend[:10], label='$\\langle t\\rangle^{\\prime}_i \\ theory$', color='black', marker='s')
        plt.plot(path[:10], np.array(ti_direct[:10]) * (1 - np.log(np.exp(1) - 1)), label='$test \\ \\langle t\\rangle_i \\ theory$', color='blue')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('$distance \\ d$', fontsize=15)
        plt.ylabel('$time$', fontsize=15)
        plt.xticks(path[:10], np.arange(len(path[:10])))
        plt.legend(frameon=False, prop={'size': 12})
        plt.tight_layout()
        plt.close("all")
        plt.figure(figsize=(4, 3))
        ax = plt.axes()
        plt.plot(path[:10], peak_time[:10], label='$\\langle t\\rangle_i \\ sim.$', linestyle='--', color='black', linewidth=3)
        plt.plot(path[:10], ti_direct[:10], label='$\\langle t\\rangle_i \\ theory$', color='red', marker='o')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('$distance \\ d$', fontsize=15)
        plt.ylabel('$time$', fontsize=15)
        plt.xticks(path[:10], np.arange(len(path[:10])))
        plt.legend(frameon=False, prop={'size': 12})
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
        plt.plot(path[1:11], peak_time[1:11], label='$\\langle t\\rangle_i \\ sim.$', linestyle='--', color='black', linewidth=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('$distance \\ d$', fontsize=15)
        plt.ylabel('$peak \\ response \\ time$', fontsize=15)
        plt.xticks(path[1:11], np.arange(1, 1 + len(path[:10])))
        plt.legend(frameon=False, prop={'size': 12})
        plt.tight_layout()
        plt.close("all")
        slope, r_square = linear_regre(path[1:11], peak_time[1:11])
        print('time:', slope, r_square)
        print('time', 1 / (1 + 1))
        plt.figure(figsize=(4, 3))
        ax = plt.axes()
        plt.plot(path[:10], peak[:10], label='$P_i^{pulse} \\ sim. $', marker='o', color='black')
        plt.plot(path[:10], Hi_direct[:10], label='$P_i^{pulse} \\ theory$', color='red', linestyle='--', linewidth=3)
        plt.plot(path[:10], Hi_amend_short[:10], label='$P_i^{\\prime (pulse)} \\ theory$', color='black', linestyle='--', linewidth=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.yscale('log')
        plt.xlabel('$distance \\ d$', fontsize=15)
        plt.ylabel('$peak \\ response$', fontsize=15)
        plt.xticks(path[:10], np.arange(len(path[:10])))
        plt.legend(frameon=False, prop={'size': 12})
        plt.tight_layout()
        plt.close("all")
        plt.figure(figsize=(4, 4))
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('$theory$', fontsize=15)
        plt.ylabel('$simulation$', fontsize=15)
        a, b = (Hi_direct[0:], peak[0:])
        modelone = LinearRegression()
        modelone.fit(np.array(a).reshape(-1, 1), np.array(b).reshape(-1, 1))
        predicts = modelone.predict(np.array(a).reshape(-1, 1))
        R2 = modelone.score(np.array(a).reshape(-1, 1), np.array(b).reshape(-1, 1))
        R2 = round(R2, 4)
        ax.scatter(a, b, color='red', alpha=1, label='$P_i^{pulse}$', s=1)
        print('direct coef_:%.3f' % modelone.coef_.item())
        print('direct r^2:%.3f' % R2)
        c, d = (Hi_amend_long[0:], peak[0:])
        modeltwo = LinearRegression()
        modeltwo.fit(np.array(c).reshape(-1, 1), np.array(d).reshape(-1, 1))
        predicts = modeltwo.predict(np.array(c).reshape(-1, 1))
        R2 = modeltwo.score(np.array(c).reshape(-1, 1), np.array(d).reshape(-1, 1))
        R2 = round(R2, 4)
        ax.scatter(c, d, color='gray', alpha=1, label='$P_i^{\\prime(pulse)}$', s=1)
        print('amend coef_:%.3f' % modeltwo.coef_.item())
        print('amend r^2:%.3f' % R2)
        combined_list = list(a) + list(b) + list(c) + list(d)
        ax.plot(np.linspace(min(combined_list), max(combined_list), 1000), np.linspace(min(combined_list), max(combined_list), 1000), color='black', linewidth=2, linestyle='dashed', alpha=0.8)
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.legend(frameon=False, prop={'size': 15})
        plt.tight_layout()
        plt.close("all")
        plt.figure(figsize=(4, 4))
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('$theory$', fontsize=15)
        plt.ylabel('$simulation$', fontsize=15)
        a, b = (ti_direct, peak_time)
        modelone = LinearRegression()
        modelone.fit(np.array(a).reshape(-1, 1), np.array(b).reshape(-1, 1))
        predicts = modelone.predict(np.array(a).reshape(-1, 1))
        R2 = modelone.score(np.array(a).reshape(-1, 1), np.array(b).reshape(-1, 1))
        R2 = round(R2, 4)
        ax.scatter(a, b, color='red', alpha=1, label='$\\langle t\\rangle_i$', s=1)
        print('direct coef_:%.3f' % modelone.coef_.item())
        print('direct r^2:%.3f' % R2)
        c, d = (ti_amend, peak_time)
        modeltwo = LinearRegression()
        modeltwo.fit(np.array(c).reshape(-1, 1), np.array(d).reshape(-1, 1))
        predicts = modeltwo.predict(np.array(c).reshape(-1, 1))
        R2 = modeltwo.score(np.array(c).reshape(-1, 1), np.array(d).reshape(-1, 1))
        R2 = round(R2, 4)
        ax.scatter(c, d, color='gray', alpha=1, label='$\\langle t\\rangle^{\\prime}_i$', s=1)
        print('amend coef_:%.3f' % modeltwo.coef_.item())
        print('amend r^2:%.3f' % R2)
        combined_list = list(a) + list(b) + list(c) + list(d)
        combined_list = [value for value in combined_list if value > 1e-10]
        ax.plot(np.linspace(min(combined_list), max(combined_list), 1000), np.linspace(min(combined_list), max(combined_list), 1000), color='black', linewidth=2, linestyle='dashed', alpha=0.8)
        plt.tight_layout()
        plt.legend(frameon=False, prop={'size': 15})
        plt.close("all")
        plt.figure(figsize=(4, 3))
        ax = plt.axes()
        plt.plot(path[1:11], amp[1:11], label='$A_i^{pulse}$', linestyle='--', color='black', linewidth=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.yscale('log')
        plt.xlabel('$distance \\ d$', fontsize=15)
        plt.ylabel('$total \\ response \\ strength$', fontsize=15)
        plt.xticks(path[1:11], np.arange(1, 1 + len(path[:10])))
        plt.legend(frameon=False, prop={'size': 12})
        plt.tight_layout()
        plt.close("all")
        slope, r_square = linear_regre(path[1:11], np.log(amp[1:11]))
        print('amp:', slope, r_square)
        print('amp', -np.log(1 + 1 / 1))
        plt.figure(figsize=(4, 3))
        ax = plt.axes()
        plt.plot(path[1:11], peak[1:11], label='$P_i^{pulse} \\ sim. $', linestyle='--', color='black', linewidth=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.yscale('log')
        plt.xlabel('$distance \\ d$', fontsize=15)
        plt.ylabel('$peak \\ response$', fontsize=15)
        plt.xticks(path[1:11], np.arange(1, 1 + len(path[:10])))
        plt.legend(frameon=False, prop={'size': 12})
        plt.tight_layout()
        plt.close("all")
        slope, r_square = linear_regre(path[1:11], np.log(amp[1:11]))
        print('amp:', slope, r_square)
        print('amp', -np.log(1 + 1 / 1))
        plt.figure(figsize=(4, 3))
        ax = plt.axes()
        plt.plot(path[:10], decay_time[:10], label='$\\tau_i^{decay} sim.$', marker='o', color='black')
        plt.plot(path[:10], decay_matrix[:10], label='$\\tau_i^{decay} \\ theory$', color='black', linestyle='--', linewidth=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('$distance \\ d$', fontsize=15)
        plt.ylabel('$time$', fontsize=15)
        plt.xticks(path[:10], np.arange(len(path[:10])))
        plt.legend(frameon=False, prop={'size': 12})
        plt.tight_layout()
        plt.close("all")
        plt.figure(figsize=(4, 3))
        ax = plt.axes()
        plt.plot(path[:10], decay_time[:10], label='$\\tau_i^{decay} sim.$', linestyle='--', color='black', linewidth=3)
        plt.plot(path[:10], np.array(decay_matrix[:10]) / (1.0 - 1.0 / np.exp(1)), label='$\\tau_i^{decay} \\ test$', color='black', marker='o')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('$distance \\ d$', fontsize=15)
        plt.ylabel('$time$', fontsize=15)
        plt.xticks(path[:10], np.arange(len(path[:10])))
        plt.legend(frameon=False, prop={'size': 12})
        plt.tight_layout()
        plt.close("all")
        plt.figure(figsize=(4, 4))
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('$theory$', fontsize=15)
        plt.ylabel('$simulation$', fontsize=15)
        a, b = (decay_time[:15], decay_matrix[:15])
        modelone = LinearRegression()
        modelone.fit(np.array(a).reshape(-1, 1), np.array(b).reshape(-1, 1))
        predicts = modelone.predict(np.array(a).reshape(-1, 1))
        R2 = modelone.score(np.array(a).reshape(-1, 1), np.array(b).reshape(-1, 1))
        R2 = round(R2, 4)
        ax.scatter(a, b, color='gray', alpha=1.0, label='$\\tau_i^{decay}$', s=20)
        print('decay coef_:%.3f' % modelone.coef_.item())
        print('decay r^2:%.3f' % R2)
        print('decay intercept:%.3f' % modelone.intercept_.item())
        combined_list = list(a) + list(b)
        ax.plot(np.linspace(min(combined_list), max(combined_list), 1000), np.linspace(min(combined_list), max(combined_list), 1000), color='black', linewidth=2, linestyle='dashed', alpha=0.8)
        plt.legend(frameon=False, prop={'size': 15})
        plt.tight_layout()
        plt.close("all")
        plt.figure(figsize=(4, 3))
        ax = plt.axes()
        colors = ['#FF4500', '#FF6347', '#FF7F50', '#FFA07A', '#FFDAB9']
        for i, color in zip(np.arange(1, 6), colors):
            plt.plot(t, y[:, i], label='d=' + str(i), color=color)
        plt.legend(frameon=False, prop={'size': 12})
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('$t$', fontsize=15)
        plt.ylabel('$\\Delta x_i^{pulse}$', fontsize=15)
        plt.xlim([0, 6])
        plt.tight_layout()
        plt.close("all")
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
        plt.ylabel('$\\Delta x_i^{pulse}$', fontsize=15)
        plt.legend(frameon=False, prop={'size': 12})
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlim([0, 3])
        plt.tight_layout()
        plt.close("all")
        plt.figure(figsize=(6, 5))
        ax = plt.axes()
        plt.plot(path[1:], (np.abs(np.array(ti_amend) - np.array(peak_time)) / np.array(peak_time))[1:], marker='s')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('$distance \\ d$')
        plt.ylabel('$time \\ error$')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend(frameon=False)
        plt.close("all")
        time_error = (np.abs(np.array(ti_amend) - np.array(peak_time)) / np.array(peak_time))[1:]
        plt.figure(figsize=(6, 5))
        ax = plt.axes()
        plt.plot(path[1:], (np.array(Hi_direct) / np.array(peak))[1:], marker='s')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('$distance \\ d$')
        plt.ylabel('$amplitude \\ error$')
        plt.xscale('log')
        plt.legend(frameon=False)
        plt.close("all")
        amp_error = (np.array(Hi_direct) / np.array(peak))[1:]
        np.save('y_pulse.npy', np.array(y))
        return (time_error, amp_error, path[1:], decay_time, decay_matrix, ti_amend, peak_time, Hi_amend_long, peak)

def chain_simulation(N, weight, t_span, t_evel):
    G, matrix = generate_graph_matrix('directed_chain', N, {'length': N})
    plot_directed_weighted_graph(G)
    matrix = matrix.T
    matrix = weight * matrix
    plt.imshow(matrix)
    plt.close("all")
    diag_matrix = np.zeros((N, N))
    beta = 1.0
    for i in range(N):
        diag_matrix[i][i] = beta
    H = generate_H(matrix, diag_matrix)
    print(H)
    plt.imshow(H)
    plt.close("all")
    m = 0
    strength = 1
    initial_v = [0 if i != m else strength for i in range(N)]
    input_type = 'zero'
    params = {}
    I = generate_input(input_type, params, N)
    I_values = np.array([I(ti) for ti in t_evel])
    solver_euler = ODEEuler(H, I, initial_v)
    t_eval, solution_euler = solver_euler.solve(t_span, t_evel)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), gridspec_kw={'height_ratios': [1, 3]})
    y_plot = list([0 for i in range(100)]) + list(solution_euler[:, 1])
    I_plot = [0 if i != 101 else 1 for i in range(len(y_plot))]
    ax1.plot(I_plot[:5000], color='black', linewidth=3.0)
    ax1.set_ylabel('Input')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis('off')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.plot(y_plot[:5000], color='black', linewidth=3.0)
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
    y_plot = list([0 for i in range(100)]) + list(solution_euler[:, 1])
    I_plot = [0 if i != 101 else 1 for i in range(len(y_plot))]
    t_plot = np.arange(len(y_plot[:5000]))
    ax2.plot(t_plot[:np.argmax(y_plot[:5000])], y_plot[:5000][:np.argmax(y_plot[:5000])], color='black', linewidth=3.0)
    ax2.plot(t_plot[np.argmax(y_plot[:5000]):], y_plot[:5000][np.argmax(y_plot[:5000]):], color='#B41D23', linewidth=3.0)
    plt.legend(frameon=False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.fill_between(np.arange(len(y_plot[:5000])), y_plot[:5000], 0, color='#C5D2E8', alpha=1.0)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlim([0, len(y_plot[:5200])])
    ax2.set_ylim([0, 1.2 * np.max(y_plot[:5000])])
    ax2.plot([np.argmax(y_plot[:5000]), np.argmax(y_plot[:5000])], [0, np.max(y_plot)], color='#B41D23', linestyle=':')
    ax2.plot([0, np.argmax(y_plot[:5000])], [np.max(y_plot), np.max(y_plot)], color='#406CB4', linestyle=':')
    ax2.annotate('', xy=(len(y_plot[:5200]), 0), xytext=(0, 0), arrowprops=dict(facecolor='black', arrowstyle='->'))
    ax2.annotate('', xy=(0, 1.2 * np.max(y_plot[:5000])), xytext=(0, 0), arrowprops=dict(facecolor='black', arrowstyle='->'))
    ax2.text(len(y_plot[:5000]) + 300, 0, '$t$', fontsize=15, verticalalignment='center', horizontalalignment='left')
    ax2.text(0, 1.2 * np.max(y_plot[:5000]) + 0.02, '$\\Delta x_i^{pulse}$', fontsize=15, verticalalignment='bottom', horizontalalignment='left')
    plt.tight_layout()
    plt.close("all")
    fig, ax2 = plt.subplots(1, 1, figsize=(4, 3))
    y_plot = list(solution_euler[:, 0])
    I_plot = [0 if i != 101 else 1 for i in range(len(y_plot))]
    t_plot = np.arange(len(y_plot[:5000]))
    ax2.plot(t_plot[np.argmax(y_plot[:5000]):], y_plot[:5000][np.argmax(y_plot[:5000]):], color='#B41D23', linewidth=3.0)
    plt.legend(frameon=False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.fill_between(np.arange(len(y_plot[:5000])), y_plot[:5000], 0, color='#CACEE5', alpha=1.0)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlim([0, len(y_plot[:5200])])
    ax2.set_ylim([0, 1.2 * np.max(y_plot[:5000])])
    ax2.annotate('', xy=(len(y_plot[:5200]), 0), xytext=(0, 0), arrowprops=dict(facecolor='black', arrowstyle='->'))
    ax2.annotate('', xy=(0, 1.2 * np.max(y_plot[:5000])), xytext=(0, 0), arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.tight_layout()
    plt.close("all")
    simu_params = {'t': list(t_eval), 'y': list(solution_euler)}
    theory_params = {'H': list(H), 'strength': strength, 'm': m, 'matrix': list(matrix.T)}
    time_error, amp_error, path, thr_decaytime, sim_decaytime, thr_peaktime, sim_peak_time, thr_peak, sim_peak = generate_pic(input_type, simu_params, theory_params, N)
    np.save('pic_two/pulse_decaytime_directedchain_equal.npy', [thr_decaytime, sim_decaytime])
    np.save('pic_two/pulse_peaktime_directedchain_equal.npy', [thr_peaktime, sim_peak_time])
    np.save('pic_two/pulse_peak_directedchain_equal.npy', [thr_peak, sim_peak])
    print('save success!')

if __name__ == "__main__":
    weight = 1.0
    t_span = [0, 100]
    t_evel = np.linspace(0, 100, 100000)
    N = 80
    chain_simulation(N, weight, t_span, t_evel)
