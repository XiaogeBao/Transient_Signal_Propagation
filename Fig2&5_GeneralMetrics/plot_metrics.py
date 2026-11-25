from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from PIL import Image
from io import BytesIO

DATA_DIR = Path("pic_two")

METRICS = {
    "constant": ("exptime", "exptimeconstant"),
    "pulse": ("decaytime", "peaktime", "peak"),
    "square": ("peak",),
    "noiseauto": ("exptime", "exptimeconstant"),
    "noisecov": ("peaktime", "peak"),
}

def topologies(panel):
    chain = "undirectedchain" if panel in ("noiseauto", "noisecov") else "directedchain"
    random = "ER_random" if panel == "noisecov" else "random"
    return (chain, "regular", random, "small-world", "scale-free", "geometric")

def array_path(panel, metric, topology):
    return DATA_DIR / f"{panel}_{metric}_{topology}_equal.npy"

def load_pair(panel, metric, topology):
    return np.load(array_path(panel, metric, topology), allow_pickle=True)

def rank_correlation(A, B):
    return spearmanr(A, B)[0]

def combined_bar_plot():
    net_name_list = topologies("constant")
    
    means_sigtime = []
    std_sigtime = []
    means_sigtimeconstant = []
    std_sigtimeconstant = []
    rank_corr_sigtime = []
    rank_corr_sigtimeconstant = []
    
    gap = 3  # Adjust this value to increase or decrease the gap
    
    for net_name in net_name_list:
        loaded_lists_sigtime = load_pair("constant", "exptime", net_name)
        thr_loaded_sigtime = loaded_lists_sigtime[0]
        sim_loaded_sigtime = loaded_lists_sigtime[1]
        relative_error_sigtime = np.abs(sim_loaded_sigtime - thr_loaded_sigtime) / sim_loaded_sigtime
        
        loaded_lists_sigtimeconstant = load_pair("constant", "exptimeconstant", net_name)
        thr_loaded_sigtimeconstant = loaded_lists_sigtimeconstant[0]
        sim_loaded_sigtimeconstant = loaded_lists_sigtimeconstant[1]
        relative_error_sigtimeconstant = np.abs(sim_loaded_sigtimeconstant - thr_loaded_sigtimeconstant) / sim_loaded_sigtimeconstant
        
        means_sigtime.append(np.mean(relative_error_sigtime))
        std_sigtime.append(np.std(relative_error_sigtime))
        
        means_sigtimeconstant.append(np.mean(relative_error_sigtimeconstant))
        std_sigtimeconstant.append(np.std(relative_error_sigtimeconstant))

        corr_sigtime = rank_correlation(thr_loaded_sigtime, sim_loaded_sigtime)
        corr_sigtimeconstant = rank_correlation(thr_loaded_sigtimeconstant, sim_loaded_sigtimeconstant)
        rank_corr_sigtime.append(corr_sigtime)
        rank_corr_sigtimeconstant.append(corr_sigtimeconstant)

    num_networks = len(net_name_list)
    positions_sigtime = np.arange(0, num_networks * (2 + gap), (2 + gap))  # Positions for sigtime
    positions_sigtimeconstant = positions_sigtime + 1  # Positions for sigtimeconstant next to sigtime
    
    plt.figure(figsize=(12, 3))  # Adjust the figure size if needed
    
    plt.bar(positions_sigtime, means_sigtime, yerr=std_sigtime, capsize=5, width=0.8, label=r'$resp. \ time$', color='lightcoral')

    plt.bar(positions_sigtimeconstant, means_sigtimeconstant, yerr=std_sigtimeconstant, capsize=5, width=0.8, label=r'$time \ const.$', color='darkred')

    for i in range(num_networks):
        plt.text(positions_sigtime[i], means_sigtime[i] + std_sigtime[i] + 0.02, f"{rank_corr_sigtime[i]:.2f}", ha='center', va='bottom', fontsize=12, color='black')
        plt.text(positions_sigtimeconstant[i], means_sigtimeconstant[i] + std_sigtimeconstant[i] + 0.02, f"{rank_corr_sigtimeconstant[i]:.2f}", ha='center', va='bottom', fontsize=12,color='black')
        
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)

    plt.xticks([])
    plt.yticks(fontsize = 20)
    plt.ylim([1e-2, 1e0])
    plt.yscale('log')
    
    plt.legend(fontsize = 20, loc = 'upper left', frameon=False)

    plt.tight_layout()

    return plt.gcf()

def pulse_panel_combined():
    net_name_list = topologies("pulse")
    
    means_ratio1 = []
    std_ratio1 = []
    means_ratio2 = []
    std_ratio2 = []
    means_ratio3 = []
    std_ratio3 = []
    rank_corr_ratio1 = []
    rank_corr_ratio2 = []
    rank_corr_ratio3 = []

    gap = 3  # Adjust this value to increase or decrease the gap
    
    for net_name in net_name_list:
        if net_name == 'directedchain':
            loaded_lists_ratio1 = load_pair("pulse", "decaytime", net_name)
            thr_loaded_ratio1 = loaded_lists_ratio1[0][:15]
            sim_loaded_ratio1 = loaded_lists_ratio1[1][:15] 
            relative_error_ratio1 = np.abs(sim_loaded_ratio1 - thr_loaded_ratio1) / sim_loaded_ratio1

            loaded_lists_ratio2 = load_pair("pulse", "peaktime", net_name)
            thr_loaded_ratio2 = loaded_lists_ratio2[0][1:]
            sim_loaded_ratio2 = loaded_lists_ratio2[1][1:]
            relative_error_ratio2 = np.abs(sim_loaded_ratio2 - thr_loaded_ratio2) / sim_loaded_ratio2
        else:
            loaded_lists_ratio1 = load_pair("pulse", "decaytime", net_name)
            thr_loaded_ratio1 = loaded_lists_ratio1[0]
            sim_loaded_ratio1 = loaded_lists_ratio1[1]
            relative_error_ratio1 = np.abs(sim_loaded_ratio1 - thr_loaded_ratio1) / sim_loaded_ratio1

            loaded_lists_ratio2 = load_pair("pulse", "peaktime", net_name)
            thr_loaded_ratio2 = loaded_lists_ratio2[0]
            sim_loaded_ratio2 = loaded_lists_ratio2[1]
            relative_error_ratio2 = np.abs(sim_loaded_ratio2 - thr_loaded_ratio2) / sim_loaded_ratio2

        loaded_lists_ratio3 = load_pair("pulse", "peak", net_name)
        thr_loaded_ratio3 = loaded_lists_ratio3[0]
        sim_loaded_ratio3 = loaded_lists_ratio3[1]
        relative_error_ratio3 = thr_loaded_ratio3 / sim_loaded_ratio3

        means_ratio1.append(np.mean(relative_error_ratio1))
        std_ratio1.append(np.std(relative_error_ratio1))
        
        means_ratio2.append(np.mean(relative_error_ratio2))
        std_ratio2.append(np.std(relative_error_ratio2))
        
        means_ratio3.append(np.mean(relative_error_ratio3))
        std_ratio3.append(np.std(relative_error_ratio3))

        corr_ratio1 = rank_correlation(thr_loaded_ratio1, sim_loaded_ratio1)
        corr_ratio2 = rank_correlation(thr_loaded_ratio2, sim_loaded_ratio2)
        corr_ratio3 = rank_correlation(thr_loaded_ratio3, sim_loaded_ratio3)
        rank_corr_ratio1.append(corr_ratio1)
        rank_corr_ratio2.append(corr_ratio2)
        rank_corr_ratio3.append(corr_ratio3)
    
    num_networks = len(net_name_list)
    positions_ratio2 = np.arange(0, num_networks * (3 + gap), (3 + gap))  # Positions for third ratio
    positions_ratio1 = positions_ratio2 + 1  # Positions for second ratio
    positions_ratio3 = positions_ratio2 + 2  # Positions for first ratio

    fig, ax1 = plt.subplots(figsize=(12, 3))  # Adjust the figure size if needed

    ax1.bar(positions_ratio2, means_ratio2, yerr=std_ratio2, capsize=5, width=0.8, label=r'$resp. \ time$', color='#F2CCDB')
    ax1.bar(positions_ratio1, means_ratio1, yerr=std_ratio1, capsize=5, width=0.8, label=r'$decay \ time \ const.$', color='#DB6598')

    for i in range(num_networks):
        offset = 0.0  # Base offset
        if abs(means_ratio1[i] - means_ratio2[i]) < 0.1:  # If values are too close, increase the offset
            offset += 0.02  # Increase the offset to separate the text annotations

        ax1.text(positions_ratio1[i], means_ratio1[i] + std_ratio1[i] + offset, f"{rank_corr_ratio1[i]:.2f}", 
                ha='center', va='bottom', fontsize=12, color='black')
        
        ax1.text(positions_ratio2[i], means_ratio2[i] + std_ratio2[i] + offset + 0.02, f"{rank_corr_ratio2[i]:.2f}", 
                ha='center', va='bottom', fontsize=12, color='black')


    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(True)
    ax1.spines['left'].set_visible(False)
    ax1.set_yscale('log')
    ax1.set_ylim([1e-2, 10**1])
    ax1.tick_params(axis='y', labelsize=20)

    ax2 = ax1.twinx()
    
    ax2.bar(positions_ratio3, means_ratio3, yerr=std_ratio3, capsize=5, width=0.8, label=r'$peak \ resp.$', color='#C84783')


    for i in range(num_networks):
        plt.text(positions_ratio3[i], means_ratio3[i] + std_ratio3[i] + 0.01, f"{rank_corr_ratio3[i]:.2f}", 
                 ha='center', va='bottom', fontsize=12, color='black')
        
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(True)
    ax2.spines['bottom'].set_visible(True)
    ax2.set_ylim([0, 2.0])  # Adjust the upper limit as needed
    ax2.set_xticks([])
    ax2.tick_params(axis='y', labelsize=20)

    ax1.legend(loc='upper left', fontsize=20, frameon=False)
    ax2.legend(loc='upper right', fontsize=20, frameon=False)
    
    plt.tight_layout()

    return plt.gcf()

def square_panel(square_var):
    net_name_list = topologies("square")

    means_square = []
    std_square = []
    rank_corr_square = []

    for net_name in net_name_list:
        loaded_lists = load_pair("square", square_var, net_name)
        thr_loaded = loaded_lists[0]
        sim_loaded = loaded_lists[1]
        thr_list = thr_loaded.flatten().tolist()
        sim_list = sim_loaded.flatten().tolist()

        relative_error = np.array(thr_list) / np.array(sim_list)

        nonzero_relative_error = [x for x in relative_error if x > 1e-8]

        means_square.append(np.mean(nonzero_relative_error))
        std_square.append(np.std(nonzero_relative_error))

        corr_square = rank_correlation(thr_list, sim_list)
        rank_corr_square.append(corr_square)


    positions_square = np.arange(len(net_name_list))

    plt.figure(figsize=(12, 3))  # Adjust the figure size if needed

    plt.bar(positions_square, means_square, yerr=std_square, capsize=3, width=0.3, color='#1D4592', alpha = 0.8, label=r'$peak \ resp.$')

    for i in range(len(net_name_list)):
        plt.text(positions_square[i], means_square[i] + std_square[i] + 0.05, f"{rank_corr_square[i]:.2f}", 
                 ha='center', va='bottom', fontsize=12, color='black')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)

    plt.xticks(positions_square, net_name_list, fontsize=20)
    plt.yticks(fontsize=20)

    plt.gca().yaxis.set_label_position('right')
    plt.gca().yaxis.tick_right()

    plt.ylim(0, 1.5)  # Adjust based on your data range

    plt.legend(fontsize=20, frameon=False, loc= 'upper right')
    plt.xticks([])
    plt.yticks(np.round(np.linspace(0, 1.6, 5),1), fontsize=20) 
    plt.tight_layout()

    return plt.gcf()

def auto_panel_combined():
    net_name_list = topologies("noiseauto")
    
    means_sigtime = []
    std_sigtime = []
    means_sigtimeconstant = []
    std_sigtimeconstant = []
    rank_corr_sigtime = []
    rank_corr_sigtimeconstant = []

    gap = 3  # Adjust this value to increase or decrease the gap
    
    for net_name in net_name_list:
        loaded_lists_sigtime = load_pair("noiseauto", "exptime", net_name)
        
        if net_name == 'undirectedchain':
            thr_loaded_sigtime = loaded_lists_sigtime[0][:15]
            sim_loaded_sigtime = loaded_lists_sigtime[1][:15]
        else:
            thr_loaded_sigtime = loaded_lists_sigtime[0]
            sim_loaded_sigtime = loaded_lists_sigtime[1]
        
        relative_error_sigtime = np.abs(sim_loaded_sigtime - thr_loaded_sigtime) / sim_loaded_sigtime
        
        loaded_lists_sigtimeconstant = load_pair("noiseauto", "exptimeconstant", net_name)
        
        if net_name == 'undirectedchain':
            thr_loaded_sigtimeconstant = loaded_lists_sigtimeconstant[0][:15]
            sim_loaded_sigtimeconstant = loaded_lists_sigtimeconstant[1][:15]
        else:
            thr_loaded_sigtimeconstant = loaded_lists_sigtimeconstant[0]
            sim_loaded_sigtimeconstant = loaded_lists_sigtimeconstant[1]
        
        relative_error_sigtimeconstant = np.abs(sim_loaded_sigtimeconstant - thr_loaded_sigtimeconstant) / sim_loaded_sigtimeconstant

        means_sigtime.append(np.mean(relative_error_sigtime))
        std_sigtime.append(np.std(relative_error_sigtime))
        
        means_sigtimeconstant.append(np.mean(relative_error_sigtimeconstant))
        std_sigtimeconstant.append(np.std(relative_error_sigtimeconstant))
        
        corr_sigtime = rank_correlation(thr_loaded_sigtime, sim_loaded_sigtime)
        corr_sigtimeconstant = rank_correlation(thr_loaded_sigtimeconstant, sim_loaded_sigtimeconstant)
        rank_corr_sigtime.append(corr_sigtime)
        rank_corr_sigtimeconstant.append(corr_sigtimeconstant)

    num_networks = len(net_name_list)
    positions_sigtimeconstant = np.arange(num_networks)  # Positions for sigtimeconstant

    plt.figure(figsize=(12, 3))  # Adjust the figure size if needed
    
    plt.bar(positions_sigtimeconstant, means_sigtimeconstant, yerr=std_sigtimeconstant, capsize=3, width=0.3, label=r'$time \ const.$', color='#569F27')

    for i in range(len(net_name_list)):
        plt.text(positions_sigtimeconstant[i], means_sigtimeconstant[i] + std_sigtimeconstant[i] + 0.05, f"{rank_corr_sigtimeconstant[i]:.2f}", 
                 ha='center', va='bottom', fontsize=12, color='black')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)

    plt.xticks([])
    plt.yticks(fontsize=20)

    plt.ylim([1e-3, 1e1])
    plt.yscale('log')

    plt.legend(fontsize=20, frameon=False, loc='upper left')

    plt.tight_layout()

    return plt.gcf()

def cov_panel_combined():
    net_name_list = topologies("noisecov")
    
    means_time = []
    std_time = []
    means_strength = []
    std_strength = []
    rank_corr_time = []
    rank_corr_strength = []

    gap = 3  # Adjust this value to increase or decrease the gap
    
    for net_name in net_name_list:
        loaded_lists_time = load_pair("noisecov", "peaktime", net_name)
        if net_name == 'undirectedchain':
            thr_loaded_time = loaded_lists_time[0][1:15, 1:15]
            sim_loaded_time = loaded_lists_time[1][1:15, 1:15]
        else:
            thr_loaded_time = loaded_lists_time[0]
            sim_loaded_time = loaded_lists_time[1]
        
        thr_list_time = thr_loaded_time.flatten()
        sim_list_time = sim_loaded_time.flatten()
        
        valid = np.abs(sim_list_time) > 1e-8
        relative_error_time = np.abs((sim_list_time[valid] - thr_list_time[valid]) / sim_list_time[valid])
        means_time.append(np.mean(relative_error_time))
        std_time.append(np.std(relative_error_time))

        rank_corr_time.append(rank_correlation(thr_list_time, sim_list_time))

        loaded_lists_strength = load_pair("noisecov", "peak", net_name)
        if net_name == 'undirectedchain':
            thr_loaded_strength = loaded_lists_strength[0][1:15, 1:15]
            sim_loaded_strength = loaded_lists_strength[1][1:15, 1:15]
        else:
            thr_loaded_strength = loaded_lists_strength[0]
            sim_loaded_strength = loaded_lists_strength[1]
        
        thr_list_strength = thr_loaded_strength.flatten()
        sim_list_strength = sim_loaded_strength.flatten()
        
        relative_ratio_strength = thr_list_strength / sim_list_strength
        means_strength.append(np.mean(relative_ratio_strength))
        std_strength.append(np.std(relative_ratio_strength))

        rank_corr_strength.append(rank_correlation(thr_list_strength, sim_list_strength))
    
    num_networks = len(net_name_list)
    positions_time = np.arange(0, num_networks * (2 + gap), (2 + gap))  # Positions for time
    positions_strength = positions_time + 1  # Positions for strength next to time

    fig, ax1 = plt.subplots(figsize=(12, 3))  # Adjust the figure size if needed
    
    ax1.bar(positions_time, means_time, yerr=std_time, capsize=5, width=0.8, label=r'$resp. \ time$', color='#467F2D')

    for i in range(num_networks):
        ax1.text(positions_time[i], means_time[i] + std_time[i] + 0.05, f"{rank_corr_time[i]:.2f}",
                 ha='center', va='bottom', fontsize=12, color='black')

    ax1.set_yscale('log')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(True)
    ax1.spines['bottom'].set_visible(True)
    
    ax1.set_ylim([1e-2, 20])  # Adjust the lower limit if necessary
    
    ax1.set_xticks([])
    ax1.tick_params(axis='y', labelsize=20)
    
    ax2 = ax1.twinx()
    ax2.bar(positions_strength, means_strength, yerr=std_strength, capsize=5, width=0.8, label=r'$peak \ resp.$', color='#155F32')

    for i in range(num_networks):
        ax2.text(positions_strength[i], means_strength[i] + std_strength[i] + 0.05, f"{rank_corr_strength[i]:.2f}",
                 ha='center', va='bottom', fontsize=12, color='black')
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(True)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(axis='y', labelsize=20)
    
    ax2.set_ylim([0, 3])  # Adjust based on your data
    
    ax1.legend(loc='upper left', fontsize=20, frameon=False)
    ax2.legend(loc='upper right', fontsize=20, frameon=False)
    plt.tight_layout()

    return plt.gcf()

def render():
    OUT = Path("figures")
    OUT.mkdir(exist_ok=True)
    required = [array_path(panel, metric, topology)
                for panel, metrics in METRICS.items()
                for topology in topologies(panel) for metric in metrics]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing summary arrays:\n" + "\n".join(missing))
    panels = []
    for name, draw in zip(METRICS, (combined_bar_plot, pulse_panel_combined,
                                    lambda: square_panel("peak"), auto_panel_combined,
                                    cov_panel_combined)):
        fig = draw()
        fig.savefig(OUT / f"{name}.pdf")
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=300)
        panels.append(Image.open(buffer).convert("RGB"))
        plt.close(fig)
    combined = Image.new("RGB", (max(p.width for p in panels), sum(p.height for p in panels)), "white")
    y = 0
    for panel in panels:
        combined.paste(panel, (0, y))
        y += panel.height
    combined.save(OUT / "response_metrics.png")
    return combined
