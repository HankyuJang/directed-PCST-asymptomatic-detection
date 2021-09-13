"""
Description: This script generates figures for paper
(Fig5, varying alpha)
"""
import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
# import seaborn as sns
import matplotlib.pyplot as plt

from utils.networkx_operations import *
import numpy as np
import pandas as pd
import pickle
from numpy import unravel_index

def load_EXP1_npz(filename):
    npzfile = np.load(filename)
    accuracy_array = npzfile["accuracy_array"]   #0
    precision_array = npzfile["precision_array"]   #1
    recall_array = npzfile["recall_array"]   #2
    f1_array = npzfile["f1_array"]   #3
    TP_array = npzfile["TP_array"]   #4
    MCC_array = npzfile["MCC_array"]   #5
    cost_array = npzfile["cost_array"]   #6
    elapsed_time_array = npzfile["elapsed_time_array"]   #7
    npzfile.close()
    return accuracy_array, precision_array, recall_array, f1_array, TP_array, MCC_array, cost_array, elapsed_time_array

def load_EXP1_LP_npz(filename):
    npzfile = np.load(filename)
    accuracy_array = npzfile["accuracy_array"]   #0
    precision_array = npzfile["precision_array"]   #1
    recall_array = npzfile["recall_array"]   #2
    f1_array = npzfile["f1_array"]   #3
    TP_array = npzfile["TP_array"]   #4
    MCC_array = npzfile["MCC_array"]   #5
    npzfile.close()
    return accuracy_array, precision_array, recall_array, f1_array, TP_array, MCC_array

# Reshape 3x3x10 array to 9x10 array. Then, get max on axis=0
def prepare_max_score_per_alpha(metric_idx, EXP1_MCA, EXP1_T_i1, EXP1_T_i2, EXP1_LP):
    MCA_data = np.max(EXP1_MCA[metric_idx][:,beta_idx,:].reshape(-1,10), axis=0)
    T_i1_data = np.max(EXP1_T_i1[metric_idx][:,beta_idx,:].reshape(-1,10), axis=0)
    T_i2_data = np.max(EXP1_T_i2[metric_idx][:,beta_idx,:].reshape(-1,10), axis=0)
    
    EXP1_LP_avg = np.mean(EXP1_LP[metric_idx], axis=-1)
    LP_data = np.max(EXP1_LP_avg[:,beta_idx,:].reshape(-1,10), axis=0)
    return MCA_data, T_i1_data, T_i2_data, LP_data

def line_graph_synthetic(MCA, T_i1, T_i2, LP, xlabel, y_lim, title, filename):
    width = 0.2
    x = np.arange(len(xlabel))

    # ind = np.arange(len(xlabel))
    fig, ax = plt.subplots()
    rects_MCA = ax.bar(x-3*width/2, MCA, width, label="MCA", color="lightskyblue")
    rects_T_i1 = ax.bar(x-width/2, T_i1, width, label=r"$GREEDY_1$", color="dodgerblue")
    rects_T_i2 = ax.bar(x+width/2, T_i2, width, label=r"$GREEDY_2$", color="navy")
    rects_LP = ax.bar(x+3*width/2, LP, width, label="LP", color="black")

    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabel, rotation = 45)
    ax.set_ylim(y_lim)
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig("plots_matplotlib/{}".format(filename), dpi=300)
    plt.close()
    
def line_graph_synthetic_side_by_side(MCA_fig1, T_i1_fig1, T_i2_fig1, LP_fig1, MCA_fig2, T_i1_fig2, T_i2_fig2, LP_fig2, xlabel, y_lim, title1, title2, filename):
    fig = plt.figure(figsize=(16,6), dpi=300)
    gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
    (ax1, ax2) = gs.subplots(sharex='col', sharey='row')

    width = 0.2
    x = np.arange(len(xlabel))

    # ind = np.arange(len(xlabel))
    # fig, ax = plt.subplots()
    rects_MCA_fig1 = ax1.bar(x-3*width/2, MCA_fig1, width, label="MCA", color="lightskyblue")
    rects_T_i1_fig1 = ax1.bar(x-width/2, T_i1_fig1, width, label=r"$GREEDY_1$", color="dodgerblue")
    rects_T_i2_fig1 = ax1.bar(x+width/2, T_i2_fig1, width, label=r"$GREEDY_2$", color="navy")
    rects_LP_fig1 = ax1.bar(x+3*width/2, LP_fig1, width, label="LP", color="black")

    ax1.set_title(title1, fontsize=30)
    ax1.set_ylabel("Score", fontsize=25)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.set_xlabel(r"$\alpha$", fontsize=25)
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabel, rotation = 90, fontsize=25)
    ax1.set_ylim(y_lim)
    ax1.legend(prop={'size': 20}, loc="upper left")

    # ---------------------------
    rects_MCA_fig2 = ax2.bar(x-3*width/2, MCA_fig2, width, label="MCA", color="lightskyblue")
    rects_T_i1_fig2 = ax2.bar(x-width/2, T_i1_fig2, width, label=r"$GREEDY_1$", color="dodgerblue")
    rects_T_i2_fig2 = ax2.bar(x+width/2, T_i2_fig2, width, label=r"$GREEDY_2$", color="navy")
    rects_LP_fig2 = ax2.bar(x+3*width/2, LP_fig2, width, label="LP", color="black")

    # ax2.set_ylabel("Score")
    ax2.set_title(title2, fontsize=30)
    ax2.set_xlabel(r"$\alpha$", fontsize=25)
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabel, rotation = 90, fontsize=25)
    ax2.set_ylim(y_lim)
    # ax2.legend(prop={'size': 20})

    plt.tight_layout()
    plt.savefig("plots_matplotlib/{}".format(filename), dpi=300)
    plt.close()
    
def get_TPR(n_ASYMP, TP_list):
    return [TP/float(n_ASYMP) for TP in TP_list]

if __name__ == "__main__":
    x = 1
    f1_idx = 3
    TP_idx = 4
    MCC_idx = 5
    beta_list = [1,2,4]
    # beta_list = [1]
    gamma_list = [0, 16, 128]
    ####################
    # beta_idx = 0
    # beta = beta_list[beta_idx]

    for beta_idx, beta in enumerate(beta_list):
        EXP1_MCA = load_EXP1_npz("npz/EXP1_MCA_x{}_v3.npz".format(int(x)))
        EXP1_T_i1 = load_EXP1_npz("npz/EXP1_T_i1_x{}_v3.npz".format(int(x)))
        EXP1_T_i2 = load_EXP1_npz("npz/EXP1_T_i2_x{}_v3.npz".format(int(x)))
        EXP1_LP = load_EXP1_LP_npz("npz/LP/EXP1_LP.npz".format(int(x)))

        alpha_list = [0] + [pow(2, x) for x in range(9)]
        xlabel = [r"$\alpha={}$".format(alpha) for alpha in alpha_list]
        # MCC
        MCA_data_MCC, T_i1_data_MCC, T_i2_data_MCC, LP_data_MCC= prepare_max_score_per_alpha(MCC_idx, EXP1_MCA, EXP1_T_i1, EXP1_T_i2, EXP1_LP)
        title = "MCC score"
        filename = "EXP1_MCC_line_beta{}.png".format(beta)
        y_lim = (0, 0.5)
        line_graph_synthetic(MCA_data_MCC, T_i1_data_MCC, T_i2_data_MCC, LP_data_MCC, xlabel, y_lim, title, filename)

        # f1
        xlabel = [r"$\alpha={}$".format(alpha) for alpha in alpha_list]
        MCA_data_f1, T_i1_data_f1, T_i2_data_f1, LP_data_f1 = prepare_max_score_per_alpha(f1_idx, EXP1_MCA, EXP1_T_i1, EXP1_T_i2, EXP1_LP)
        title = "F1 score"
        filename = "EXP1_f1_line_beta{}.png".format(beta)
        y_lim = (0, 0.5)
        line_graph_synthetic(MCA_data_f1, T_i1_data_f1, T_i2_data_f1, LP_data_f1, xlabel, y_lim, title, filename)

        # MCC and f1
        xlabel = alpha_list
        title1 = "MCC score"
        title2 = "F1 score"
        filename = "EXP1_MCC_f1_line_beta{}.png".format(beta)
        y_lim = (0, 0.5)
        line_graph_synthetic_side_by_side(MCA_data_MCC, T_i1_data_MCC, T_i2_data_MCC, LP_data_MCC, MCA_data_f1, T_i1_data_f1, T_i2_data_f1, LP_data_f1, xlabel, y_lim, title1, title2, filename)

        # TP
        G = nx.read_graphml("data/G_synthetic_step2_beta{}_x{}_v3.graphml".format(int(beta), int(x)))
        G = relabel_nodes_str_to_tuple(G) 
        ASYMP = set([v for v in G.nodes() if G.nodes[v]["ASYMP"]])
        n_ASYMP = len(ASYMP)
        print("Number of ASYMP cases: {}".format(n_ASYMP))

        MCA_data_TP, T_i1_data_TP, T_i2_data_TP, LP_data_TP = prepare_max_score_per_alpha(TP_idx, EXP1_MCA, EXP1_T_i1, EXP1_T_i2, EXP1_LP)
        MCA_data_TPR = get_TPR(n_ASYMP, MCA_data_TP)
        T_i1_data_TPR = get_TPR(n_ASYMP, T_i1_data_TP)
        T_i2_data_TPR = get_TPR(n_ASYMP, T_i2_data_TP)
        LP_data_TPR = get_TPR(n_ASYMP, LP_data_TP)
        title = "True positive rate"
        filename = "EXP1_TPR_line_beta{}.png".format(beta)
        y_lim = (0, 1.0)
        line_graph_synthetic(MCA_data_TPR, T_i1_data_TPR, T_i2_data_TPR, LP_data_TPR, xlabel, y_lim, title, filename)
