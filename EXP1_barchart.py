"""
Description: This script generates figures for paper
(Fig4, performance of all methods)
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

def load_result_table(filename):
    df_result = pd.read_csv("table/{}".format(filename), index_col=0)
    return df_result

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

def load_B(pklname, x, beta):
    B = dict()
    ASYMP_dict = pickle.load(open(pklname, "rb"))
    baseline_method_list = list(ASYMP_dict.keys())
    print(baseline_method_list)
    for idx_baseline, baseline_method in enumerate(baseline_method_list):
        npzfile = np.load("npz/EXP1_B_{}_x{}_beta{}.npz".format(baseline_method, int(x), int(beta)))
        accuracy_array = npzfile["accuracy_array"]   #0
        precision_array = npzfile["precision_array"]   #1
        recall_array = npzfile["recall_array"]   #2
        f1_array = npzfile["f1_array"]   #3
        TP_array = npzfile["TP_array"]   #4
        MCC_array = npzfile["MCC_array"]   #5
        npzfile.close()
        B[baseline_method] = [accuracy_array, precision_array, recall_array, f1_array, TP_array, MCC_array]
    return B, baseline_method_list

def prepare_list_of_AUCs(B, metric_idx, EXP1_MCA, EXP1_T_i1, EXP1_T_i2, EXP1_LP):
    B_frontier_metric = B["frontier"][metric_idx][()] # 0 dim array
    B_contact_top3_metric = B["contact_top3"][metric_idx][()] # 0 dim array
    B_contact_top5_metric = B["contact_top5"][metric_idx][()] # 0 dim array
    B_contact_top10_metric = B["contact_top10"][metric_idx][()] # 0 dim array
    B_LOS_top3_metric = B["LOS_top3"][metric_idx][()] # 0 dim array
    B_LOS_top5_metric = B["LOS_top5"][metric_idx][()] # 0 dim array
    B_LOS_top10_metric = B["LOS_top10"][metric_idx][()] # 0 dim array

    MCA_alpha_0_metric = np.max(EXP1_MCA[metric_idx][:,beta_idx,0])
    MCA_alpha_geq1_metric = np.max(EXP1_MCA[metric_idx][:,beta_idx,1:])

    T_i1_alpha_0_metric = np.max(EXP1_T_i1[metric_idx][:,beta_idx,0])
    T_i1_alpha_geq1_metric = np.max(EXP1_T_i1[metric_idx][:,beta_idx,1:])

    T_i2_alpha_0_metric = np.max(EXP1_T_i2[metric_idx][:,beta_idx,0])
    T_i2_alpha_geq1_metric = np.max(EXP1_T_i2[metric_idx][:,beta_idx,1:])

    EXP1_LP_avg = np.mean(EXP1_LP[metric_idx], axis=-1)
    LP_alpha_0_metric = np.max(EXP1_LP_avg[:,beta_idx,0])
    LP_alpha_geq1_metric = np.max(EXP1_LP_avg[:,beta_idx,1:])

    idx_MCA_alpha_0_metric = np.argmax(EXP1_MCA[metric_idx][:,beta_idx,0])
    idx_MCA_alpha_geq1_metric = np.argmax(EXP1_MCA[metric_idx][:,beta_idx,1:])
    idx_T_i1_alpha_0_metric = np.argmax(EXP1_T_i1[metric_idx][:,beta_idx,0])
    idx_T_i1_alpha_geq1_metric = np.argmax(EXP1_T_i1[metric_idx][:,beta_idx,1:])
    idx_T_i2_alpha_0_metric = np.argmax(EXP1_T_i2[metric_idx][:,beta_idx,0])
    idx_T_i2_alpha_geq1_metric = np.argmax(EXP1_T_i2[metric_idx][:,beta_idx,1:])
    idx_LP_alpha_0_metric = np.argmax(EXP1_LP_avg[:,beta_idx,0])
    idx_LP_alpha_geq1_metric = np.argmax(EXP1_LP_avg[:,beta_idx,1:])

    metric_baseline = [B_frontier_metric, B_contact_top3_metric, B_contact_top5_metric, B_contact_top10_metric, B_LOS_top3_metric, B_LOS_top5_metric, B_LOS_top10_metric]
    metric_alpha_0 = [MCA_alpha_0_metric, T_i1_alpha_0_metric, T_i2_alpha_0_metric, LP_alpha_0_metric]
    metric_alpha_geq1 = [MCA_alpha_geq1_metric, T_i1_alpha_geq1_metric, T_i2_alpha_geq1_metric, LP_alpha_geq1_metric]

    argmax_idx = [idx_MCA_alpha_0_metric, idx_MCA_alpha_geq1_metric, \
            idx_T_i1_alpha_0_metric, idx_T_i1_alpha_geq1_metric,\
            idx_T_i2_alpha_0_metric, idx_T_i2_alpha_geq1_metric, \
            idx_LP_alpha_0_metric, idx_LP_alpha_geq1_metric
            ]

    return metric_baseline, metric_alpha_0, metric_alpha_geq1, argmax_idx

def prepare_list_of_AUCs_w_index(B, metric_idx, EXP1_MCA, EXP1_T_i1, EXP1_T_i2, EXP1_LP, argmax_index):
    B_frontier_metric = B["frontier"][metric_idx][()] # 0 dim array
    B_contact_top3_metric = B["contact_top3"][metric_idx][()] # 0 dim array
    B_contact_top5_metric = B["contact_top5"][metric_idx][()] # 0 dim array
    B_contact_top10_metric = B["contact_top10"][metric_idx][()] # 0 dim array
    B_LOS_top3_metric = B["LOS_top3"][metric_idx][()] # 0 dim array
    B_LOS_top5_metric = B["LOS_top5"][metric_idx][()] # 0 dim array
    B_LOS_top10_metric = B["LOS_top10"][metric_idx][()] # 0 dim array

    MCA_alpha_0_metric = EXP1_MCA[metric_idx][:,beta_idx,0][argmax_index[0]]

    max_index = unravel_index(argmax_index[1], EXP1_MCA[metric_idx][:,beta_idx,1:].shape)
    MCA_alpha_geq1_metric = EXP1_MCA[metric_idx][:,beta_idx,1:][max_index]

    T_i1_alpha_0_metric = EXP1_T_i1[metric_idx][:,beta_idx,0][argmax_index[2]]

    max_index = unravel_index(argmax_index[3], EXP1_T_i1[metric_idx][:,beta_idx,1:].shape)
    T_i1_alpha_geq1_metric = EXP1_T_i1[metric_idx][:,beta_idx,1:][max_index]

    T_i2_alpha_0_metric = EXP1_T_i2[metric_idx][:,beta_idx,0][argmax_index[4]]

    max_index = unravel_index(argmax_index[5], EXP1_T_i2[metric_idx][:,beta_idx,1:].shape)
    T_i2_alpha_geq1_metric = EXP1_T_i2[metric_idx][:,beta_idx,1:][max_index]

    # LP has 10 sets
    EXP1_LP_avg = np.mean(EXP1_LP[metric_idx], axis=-1)
    LP_alpha_0_metric = EXP1_LP_avg[:,beta_idx,0][argmax_index[6]]

    max_index = unravel_index(argmax_index[7], EXP1_LP_avg[:,beta_idx,1:].shape)
    LP_alpha_geq1_metric = EXP1_LP_avg[:,beta_idx,1:][max_index]

    metric_baseline = [B_frontier_metric, B_contact_top3_metric, B_contact_top5_metric, B_contact_top10_metric, B_LOS_top3_metric, B_LOS_top5_metric, B_LOS_top10_metric]
    metric_alpha_0 = [MCA_alpha_0_metric, T_i1_alpha_0_metric, T_i2_alpha_0_metric, LP_alpha_0_metric]
    metric_alpha_geq1 = [MCA_alpha_geq1_metric, T_i1_alpha_geq1_metric, T_i2_alpha_geq1_metric, LP_alpha_geq1_metric]
    return metric_baseline, metric_alpha_0, metric_alpha_geq1

# def bar_chart_synthetic(baseline, alpha_0, alpha_geq1, xlabel, y_lim, title, filename):
def bar_chart_synthetic(data_for_fig, xlabel, y_lim, title, filename):
    width, gap = 0.9, 1

    x_frontier = np.arange(len(data_for_fig[0]))
    x_contact_topk = np.arange(len(data_for_fig[1])) + gap + len(data_for_fig[0])
    x_LOS_topk = np.arange(len(data_for_fig[2])) + 2*gap + len(data_for_fig[0]) + len(data_for_fig[1])
    x_PCST = np.arange(len(data_for_fig[3])) + 3*gap + len(data_for_fig[0]) + len(data_for_fig[1]) + len(data_for_fig[2])

    # ind = np.concatenate((x_baseline, x_alpha_0, x_alpha_geq1))
    ind = np.concatenate((x_frontier, x_contact_topk, x_LOS_topk, x_PCST))
    fig, ax = plt.subplots()

    rects_frontier = ax.bar(x_frontier, data_for_fig[0], width, color="tab:gray", label="Frontier")
    rects_contact_topk = ax.bar(x_contact_topk,  data_for_fig[1], width, color="tab:olive", label="Contact top k")
    rects_LOS_topk = ax.bar(x_LOS_topk,  data_for_fig[2], width, color="tab:pink", label="LOS top k")
    rects_PCST = ax.bar(x_PCST,  data_for_fig[3], width, color="tab:blue", label="Directed PCST")
    # rects_baseline = ax.bar(x_baseline, baseline, width, color="tab:gray", label="Baseline")
    # rects_alpha_0 = ax.bar(x_alpha_0, alpha_0, width, color="tab:olive", label=r"$\alpha=0$")
    # rects_alpha_geq1 = ax.bar(x_alpha_geq1, alpha_geq1, width, color="tab:blue", label="Directed PCST")

    ax.set_ylabel("Score")
    ax.set_xticks(ind)
    ax.set_xticklabels(xlabel, rotation = 45)
    ax.set_ylim(y_lim)
    ax.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig("plots_matplotlib/{}".format(filename), dpi=300)
    plt.close()

def bar_chart_synthetic_side_by_side(data_for_fig1, data_for_fig2, xlabel, y_lim, title1, title2, filename):
    fig = plt.figure(figsize=(16,6), dpi=300)
    gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
    (ax1, ax2) = gs.subplots(sharex='col', sharey='row')
    # fig.suptitle(title)
    width, gap = 0.9, 1

    #FIG1#################################################
    x_frontier = np.arange(len(data_for_fig1[0]))
    x_contact_topk = np.arange(len(data_for_fig1[1])) + gap + len(data_for_fig1[0])
    x_LOS_topk = np.arange(len(data_for_fig1[2])) + 2*gap + len(data_for_fig1[0]) + len(data_for_fig1[1])
    x_PCST = np.arange(len(data_for_fig1[3])) + 3*gap + len(data_for_fig1[0]) + len(data_for_fig1[1]) + len(data_for_fig1[2])

    # ind = np.concatenate((x_baseline, x_alpha_0, x_alpha_geq1))
    ind = np.concatenate((x_frontier, x_contact_topk, x_LOS_topk, x_PCST))
    # fig, ax1 = plt.subplots()

    rects_frontier = ax1.bar(x_frontier, data_for_fig1[0], width, color="tab:gray", label="Frontier")
    rects_contact_topk = ax1.bar(x_contact_topk,  data_for_fig1[1], width, color="tab:olive", label="Contact top k")
    rects_LOS_topk = ax1.bar(x_LOS_topk,  data_for_fig1[2], width, color="tab:pink", label="LOS top k")
    rects_PCST = ax1.bar(x_PCST,  data_for_fig1[3], width, color="tab:blue", label="Directed PCST")

    ax1.set_title(title1, fontsize=30)
    ax1.set_ylabel("Score", fontsize=25)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.set_xticks(ind)
    ax1.set_xticklabels(xlabel, rotation = 90, fontsize=25)
    ax1.set_ylim(y_lim)
    ax1.legend(prop={'size': 20}, loc="upper left")
    #FIG2#################################################
    ind = np.concatenate((x_frontier, x_contact_topk, x_LOS_topk, x_PCST))
    # fig, ax2 = plt.subplots()

    rects_frontier = ax2.bar(x_frontier, data_for_fig2[0], width, color="tab:gray", label="Frontier")
    rects_contact_topk = ax2.bar(x_contact_topk,  data_for_fig2[1], width, color="tab:olive", label="Contact top k")
    rects_LOS_topk = ax2.bar(x_LOS_topk,  data_for_fig2[2], width, color="tab:pink", label="LOS top k")
    rects_PCST = ax2.bar(x_PCST,  data_for_fig2[3], width, color="tab:blue", label="Directed PCST")

    ax2.set_title(title2, fontsize=30)
    # ax2.set_ylabel("Score")
    ax2.set_xticks(ind)
    ax2.set_xticklabels(xlabel, rotation = 90, fontsize=25)
    ax2.set_ylim(y_lim)
    # ax2.legend(prop={'size': 20})
    #plt#################################################

    # plt.title(title)
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
    ####################
    # beta_idx = 0
    # beta = beta_list[beta_idx]

    for beta_idx, beta in enumerate(beta_list):
        # load baselines
        pklname = "data/EXP1_ASYMP_dict_beta{}.pkl".format(int(beta))
        B, baseline_method_list = load_B(pklname, x, beta)

        xlabel = ["Frontier"] + ["10%", "5%", "3%"] * 2 + ["MCA", r"$GREEDY_1$", r"$GREEDY_2$", "LP"]

        EXP1_MCA = load_EXP1_npz("npz/EXP1_MCA_x{}_v3.npz".format(int(x)))
        EXP1_T_i1 = load_EXP1_npz("npz/EXP1_T_i1_x{}_v3.npz".format(int(x)))
        EXP1_T_i2 = load_EXP1_npz("npz/EXP1_T_i2_x{}_v3.npz".format(int(x)))
        EXP1_LP = load_EXP1_LP_npz("npz/LP/EXP1_LP.npz".format(int(x)))

        # xlabel = baseline_method_list + ["MCA", "DST.i=1"] #*2
        ##################
        # MCC
        MCC_baseline, MCC_alpha_0, MCC_alpha_geq1, argmax_idx = prepare_list_of_AUCs(B, MCC_idx, EXP1_MCA, EXP1_T_i1, EXP1_T_i2, EXP1_LP)
        frontier = [MCC_baseline[0]]
        contact_topk = MCC_baseline[1:4][::-1]
        LOS_topk = MCC_baseline[4:][::-1] 

        title = "MCC score"
        filename = "EXP1_MCC_beta{}.png".format(beta)
        y_lim = (0, 0.5)
        data_for_fig1 = (frontier, contact_topk, LOS_topk, MCC_alpha_geq1)
        bar_chart_synthetic(data_for_fig1, xlabel, y_lim, title, filename)

        # Rest of the results from algo is based on the best performing model on MCC
        # f1
        f1_baseline, f1_alpha_0, f1_alpha_geq1 = prepare_list_of_AUCs_w_index(B, f1_idx, EXP1_MCA, EXP1_T_i1, EXP1_T_i2, EXP1_LP, argmax_idx)
        frontier = [f1_baseline[0]]
        contact_topk = f1_baseline[1:4][::-1]
        LOS_topk = f1_baseline[4:][::-1] 
        title = "F1 score"
        filename = "EXP1_f1_beta{}.png".format(beta)
        y_lim = (0, 0.5)
        data_for_fig2 = (frontier, contact_topk, LOS_topk, f1_alpha_geq1)
        bar_chart_synthetic(data_for_fig2, xlabel, y_lim, title, filename)
        print("beta: {}".format(beta))
        print("[Baseline] Best MCC: {:.3f}, F1: {:.3f}".format(max(MCC_baseline), max(f1_baseline)))
        print("[Dir-PCST] Best MCC: {:.3f}, F1: {:.3f}".format(max(MCC_alpha_geq1), max(f1_alpha_geq1)))

        # SIDE by side bar chart
        title1 = "MCC score"
        title2 = "F1 score"
        filename = "EXP1_MCC_f1_beta{}.png".format(beta)
        y_lim = (0, 0.5)
        bar_chart_synthetic_side_by_side(data_for_fig1, data_for_fig2, xlabel, y_lim, title1, title2, filename)

        # TP
        G = nx.read_graphml("data/G_synthetic_step2_beta{}_x{}_v3.graphml".format(int(beta), int(x)))
        G = relabel_nodes_str_to_tuple(G) 
        ASYMP = set([v for v in G.nodes() if G.nodes[v]["ASYMP"]])
        n_ASYMP = len(ASYMP)
        print("Number of ASYMP cases: {}".format(n_ASYMP))

        TP_baseline, TP_alpha_0, TP_alpha_geq1 = prepare_list_of_AUCs_w_index(B, TP_idx, EXP1_MCA, EXP1_T_i1, EXP1_T_i2, EXP1_LP, argmax_idx)
        TPR_baseline = get_TPR(n_ASYMP, TP_baseline)
        TPR_alpha_0 = get_TPR(n_ASYMP, TP_alpha_0)
        TPR_alpha_geq1 = get_TPR(n_ASYMP, TP_alpha_geq1)

        frontier = [TPR_baseline[0]]
        contact_topk = TPR_baseline[1:4][::-1]
        LOS_topk = TPR_baseline[4:][::-1] 

        title = "True positive rate"
        filename = "EXP1_TPR_beta{}.png".format(beta)
        y_lim = (0, 1.0)
        data_for_fig = (frontier, contact_topk, LOS_topk, TPR_alpha_geq1)
        bar_chart_synthetic(data_for_fig, xlabel, y_lim, title, filename)

