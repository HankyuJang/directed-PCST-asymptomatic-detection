"""
Description: This script calibrate result from LP solution.
Probablistically select ASYMPs.
Then, compute metric features from those.
"""

from utils.networkx_operations import *
from utils.pandas_operations import *
from utils.time_operations import *
from tqdm import tqdm
import pandas as pd
import numpy as np
import copy
import argparse

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

def compute_metric_measures(ASYMP, ASYMP_pred):
    accuracy = accuracy_score(ASYMP, ASYMP_pred)
    precision = precision_score(ASYMP, ASYMP_pred)
    recall = recall_score(ASYMP, ASYMP_pred)
    f1 = f1_score(ASYMP, ASYMP_pred)
    TP = (ASYMP & ASYMP_pred).sum()
    MCC = matthews_corrcoef(ASYMP, ASYMP_pred)
    return accuracy, precision, recall, f1, TP, MCC

def update_metric_arrays(idx_gamma, idx_beta, idx_alpha, idx_cnt):
    accuracy_array[idx_gamma, idx_beta, idx_alpha, idx_cnt] = accuracy
    precision_array[idx_gamma, idx_beta, idx_alpha, idx_cnt] = precision
    recall_array[idx_gamma, idx_beta, idx_alpha, idx_cnt] = recall
    f1_array[idx_gamma, idx_beta, idx_alpha, idx_cnt] = f1
    TP_array[idx_gamma, idx_beta, idx_alpha, idx_cnt] = TP
    MCC_array[idx_gamma, idx_beta, idx_alpha, idx_cnt] = MCC
    n_ASYMP_array[idx_gamma, idx_beta, idx_alpha, idx_cnt] = n_ASYMP

if __name__ == "__main__":
    gamma_list = [0, 16, 128]
    beta_list = [1, 2, 4]
    alpha_list = [0] + [pow(2, x) for x in range(9)]
    x = 1
    n_sample = 10 

    # here, n_sample+1 because we save one more set of ASYMP by choosing only those with y in sum prob = 1
    accuracy_array = np.zeros((len(gamma_list), len(beta_list), len(alpha_list), n_sample))
    precision_array = np.zeros((len(gamma_list), len(beta_list), len(alpha_list), n_sample))
    recall_array = np.zeros((len(gamma_list), len(beta_list), len(alpha_list), n_sample))
    f1_array = np.zeros((len(gamma_list), len(beta_list), len(alpha_list), n_sample))
    TP_array = np.zeros((len(gamma_list), len(beta_list), len(alpha_list), n_sample)).astype(int)
    MCC_array = np.zeros((len(gamma_list), len(beta_list), len(alpha_list), n_sample))
    n_ASYMP_array = np.zeros((len(gamma_list), len(beta_list), len(alpha_list), n_sample)).astype(int)

    for idx_gamma, gamma in enumerate(gamma_list):
        for idx_beta, beta in enumerate(beta_list):
            # if idx_gamma==2 and idx_beta==2:
                # break
            for idx_alpha, alpha in enumerate(alpha_list):
                print("gamma: {}, beta: {}, alpha: {}".format(gamma, beta, alpha))
                # print("Loading G...")
                G = nx.read_graphml("data/G_synthetic_step2_beta{}_x{}_v3.graphml".format(int(beta), int(x)))
                G = relabel_nodes_str_to_tuple(G) 
                # print(nx.info(G))
                X = set([v for v in G.nodes() if G.nodes[v]["terminal"]])
                len_X = len(X)
                k = len_X
                r = (0, -1)
                add_dummy_node(G, r, 0.0, gamma)
                G.nodes[r]["ASYMP"]=False
                G.nodes[r]["ASYMP_pred"]=False

                print("Update edge weights by subtracting scaled up the node weights")
                W_over_k = alpha * sum([G.nodes[v]["prob"] for v in G.nodes()]) / k
                for e in G.edges():
                    src = e[0]
                    dst = e[1]
                    weight = G.edges[src, dst]["weight"]
                    if dst in X:
                        adj_weight = weight + W_over_k
                        # print(G.nodes[dst]["prob"])
                    else:
                        adj_weight = weight - alpha * G.nodes[dst]["prob"]
                    G.edges[src, dst]["weight"] = adj_weight

                ASYMP = np.array([G.nodes[v]["ASYMP"] for v in G.nodes()])

                print("Loading G_solution...")
                G_solution = nx.read_graphml("result/synthetic_LP_month1_beta{}_gamma{}_alpha{}_v3.graphml".format(int(beta), int(gamma), int(alpha)))
                G_solution = relabel_nodes_str_to_tuple(G_solution) 
                print(nx.info(G_solution))


                #################################################################################################
                # Now select ASYMP probablistically.
                sum_in_y_array = np.zeros((len(G)))
                for idx, v in enumerate(G.nodes()):
                    if v in G_solution:
                        sum_in_y_array[idx] = sum([G_solution.edges[e]["flow"] for e in G_solution.in_edges(v)])

                # some y value have 1.0000000000002 which cause error when calling function np.random.binomial
                sum_in_y_array = np.clip(sum_in_y_array, 0, 1)

                idx_cnt = 0
                for i in range(10):
                    ASYMP_pred = np.random.binomial(1, sum_in_y_array).astype(bool)
                    # ASYMP_pred = ASYMP_pred | ASYMP_sampled
                    accuracy, precision, recall, f1, TP, MCC = compute_metric_measures(ASYMP, ASYMP_pred)
                    n_ASYMP = ASYMP_pred.sum()
                    print("nASYMP: {}, Acc: {:.3f}, Prec: {:.3f}, Rec: {:.3f}, f1: {:.3f}, TP: {}, MCC:{:.3f}".format(n_ASYMP, accuracy, precision, recall, f1, TP, MCC))
                    update_metric_arrays(idx_gamma, idx_beta, idx_alpha, idx_cnt)
                    idx_cnt += 1

    # save results
    print("Saving results as npz/LP/EXP1_LP.npz")
    np.savez("npz/LP/EXP1_LP".format(int(x)),
            accuracy_array=accuracy_array,
            precision_array=precision_array,
            recall_array=recall_array,
            f1_array=f1_array,
            TP_array=TP_array,
            MCC_array=MCC_array,
            n_ASYMP_array=n_ASYMP_array
            )
