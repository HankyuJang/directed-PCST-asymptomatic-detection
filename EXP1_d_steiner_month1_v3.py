"""
Description: This script runs d_steiner on synthetic data
"""

from utils.networkx_operations import *
from utils.pandas_operations import *
from utils.time_operations import *
from utils.steiner_tree_te import *
# from utils.steiner_tree_v3 import *
from tqdm import tqdm
# import pickle
import pandas as pd
import numpy as np
import copy
import argparse

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='d_steiner on month1')
    parser.add_argument('-i', '--i', type=int, default=2,
                        help= 'paramter of the d_steiner algorithm.')
    args = parser.parse_args()

    i = args.i

    gamma_list = [0.0, 16.0, 128.0]
    beta_list = [1.0, 2.0, 4.0]
    alpha_list = [0.0] + [float(pow(2, x)) for x in range(9)]
    x_list = [1]

    cost_array = np.zeros((len(gamma_list), len(beta_list), len(alpha_list)))
    elapsed_time_array = np.zeros((len(gamma_list), len(beta_list), len(alpha_list)))
    accuracy_array = np.zeros((len(gamma_list), len(beta_list), len(alpha_list)))
    precision_array = np.zeros((len(gamma_list), len(beta_list), len(alpha_list)))
    recall_array = np.zeros((len(gamma_list), len(beta_list), len(alpha_list)))
    f1_array = np.zeros((len(gamma_list), len(beta_list), len(alpha_list)))
    TP_array = np.zeros((len(gamma_list), len(beta_list), len(alpha_list))).astype(int)
    MCC_array = np.zeros((len(gamma_list), len(beta_list), len(alpha_list)))
    print(beta_list, gamma_list, alpha_list)

    for x in x_list:
        for idx_gamma, gamma in enumerate(gamma_list):
            for idx_beta, beta in enumerate(beta_list):
                for idx_alpha, alpha in enumerate(alpha_list):
                    # print("Loading the graph...")
                    # G = nx.read_graphml("data/G_synthetic_step2_beta{}.graphml".format(int(beta)))
                    G = nx.read_graphml("data/G_synthetic_step2_beta{}_x{}_v3.graphml".format(int(beta), int(x)))
                    G = relabel_nodes_str_to_tuple(G) 
                    day_list = [d for v,d in G.nodes()]
                    max_day = max(day_list)
                    # X holds the steiner points
                    X_original = set([v for v in G.nodes() if G.nodes[v]["terminal"]])
                    X = set([v for v in G.nodes() if G.nodes[v]["terminal"]])
                    len_X = len(X)
                    k = len_X
                    print("Number of terminals: {}".format(k))
                    # print(nx.info(G))
                    # r is the root node
                    r = (0, -1)
                    # Add a dummy node, then connect it to all other nodes
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

                    print("Compute SP for i={}".format(i))
                    SP = dict()
                    # (1) Generate shortest path from r to all nodes
                    # p = nx.shortest_path(G, source=r, weight="weight")
                    # This SP algorithm can handle negative weights
                    p = nx.single_source_bellman_ford_path(G, source=r, weight="weight")
                    SP[r] = copy.deepcopy(p)

                    # (2) Generate shortest path from all nodes to all terminal nodes
                    # IDEA: reverse all edges, then compute shortest path from t to all nodes. Then, reverse the direction of final solution
                    G_reversed = G.reverse()
                    for t in tqdm(X):
                        # p = nx.shortest_path(G, target=t, weight="weight")
                        # This SP algorithm can handle negative weights
                        p = nx.single_source_bellman_ford_path(G_reversed, source=t, weight="weight")
                        for src in p.keys():
                            if src in SP:
                                SP[src][t] = copy.deepcopy(p[src][::-1])
                            else:
                                SP[src] = dict()
                                SP[src][t] = copy.deepcopy(p[src][::-1])

                    print("computing the tree using d_steiner algorithm with i={}, gamma={}, beta={}, alpha={}...".format(i, gamma, beta, alpha))
                    start = timer()
                    T = d_steiner(SP, i, G, X_original, k, r, X)
                    end = timer()
                    elapsed_time = get_elapsed_time(start, end)
                    elapsed_time_array[idx_gamma, idx_beta, idx_alpha] = elapsed_time
                    # Add node weights
                    node_weight_list = []
                    for node in T.nodes:
                        weight=G.nodes[node]["prob"]
                        node_weight_list.append((node, {"prob": weight}))
                    T.add_nodes_from(node_weight_list)
                    # Save the graph
                    nx.write_graphml(T, "result/synthetic_T_month1_i{}_beta{}_gamma{}_alpha{}_v3.graphml".format(i, int(beta), int(gamma), int(alpha)), named_key_ids=True)
                    cost = sum([T.edges[e]["weight"] for e in T.edges()])
                    cost_array[idx_gamma, idx_beta, idx_alpha] = cost

                    print(nx.info(T))
                    print("Elapsed_time: {:.1f} mins, Cost: {:.1f}".format(elapsed_time/60, cost))
                    
                    # get ASYMP
                    ASYMP = np.array([G.nodes[v]["ASYMP"] for v in G.nodes()])
                    # Add ASYMP_pred to G
                    for v in G:
                        if v in T:
                            G.nodes[v]["ASYMP_pred"] = True
                        else:
                            G.nodes[v]["ASYMP_pred"] = False
                    ASYMP_pred = np.array([G.nodes[v]["ASYMP_pred"] for v in G.nodes()])

                    accuracy = accuracy_score(ASYMP, ASYMP_pred)
                    precision = precision_score(ASYMP, ASYMP_pred)
                    recall = recall_score(ASYMP, ASYMP_pred)
                    f1 = f1_score(ASYMP, ASYMP_pred)
                    TP = (ASYMP & ASYMP_pred).sum()
                    MCC = matthews_corrcoef(ASYMP, ASYMP_pred)

                    accuracy_array[idx_gamma, idx_beta, idx_alpha] = accuracy
                    precision_array[idx_gamma, idx_beta, idx_alpha] = precision
                    recall_array[idx_gamma, idx_beta, idx_alpha] = recall
                    f1_array[idx_gamma, idx_beta, idx_alpha] = f1
                    TP_array[idx_gamma, idx_beta, idx_alpha] = TP
                    MCC_array[idx_gamma, idx_beta, idx_alpha] = MCC
                    print("Acc: {:.3f}, Prec: {:.3f}, Rec: {:.3f}, f1: {:.3f}, TP: {}, MCC:{:.3f}".format(accuracy, precision, recall, f1, TP, MCC))

        # save results
        np.savez("npz/EXP1_T_i{}_x{}_v3".format(i, int(x)),
                accuracy_array=accuracy_array,
                precision_array=precision_array,
                recall_array=recall_array,
                f1_array=f1_array,
                TP_array=TP_array,
                MCC_array=MCC_array,
                cost_array=cost_array,
                elapsed_time_array=elapsed_time_array
                )
