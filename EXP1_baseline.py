"""
Description: Baseline experiment on synthetic data
"""

from utils.networkx_operations import *
from utils.pandas_operations import *
from utils.time_operations import *
from tqdm import tqdm
# import pickle
import pandas as pd
import numpy as np
import copy
import argparse

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import pickle

if __name__ == "__main__":


    beta_list = [1.0, 2.0, 4.0]

    accuracy_array = np.zeros((len(beta_list)))
    precision_array = np.zeros((len(beta_list)))
    recall_array = np.zeros((len(beta_list)))
    f1_array = np.zeros((len(beta_list)))
    TP_array = np.zeros((len(beta_list)))
    MCC_array = np.zeros((len(beta_list)))
    print(beta_list)

    x = 1

    for idx_beta, beta in enumerate(beta_list):
        # print("Loading the graph...")
        G = nx.read_graphml("data/G_synthetic_step2_beta{}_x{}_v3.graphml".format(int(beta), int(x)))
        G = relabel_nodes_str_to_tuple(G) 
        # r is the root node
        r = (0, -1)
        # Add a dummy node, then connect it to all other nodes
        add_dummy_node(G, r, 0.0, 0.0)
        G.nodes[r]["ASYMP"]=False
        G.nodes[r]["ASYMP_pred"]=False

        # get ASYMP
        ASYMP = np.array([G.nodes[v]["ASYMP"] for v in G.nodes()])

        ASYMP_dict = pickle.load(open("data/EXP1_ASYMP_dict_beta{}.pkl".format(int(beta)), "rb"))
        baseline_method_list = list(ASYMP_dict.keys())
        for idx_baseline, baseline_method in enumerate(baseline_method_list):
            ASYMP_nodes = ASYMP_dict[baseline_method]
            print(baseline_method)

            for v in G:
                if v in ASYMP_nodes:
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

            print("Acc: {:.3f}, Prec: {:.3f}, Rec: {:.3f}, f1: {:.3f}, TP: {}, MCC:{:.3f}".format(accuracy, precision, recall, f1, TP, MCC))

            # save results
            np.savez("npz/EXP1_B_{}_x{}_beta{}".format(baseline_method, int(x), int(beta)),
                accuracy_array=accuracy,
                precision_array=precision,
                recall_array=recall,
                f1_array=f1,
                TP_array=TP,
                MCC_array=MCC,
                )
