"""
Description: This script computes the number of ASYMP and terminals in the synthetic data
"""

import numpy as np
import pandas as pd
from utils.networkx_operations import *
from numpy import unravel_index

if __name__ == "__main__":

    x_list = [1]
    beta_list = [1.0, 2.0, 4.0]
    for x in x_list:
        # network statistics
        n_terminal_list = []
        n_ASYMP_list = []
        for beta in beta_list:
            G = nx.read_graphml("data/G_synthetic_step2_beta{}_x{}_v3.graphml".format(int(beta), int(x)))
            G = relabel_nodes_str_to_tuple(G) 
            X = [v for v in G.nodes() if G.nodes[v]["terminal"]]
            ASYMP = [v for v in G.nodes() if G.nodes[v]["ASYMP"]]
            n_terminal_list.append(len(X))
            n_ASYMP_list.append(len(ASYMP))
        df_synthetic_graph = pd.DataFrame(data={"n_terminals": n_terminal_list, "n_ASYMP": n_ASYMP_list}, index=beta_list)
        df_synthetic_graph.to_csv("table/EXP1_G_synthetic_x{}_v3.csv".format(int(x)), index=True)
