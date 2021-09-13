"""
Description: This script generates ASYMP for baselines

# NOTE: For this, we're not sharing the actual patient data, so comment out Strategy3
"""

from utils.networkx_operations import *
from utils.pandas_operations import *
from utils.time_operations import *
import pandas as pd
import numpy as np
import argparse
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='d_steiner on month1')
    parser.add_argument('-beta', '--beta', type=int, default=1,
                        help= 'beta in 1, 2, 4')
    args = parser.parse_args()
    beta = args.beta

    x = 1

    start_date = pd.Timestamp(2010, 1, 1)
    end_date = pd.Timestamp(2010, 2, 1)

    G = nx.read_graphml("data/G_synthetic_step2_beta{}_x{}_v3.graphml".format(beta, x))
    G = relabel_nodes_str_to_tuple(G) 
    terminal_node_set = set([v for v in G.nodes() if G.nodes[v]["terminal"]])
    print("terminal nodes: {}".format(len(terminal_node_set)))

    ##########################################################
    # Strategy1: Frontier. Neighbors of terminal nodes. In time extended graph, src node for the edge into terminal_case (ASYMP, terminal_case)
    ASYMP_frontier_list = []
    for terminal_node in terminal_node_set:
        neighbor_node_list = [u for u, v in G.in_edges(terminal_node)]
        ASYMP_frontier_list.extend(neighbor_node_list)
    ASYMP_frontier_set = set(ASYMP_frontier_list)
    ASYMP_frontier_set = ASYMP_frontier_set - terminal_node_set

    ##########################################################
    # Strategy2: Contact. Based on out degree 3% 5% 10%
    n_nodes = len(G)
    node_outdegree_pair_sorted = sorted(G.out_degree, key=lambda x: x[1], reverse=True)
    node_outdegree_sorted = [node for node, degree in node_outdegree_pair_sorted]
    ASYMP_contact_top3_set = set(node_outdegree_sorted[:int(n_nodes * 0.03)])
    ASYMP_contact_top5_set = set(node_outdegree_sorted[:int(n_nodes * 0.05)])
    ASYMP_contact_top10_set = set(node_outdegree_sorted[:int(n_nodes * 0.10)])

    ASYMP_contact_top3_set = ASYMP_contact_top3_set - terminal_node_set
    ASYMP_contact_top5_set = ASYMP_contact_top5_set - terminal_node_set
    ASYMP_contact_top10_set = ASYMP_contact_top10_set - terminal_node_set
    
    ##########################################################
    # NOTE: For this, we're not sharing the actual patient data, so comment out Strategy3
    # Strategy3: LOS. Based on LOS. Based on 3% 5% 10%
    df_CDI_cum = pd.read_csv("../prepare_input_for_PCST/data/CDI_EMR_cum.csv", parse_dates=["date"])
    df_CDIx_cum = pd.read_csv("../prepare_input_for_PCST/data/CDIx_EMR_cum.csv", parse_dates=["date"])
    df_CDI_cum = filter_records(df_CDI_cum, start_date, end_date)
    df_CDIx_cum = filter_records(df_CDIx_cum, start_date, end_date)

    df_dataset = pd.concat([
        df_CDI_cum[["vid", "date", "los"]],
        df_CDIx_cum[["vid", "date", "los"]] 
        ], axis=0)
    df_dataset = df_dataset.sort_values(by="los", ascending=False)
    day_array = (df_dataset.date - start_date).dt.days.values
    df_dataset.insert(loc=1, column="day", value=day_array)
    node_array = df_dataset[["vid", "day"]].apply(tuple, axis=1).values
    df_dataset.insert(loc=0, column="node", value=node_array)
    node_LOS_sorted = df_dataset.node.values

    ASYMP_LOS_top3_set = set(node_LOS_sorted[:int(n_nodes * 0.03)])
    ASYMP_LOS_top5_set = set(node_LOS_sorted[:int(n_nodes * 0.05)])
    ASYMP_LOS_top10_set = set(node_LOS_sorted[:int(n_nodes * 0.10)])

    ASYMP_LOS_top3_set = ASYMP_LOS_top3_set - terminal_node_set
    ASYMP_LOS_top5_set = ASYMP_LOS_top5_set - terminal_node_set
    ASYMP_LOS_top10_set = ASYMP_LOS_top10_set - terminal_node_set

    ##########################################################
    # Save asymptomatics
    ASYMP_dict = dict()
    ASYMP_dict["frontier"] = ASYMP_frontier_set
    ASYMP_dict["contact_top3"] = ASYMP_contact_top3_set
    ASYMP_dict["contact_top5"] = ASYMP_contact_top5_set
    ASYMP_dict["contact_top10"] = ASYMP_contact_top10_set
    ASYMP_dict["LOS_top3"] = ASYMP_LOS_top3_set
    ASYMP_dict["LOS_top5"] = ASYMP_LOS_top5_set
    ASYMP_dict["LOS_top10"] = ASYMP_LOS_top10_set

    pickle.dump(ASYMP_dict, open("data/EXP1_ASYMP_dict_beta{}.pkl".format(beta), "wb"))

    print("number of frontiers: {}".format(len(ASYMP_dict["frontier"])))
