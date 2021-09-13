"""
Description: This script runs LP on synthetic data
"""

from utils.networkx_operations import *
from utils.pandas_operations import *
from utils.time_operations import *
# from utils.steiner_tree_v3 import *
from tqdm import tqdm
# import pickle
import pandas as pd
import numpy as np
import copy
import argparse

import gurobipy as gp
from gurobipy import GRB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

if __name__ == "__main__":

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

                    print(nx.info(G))
                    print("|E| if weight <= 0: {}".format(len([e for e in G.edges() if G.edges[e]["weight"]<=0])))

                    print("Run LP on the graph where gamma={}, beta={}, alpha={}...".format(gamma, beta, alpha))
                    start = timer()

                    # Create mapping from nodes (tuple) to integer.
                    ID_V = dict([(idx, v) for idx, v in enumerate(G.nodes())])
                    V_ID = dict([(v, idx) for idx, v in enumerate(G.nodes())])
                    r = V_ID[r]

                    X_list = list(X)
                    terminals = [V_ID[s] for s in X_list]

                    # terminals = list(range(len(X_list)))
                    nodes = [V_ID[v] for v in G.nodes()]
                    arcs = [(V_ID[i], V_ID[j]) for i, j in G.edges()]
                    # cost of each edge
                    # cost = np.array([G.edges[e]["weight"] for e in G.edges()])
                    cost = {}
                    for i,j in G.edges():
                        cost[(V_ID[i], V_ID[j])] = G.edges[(i,j)]["weight"]

                    # Create optimization model
                    m = gp.Model('steiner_tree')

                    # Create variables

                    # s is a tuple, which gets decomposed if used (terminals, arcs) as parameter
                    # flow = m.addVars(terminals, arcs, name="flow")
                    print("computing flow tuples")
                    flow_list_of_tuples = []
                    for s in terminals:
                        s_V = ID_V[s]
                        s_day = s_V[1]
                        for u, v in arcs:
                            u_V = ID_V[u]
                            u_day = u_V[1]
                            if u_day < s_day:
                                flow_list_of_tuples.append((s,u,v))
                    flow = m.addVars(flow_list_of_tuples, name="flow")
                    print("num flow variables: {}".format(len(flow)))

                    # y = m.addMVar(shape=len(arcs), name="y")
                    y_list_of_tuples = []
                    for u, v in arcs:
                        # Check if flow[s,u,v] exists.
                        flow_uv = [(flow[s, u, v]) for s in terminals if (s, u, v) in flow]
                        if len(flow_uv) > 0:
                            y_list_of_tuples.append((u,v))
                    y = m.addVars(y_list_of_tuples, name="flow")
                    # y = m.addVars(arcs, name="y")
                    print("num y variables: {}".format(len(y)))

                    # m.setObjective(cost @ y, GRB.MINIMIZE)
                    print("objective...")
                    obj = sum([cost[e]*y[e] for e in arcs if e in y])
                    m.setObjective(obj, GRB.MINIMIZE)

                    # print("Constraint 0...")
                    # m.addConstrs(
                        # (y[i, j] == gp.max_([flow[s, i, j] for s in terminals if (s, i, j) in flow])
                            # for i, j in arcs if (i, j) in y), "c0")

                    # constraint 1
                    print("Constraint 1...")
                    # Root : if it's the root, G.in_edges(nbunch=r, data=False) is an empty array []
                    print("Constraint 1_(root)...") # v==r
                    m.addConstrs(
                        (sum([flow[s, V_ID[v], V_ID[u]] for v, u in G.out_edges(nbunch=ID_V[r], data=False)]) \
                        - sum([flow[s, V_ID[u], V_ID[v]] for u, v in G.in_edges(nbunch=ID_V[r], data=False)]) == 1
                            for s in terminals), "c1_r")
                    # Terminals
                    print("Constraint 1_(t)...") # v==t
                    m.addConstrs(
                        (sum([flow[s, V_ID[v], V_ID[u]] for v, u in G.out_edges(nbunch=ID_V[s], data=False) if (s, V_ID[v], V_ID[u]) in flow]) \
                        - sum([flow[s, V_ID[u], V_ID[v]] for u, v in G.in_edges(nbunch=ID_V[s], data=False) if (s, V_ID[u], V_ID[v]) in flow]) == -1
                            for s in terminals), "c1_s")

                    print("Constraint 1_(other)...") # v==node
                    other_nodes = set(nodes)
                    other_nodes -= set(terminals)
                    other_nodes -= set([r])
                    for node in other_nodes:
                        m.addConstrs(
                            (sum([flow[s, V_ID[v], V_ID[u]] for v, u in G.out_edges(nbunch=ID_V[node], data=False) if (s, V_ID[v], V_ID[u]) in flow]) \
                            - sum([flow[s, V_ID[u], V_ID[v]] for u, v in G.in_edges(nbunch=ID_V[node], data=False) if (s, V_ID[u], V_ID[v]) in flow]) == 0 
                                for s in terminals), "c1_{}".format(node))
                    
                    # Constraint 2
                    print("Constraint 2...")
                    m.addConstrs(
                        (flow[s, i, j] <= y[i, j] 
                            for s in terminals for i, j in arcs if (s, i, j) in flow), "c2")

                    # Constraint 3 - ensure tree
                    print("Constraint 3...") # node=v
                    m.addConstrs(
                        (sum([y[V_ID[u], V_ID[v]] for u, v in G.in_edges(nbunch=ID_V[node], data=False) if (V_ID[u], V_ID[v]) in y]) <= 1
                            for node in nodes), "c3")

                    # Constraint 4
                    print("Constraint 4...")
                    m.addConstrs(
                        (y[i, j] <= 1 
                            for i, j in arcs if (i, j) in y), "c4_1")
                    m.addConstrs(
                        (y[i, j] >= 0
                            for i, j in arcs if (i, j) in y), "c4_2")

                    # Constraint 5
                    print("Constraint 5...")
                    m.addConstrs(
                        (flow[s, i, j] <= 1
                            for s in terminals for i, j in arcs if (s, i, j) in flow), "c5_1")
                    m.addConstrs(
                        (flow[s, i, j] >= 0
                            for s in terminals for i, j in arcs if (s, i, j) in flow), "c5_2")

                    # for i in range(flow.shape[0]):
                        # m.addConstr(flow[i, :] <= y)

                    # Compute optimal solution
                    print("Optimize")
                    m.optimize()

                    end = timer()
                    elapsed_time = get_elapsed_time(start, end)
                    elapsed_time_array[idx_gamma, idx_beta, idx_alpha] = elapsed_time

                    # m.write("steiner_tree.lp")
                    edgelist_solution = []
                    solution = m.getAttr('x', y)
                    solution_flow = m.getAttr('x', flow)
                    #######################################
                    # Use following for loop to take max (flow) as the solution.
                    # for i, j in arcs:
                        # if (i, j) in y:
                            # solution_updated = max([solution_flow[s, i, j] for s in terminals if (s, i, j) in flow])
                            # if solution_updated > 0:
                                # weight = G.edges[ID_V[i], ID_V[j]]["weight"]
                                # edgelist_solution.append((ID_V[i], ID_V[j], {"weight":weight, "flow":solution_updated}))
                    #######################################
                    # If node weights are < 0, LP has freedom of selecting edges with negative cost
                    for i, j in arcs:
                        if (i, j) in y:
                            if solution[i,j] > 0:
                                # For negative edgeweights, the solution y will always be 1. 
                                weight = G.edges[ID_V[i], ID_V[j]]["weight"]
                                edgelist_solution.append((ID_V[i], ID_V[j], {"weight":weight, "flow":solution[i,j]}))
                                print('%s -> %s: %g' % (ID_V[i], ID_V[j], solution[i, j]))
                    #######################################
                    G_solution = nx.DiGraph()
                    # G_solution.add_weighted_edges_from(edgelist_solution)
                    G_solution.add_edges_from(edgelist_solution)
                    nx.write_graphml(G_solution, "result/synthetic_LP_month1_beta{}_gamma{}_alpha{}_v3.graphml".format(int(beta), int(gamma), int(alpha)), named_key_ids=True)
                    cost = sum([G_solution.edges[e]["weight"] * G_solution.edges[e]["flow"] for e in G_solution.edges()])
                    cost_array[idx_gamma, idx_beta, idx_alpha] = cost

                    # Save giant component, for debugging
                    largest_wcc = max(nx.weakly_connected_components(G_solution), key=len)
                    G_solution_giant = G_solution.subgraph(largest_wcc)
                    nx.write_graphml(G_solution_giant, "result/synthetic_LP_giant_month1_beta{}_gamma{}_alpha{}_v3.graphml".format(int(beta), int(gamma), int(alpha)), named_key_ids=True)

                    # if m.status == GRB.OPTIMAL:
                        # solution = m.getAttr('x', flow)
                        # for s in terminals:
                            # print('\nOptimal flows for %s:' % s)
                            # for i, j in arcs:
                                # if (s, i, j) in flow:
                                    # if solution[s, i, j] > 0:
                                        # # print('%s -> %s: %g' % (i, j, solution[s, i, j]))
                                        # print('%s -> %s: %g' % (ID_V[i], ID_V[j], solution[s, i, j]))
                    print("alpha, beta, gamma: {}, {}, {}".format(alpha, beta, gamma))
                    print("OBJ val: {:.2f}, cost(G_solution): {:.2f}".format(m.objVal, G_solution.size(weight="weight")))
                    print("Elapsed time: {:.2f} hrs".format(elapsed_time/60/60))
                    print("G_solution")
                    print(nx.info(G_solution))
                    # print("G_solution_giant")
                    # print(nx.info(G_solution_giant))

                    # get ASYMP
                    ASYMP = np.array([G.nodes[v]["ASYMP"] for v in G.nodes()])
                    # Add ASYMP_pred to G
                    for v in G:
                        if (v in G_solution) and sum([G_solution.edges[e]["flow"] for e in G_solution.in_edges(v)]) > 0.999:
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
        np.savez("npz/EXP1_LP_x{}_v3".format(int(x)),
                accuracy_array=accuracy_array,
                precision_array=precision_array,
                recall_array=recall_array,
                f1_array=f1_array,
                TP_array=TP_array,
                MCC_array=MCC_array,
                cost_array=cost_array,
                elapsed_time_array=elapsed_time_array
                )
