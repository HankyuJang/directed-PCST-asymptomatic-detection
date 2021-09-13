"""
Description: Directed steiner tree algorithm.
Takes in the SP (shortest path dictionary), which contains shortest paths from 
(1) root to all nodes
(2) all nodes to all terminal nodes
T_{BEST} may contain path already in T, so for each terminal in T_{BEST}, walk up the path, check if any node is already in tree. If so, connect the path from that node to the terminal.
Edges may be negative.
"""

import heapq
import networkx as nx
from tqdm import tqdm
from copy import deepcopy

def d_steiner(SP, i, G, X_original, k, r, X):
    # print("d_steiner input: i:{}, k:{}, r:{}".format(i,k,r))
    # 0: if there do not exist k terminals in X reachable from r
    if less_than_k_reachable_from_r(SP, G, k, r, X):
        # print("executing line 0")
        return nx.DiGraph()
    # if i==1, find k terminals closest to the root and connect them to the root using shortest paths
    if i==1:
        # print("i=1")
        t_list = []
        sp_list = []
        sp_length_list = []
        for t in X:
            try:
                sp = SP[r][t]
                sp_length = get_path_length(G, sp)
                # print("i=1, try ")
                # print("sp_length: {}".format(sp_length))
                t_list.append(t)
                sp_list.append(sp)
                sp_length_list.append(sp_length)
            except:
                continue
        # find k shortest sp_length
        idx_length_pairs = heapq.nsmallest(k, enumerate(sp_length_list), key=lambda x:x[1])
        idx_list = [idx for idx, length in idx_length_pairs]
        top_k_sp_list = [sp_list[idx] for idx in idx_list]

        # There may be more than one shortest path btw node u and v, which may lead to non-tree.
        # Hence, generate an empty tree, then add path by usign TUT_best method.
        T_i1 = nx.DiGraph()
        for sp in top_k_sp_list:
            edgelist = []
            for idx in range(len(sp)-1):
                s = sp[idx]
                t = sp[idx+1]
                weight = G.edges[s,t]["weight"]
                edgelist.append((s, t, weight))
            T_path = nx.DiGraph()
            T_path.add_weighted_edges_from(ebunch_to_add=edgelist)
            TUT_best(T_i1, T_path, X)
            # if not nx.is_tree(T_i1):
                # print("i=1. Not Tree!!!")

        # T_i1 is the tree returned when i==1 (from root, connect k closest terminals)
        # T_i1 = nx.DiGraph()
        # T_i1.add_weighted_edges_from(ebunch_to_add=edgelist)

        return T_i1

    # 1: T <- nothing
    T = nx.DiGraph()
    # print("len(T edges): {}, k: {}, len(X): {} \n".format(len(T.edges), k, len(X)))
    # 2: while k>0

    # k_prev keeps track of previous value of k in the while loop
    while k > 0:
        # 3: T_best <- nothing
        T_best = nx.DiGraph()
        # 4: for each vertex v in V, and each k_prime, 1 <- k_prime <= k
        # print("iterations: {}".format(len(G.nodes)))
        # for v in tqdm(G.successors(r)):
        # for v in tqdm(G.nodes):
        for v in G.nodes():
            # print("Outer for")
            # print("v: {}".format(v))
            for k_prime in range(1, k+1):
                # 5: T_prime
                T_rooted_at_v = d_steiner(SP, i-1, G, X_original, k_prime, v, X)
                # print("Inner for")
                if nx.is_empty(T_rooted_at_v):
                    continue
                else:
                    edgelist = []
                    sp = SP[r][v]
                    for idx in range(len(sp)-1):
                        src = sp[idx]
                        tar = sp[idx+1]
                        # print("r: {}, v: {}, src: {}, tar: {}".format(r, v, src, tar))
                        weight = G.edges[src,tar]["weight"]
                        edgelist.append((src, tar, weight))
                    T_prime = nx.DiGraph()
                    T_prime.add_weighted_edges_from(ebunch_to_add=edgelist)
                    T_prime.update(T_rooted_at_v)
                    # print("Is T_prime a tree? {}".format(nx.is_tree(T_prime)))

                # if r==v and k==1:
                    # print("X: {}".format(X))
                    # print("T_prime nodes: {}".format(T_prime.nodes()))
                    # print("T_best nodes: {}".format(T_best.nodes()))
                    # print("density(T_best): {:.2f}, density(T_prime): {:.2f}".format(density(T_best, X_original), density(T_prime, X_original)))

                # 6: update T_best based on the density
                # print(T_prime.edges)
                # print("density(T_best): {:.2f}, density(T_prime): {:.2f}".format(density(T_best, X_original), density(T_prime, X_original)))
                if density(T_best, X_original) > density(T_prime, X_original):
                    # T_best = T_prime.copy()
                    T_best = deepcopy(T_prime)
        # 7: update T, k, X ; if T_best do not contain any terminal nodes, aka k is same as before, then return T
        V_best = set(deepcopy(T_best.nodes()))
        # k_prev = k
        k = k - len(X.intersection(V_best))
        # ensure k decreases after the two nested for loops. If not, return T
        # if k_prev == k:
            # return T

        #[TODO] 
        # T.update(deepcopy(T_best))
        TUT_best(T, T_best, X)
        # print("V(T_best): {}".format(V_best))
        # print("X: {}".format(X))
        X.difference_update(V_best)
        if not nx.is_empty(T_best):
            print("\n density: {:.2f}, |V|: {}, |E|: {}, k: {}, len(X): {} \n".format(density(T, X_original), len(T.nodes()), len(T.edges()), k, len(X)))
            # nx.write_graphml(T, "result/T_not_tree_debugging{}.graphml".format(k), named_key_ids=True)
            # nx.write_graphml(T_best, "result/T_best_not_tree_debugging{}.graphml".format(k), named_key_ids=True)
        # if not nx.is_tree(T_best):
            # print("T_best at this iteration is not a tree!")
            # print("\n density: {:.2f}, |V|: {}, |E|: {}, k: {}, len(X): {} \n".format(density(T_best, X_original), len(T_best.nodes()), len(T_best.edges()), k, len(X)))
            # nx.write_graphml(T_best, "result/T_not_tree_debugging2.graphml", named_key_ids=True)
            # break
        # print(nx.info(T))

    # 8: return T
    return T

def TUT_best(T, T_best, X):
    V_best = set(deepcopy(T_best.nodes()))
    terminals_in_T_best = X.intersection(V_best)

    edgelist = []
    for t in terminals_in_T_best:
        if t in T: # If terminal is already in the solution (when this method is called from i=1, this can occur)
            continue
        v = t
        while True:
            v_pred_list = [v_pred for v_pred in T_best.predecessors(v)]
            if len(v_pred_list) > 1:
                print("Not a tree!")
            elif not v_pred_list: #no predecessor
                break
            v_pred = v_pred_list[0]

            if v_pred in T:
                # print(v_pred)
                # print(v)
                # print(T_best.edges[v_pred, v]["weight"])
                edgelist.append((v_pred, v, T_best.edges[v_pred, v]["weight"]))
                break # get out of while loop
            else:
                # print(v_pred)
                # print(v)
                # print(T_best.edges[v_pred, v]["weight"])
                edgelist.append((v_pred, v, T_best.edges[v_pred, v]["weight"]))
            v = v_pred

    T.add_weighted_edges_from(ebunch_to_add=edgelist)

# density = cost of tree (sum of edge weights) / number of terminals
# The number of terminal only contain those in the leaves
def density(tree, X_original):
    cost_of_tree = tree.size(weight="weight")
    leaves = [node for node in tree.nodes() if tree.out_degree(node)==0 and tree.in_degree(node)==1]

    number_of_terminals = len(X_original.intersection(set(leaves)))
    if number_of_terminals == 0:
        density_of_tree = 9999999
    else:
        density_of_tree = cost_of_tree / number_of_terminals
    return density_of_tree

def density_including_all_terminals(tree, X_original):
    cost_of_tree = tree.size(weight="weight")
    
    number_of_terminals = len(X_original.intersection(set(tree.nodes())))
    if number_of_terminals == 0:
        density_of_tree = 9999999
    else:
        density_of_tree = cost_of_tree / number_of_terminals
    return density_of_tree


def less_than_k_reachable_from_r(SP, G, k, r, X):
    number_of_reachable_terminals = 0
    for t in X:
        try:
            shortest_path = SP[r][t]
            number_of_reachable_terminals += 1
        except:
            continue
    return number_of_reachable_terminals < k

# path is list of nodes, such as [1,2,3,4,5]
def get_path_length(G, path):
    n_nodes = len(path)
    path_length = 0
    for idx in range(n_nodes-1):
        path_length += G[path[idx]][path[idx+1]]["weight"]
    return path_length


