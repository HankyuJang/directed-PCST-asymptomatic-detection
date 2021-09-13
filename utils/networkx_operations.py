from ast import literal_eval as make_tuple
from tqdm import tqdm
from utils.steiner_tree_te import *
import networkx as nx
import itertools
import copy

def relabel_nodes_str_to_tuple(G):
    node_list = list(G.nodes)
    relable_node_list = []
    for node in node_list:
        relable_node_list.append(make_tuple(node))
    mapping_dict = {}
    for node, relable_node in zip(node_list, relable_node_list):
        mapping_dict[node] = relable_node
    G_relabelled = nx.relabel.relabel_nodes(G, mapping_dict)
    return G_relabelled

# This dummy node gets connected to all other nodes.
# For directed graphs, direction is from dummy node to all other nodes
def add_dummy_node(G, r, node_prob, gamma):
    G.add_nodes_from([(r, {"prob":node_prob})])
    edges_to_add = []
    for v in G.nodes:
        if r != v:
            edges_to_add.append((r, v, gamma))
    G.add_weighted_edges_from(ebunch_to_add=edges_to_add)

# This SP computation can handle negative weights (but no negative cycles)
def compute_SP_r_to_all_and_all_to_X(G, r, X):
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
    return SP

# From the input graph, generate metric graph
# Input is G and set of nodes to include in the metric graph.
# The distance between nodes in the metric graph is the actual distance between two nodes in the real graph
# We need shortest path btw root to all terminals, and all terminals to all other terminals.
def gen_metric_graph(G, SP, metric_node_set):
    G_metric = nx.DiGraph()
    candidate_edges = itertools.permutations(metric_node_set, 2)
    edgelist = []
    for u, v in candidate_edges:
        try:
            sp = SP[u][v]
            sp_length = get_path_length(G, sp)
            edgelist.append((u, v, sp_length))
        except:
            continue
    G_metric.add_weighted_edges_from(ebunch_to_add=edgelist)
    return G_metric

# This algorithm assumes that G do not have cycles
def get_min_cost_arborescence(G_metric, r):
    edgelist = []
    for v in G_metric.nodes():
        if v == r:
            continue
        # line1
        min_cost_edge = 99999999.0
        for edge in G_metric.in_edges(v):
            if G_metric.edges[edge]["weight"] < min_cost_edge:
                min_cost_edge = G_metric.edges[edge]["weight"]
        # line2
        for edge in G_metric.in_edges(v):
            G_metric.edges[edge]["weight"] -= min_cost_edge
        # line3
        for edge in G_metric.in_edges(v):
            if -0.00001< G_metric.edges[edge]["weight"] < 0.00001:
                edgelist.append((edge[0], edge[1], 0.0))
    MCA = nx.DiGraph()
    MCA.add_weighted_edges_from(ebunch_to_add=edgelist)
    return MCA

# This method ensures the solution is tree
def metric_to_original_ensure_tree(G, MCA_metric, SP, X):
    T = nx.DiGraph()
    for u, v in MCA_metric.edges():
        edgelist = []
        sp = SP[u][v]
        # Walk up the path, and stop adding nodes once a node is already in the tree
        for idx in range(len(sp)-2, -1, -1):
            s = sp[idx]
            t = sp[idx+1]
            weight = G.edges[s,t]["weight"]
            edgelist.append((s, t, weight))
            if s in T:
                break
        T.add_weighted_edges_from(ebunch_to_add=edgelist)
    return T

# This method makes the tree disconnected
def metric_to_original(G, MCA_metric, SP, X):
    T = nx.DiGraph()
    for u, v in MCA_metric.edges():
        edgelist = []
        sp = SP[u][v]
        for idx in range(len(sp)-1):
            s = sp[idx]
            t = sp[idx+1]
            weight = G.edges[s,t]["weight"]
            edgelist.append((s, t, weight))
        T_path = nx.DiGraph()
        T_path.add_weighted_edges_from(ebunch_to_add=edgelist)
        TUT_best(T, T_path, X)
    return T

# This method simply adds paths
def metric_to_original_v2(G, MCA_metric, SP, X):
    T = nx.DiGraph()
    for u, v in MCA_metric.edges():
        edgelist = []
        sp = SP[u][v]
        for idx in range(len(sp)-1):
            s = sp[idx]
            t = sp[idx+1]
            weight = G.edges[s,t]["weight"]
            edgelist.append((s, t, weight))
        T_path = nx.DiGraph()
        T_path.add_weighted_edges_from(ebunch_to_add=edgelist)
        # TUT_best(T, T_path, X)
        T.update(deepcopy(T_path))
    return T
