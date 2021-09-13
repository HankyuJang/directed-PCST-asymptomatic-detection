import numpy as np
import networkx as nx

def generate_graph_statistics_directed(G):
    n = nx.number_of_nodes(G)
    m = nx.number_of_edges(G)

    in_degree_sequence = sorted([d for n, d in G.in_degree()])
    k_in_mean = np.mean(in_degree_sequence)
    k_in_max = np.max(in_degree_sequence)
    k_in_std = np.std(in_degree_sequence)

    out_degree_sequence = sorted([d for n, d in G.out_degree()])
    k_out_mean = np.mean(out_degree_sequence)
    k_out_max = np.max(out_degree_sequence)
    k_out_std = np.std(out_degree_sequence)

    wc = nx.number_weakly_connected_components(G)
    sc = nx.number_strongly_connected_components(G)
    # cc takes a long time to compute
    # cc = nx.average_clustering(G)
    # c = nx.number_connected_components(G)
    # assortativity = nx.degree_assortativity_coefficient(G)

    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    # largest_cc = max(nx.connected_components(G), key=len)
    G_giant = G.subgraph(largest_wcc)
    n_giant = nx.number_of_nodes(G_giant) 
    m_giant = nx.number_of_edges(G_giant) 

    return in_degree_sequence, out_degree_sequence, [n, m, k_in_mean, k_in_max, k_in_std, k_out_mean, k_out_max, k_out_std, wc, sc, n_giant, m_giant]

def generate_graph_statistics(G):
    n = nx.number_of_nodes(G)
    m = nx.number_of_edges(G)

    degree_sequence = np.array([d for n, d in nx.degree(G)])
    k_mean = degree_sequence.mean()
    k_max = degree_sequence.max()
    std = np.std(degree_sequence)
    cc = nx.average_clustering(G)
    c = nx.number_connected_components(G)
    assortativity = nx.degree_assortativity_coefficient(G)

    largest_cc = max(nx.connected_components(G), key=len)
    G_giant = G.subgraph(largest_cc)
    n_giant = nx.number_of_nodes(G_giant) 
    m_giant = nx.number_of_edges(G_giant) 

    return [n, m, k_mean, k_max, std, cc, c, assortativity, n_giant, m_giant]

def generate_graph_statistics_v2(G):
    n = nx.number_of_nodes(G)
    m = nx.number_of_edges(G)

    degree_sequence = np.array([d for n, d in nx.degree(G)])
    k_mean = degree_sequence.mean()
    k_max = degree_sequence.max()
    std = np.std(degree_sequence)
    clustering_coefficient = nx.average_clustering(G)
    n_connected_components = nx.number_connected_components(G)
    assortativity = nx.degree_assortativity_coefficient(G)

    largest_cc = max(nx.connected_components(G), key=len)
    G_giant = G.subgraph(largest_cc)
    n_giant = nx.number_of_nodes(G_giant) 
    m_giant = nx.number_of_edges(G_giant) 

    nodes_cc = np.array([c for c in sorted(nx.connected_components(G), key=len, reverse=True)])
    n_cc = np.zeros((n_connected_components))
    m_cc = np.zeros((n_connected_components))
    for i, cc in enumerate(nodes_cc):
        G_cc = G.subgraph(cc)
        n_cc[i] = G_cc.number_of_nodes()
        m_cc[i] = G_cc.number_of_edges()

    mean_n_cc = np.mean(n_cc)
    std_n_cc = np.std(n_cc)
    mean_m_cc = np.mean(m_cc)
    std_m_cc = np.std(m_cc)

    return [n, m, k_mean, k_max, std, clustering_coefficient, n_connected_components, assortativity, n_giant, m_giant, mean_n_cc, std_n_cc, mean_m_cc, std_m_cc]

def generate_graph_density(G):
    n = nx.number_of_nodes(G)
    m = nx.number_of_edges(G)

    size_cc = np.array([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)])

    density0 = nx.density(G)
    density1 = m / n
    density2 = np.max(size_cc) / n
    density3 = np.mean(size_cc) / n

    return density0, density1, density2, density3
