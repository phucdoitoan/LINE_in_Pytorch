

import networkx as nx
import random

def split_data(txt_file, full_file, remained_file, removed_file):

    G_full = nx.Graph()
    G_removed = nx.Graph()

    with open(txt_file, 'r') as file:
        for line in file:
            i, j = line.split()
            G_full.add_edge(int(i), int(j))

    #save G_full
    nx.write_gpickle(G_full, full_file)

    full_nodes = list(G_full.nodes)
    edges_num = G_full.number_of_edges()
    print('full nodes num: ', len(full_nodes))
    print('full edges num ', edges_num)

    G_removed.add_nodes_from(full_nodes)

    rv_edge_num = int(edges_num/2)

    rv_edges = random.sample(list(G_full.edges), rv_edge_num)

    G_full.remove_edges_from(rv_edges)
    print('G_remained: nodes %d, edges %d ' %(len(G_full.nodes), len(G_full.edges)))

    #save G remain
    nx.write_gpickle(G_full, remained_file)
    # save G removed
    G_removed.add_edges_from(rv_edges)
    nx.write_gpickle(G_removed, removed_file)
    print('G_removed: nodes %d, edges %d' %(len(G_removed.nodes), len(G_removed.edges)))



