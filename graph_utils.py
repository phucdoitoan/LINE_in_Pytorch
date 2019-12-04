

import networkx as nx
import random

def remove_edges(graph_file, remain_name, removed_name, q_remove=50):

    G = nx.read_gpickle(graph_file)
    e_G = len(G.edges)
    print('origin edges number ', e_G)

    removed_edge_num = int(e_G*q_remove/100)
    removed_edges = random.sample(G.edges, removed_edge_num)
    G.remove_edges_from(removed_edges)

    r_G = nx.Graph()
    r_G.add_edges_from(removed_edges)

    print('removed %d edges' % (len(r_G.edges)))
    print('remained %d edges' %(len(G.edges)))

    nx.write_gpickle(G, 'data/'+remain_name+'.pkl')
    nx.write_gpickle(r_G, 'data/'+removed_name+'.pkl')



