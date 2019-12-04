

import networkx as nx
import pickle
import torch
import matplotlib.pyplot as plt
import timeit
from time import sleep


def numerical_integral(l_x, l_y):
    integral = 0
    for i in range(len(l_x)-1):
        sum_i = (l_x[i+1]-l_x[i])*(l_y[i+1]+l_y[i]) / 2
        integral += sum_i

    return integral


def predict_new_edge(dist_c, threshold) -> 'tensor of size n*n contain 0-1':
    """
    :param dist_c: matrix of distance between unlinked nodes only ( (i,j) = 0 if i = j or i is linked with j)
    :param adj_c: adj matrix of complement graph : (i,j) is unlinked -> adj_c(i,j) = 1
    :param threshold: threshold to suggest new link
    :return: tensor of new edges
    """

    # point wise distance matrix
    #dist_matrix = torch.sum(torch.abs(X.unsqueeze(1) - X.unsqueeze(0))**2, dim=2)

    # only keep distance between unlinked nodes
    #dist_c = torch.mul(dist_matrix, adj_c)

    # fill out 0: make unlinked (i,j) to have value 1, linked (i,j) or (i,i) to have value 0
    dist_c_pos = 0 < dist_c
    """WARNING: maybe there are unlinked nodes i, j 
    which have embed_i = embed_j -> dist_c(i,j) = 0 but this is rarely happened -> so ignored"""

    # make (i,j) with distance smaller than threshold to have value 1
    dist_c_threshold = dist_c < threshold

    # (i,j) (j,i) = 1 only if 0 < dist(i, j) < threshold; (i,j) unlinked
    predict = (torch.mul(dist_c_pos, dist_c_threshold)).type(torch.FloatTensor)

    """predict: tensor of size (n*n) contain 0-1:
     1 on the predicted edge only: (i, j) (j, i) will be 1 if (i,j) is predicted to be an edge"""
    return predict


def tpr_fpr (positive_edge, predict_edge, total_unlinked_edge) -> ('true pos rate', 'false pos rate'):

    pos_num = torch.sum(positive_edge).item()/2
    """WARNING: only for undirected graph: 
    divided by 2 as edge (i,j) appear twice at (i,j) and (j,i)"""
    neg_num = total_unlinked_edge - pos_num

    true_pos_num = torch.sum(torch.mul(positive_edge, predict_edge))/2
    false_pos_num = torch.sum(predict_edge)/2 - true_pos_num
    """divide by 2 as predict_edge_set repeat same edge twice as (i,j) and (j,i)"""

    true_pos_rate = true_pos_num / pos_num
    false_pos_rate = false_pos_num / neg_num

    return true_pos_rate, false_pos_rate


def evaluate(remained_G_file, removed_G_file, embed_file, AUC_file):
    threshold_step = 0.01

    with open(embed_file, 'rb') as file:
        nodes_embed = pickle.load(file)

    remained_G = nx.read_gpickle(remained_G_file)
    n = len(remained_G.nodes)

    removed_G = nx.read_gpickle(removed_G_file)
    rm_edge_set = set(removed_G.edges)
    removed_G.add_nodes_from(remained_G.nodes)
    # all positive example (i,j) (j,i) will have value 1
    positive_edge = (torch.from_numpy(nx.adjacency_matrix(removed_G).todense())).type(torch.FloatTensor)


    total_unlinked_edges = n*(n-1)/2 - len(remained_G.edges)

    adj = torch.from_numpy(nx.adjacency_matrix(remained_G).todense())
    adj_c = (adj == 0).type(torch.FloatTensor) - torch.eye(n)

    TPR = []
    FPR = []

    threshold = 0.0

    # point wise distance matrix
    """WARNING: calculate dist_matrix take alot of memory space: about n*n*dim*24 bytes"""
    dist_matrix = torch.sum(torch.abs(nodes_embed.unsqueeze(1) - nodes_embed.unsqueeze(0)) ** 2, dim=2)

    # only keep distance between unlinked nodes
    dist_c = torch.mul(dist_matrix, adj_c)

    it_num = 150
    for i in range(it_num):
        threshold += i*threshold_step
        predict_edge = predict_new_edge(dist_c, threshold)
        tpr, fpr = tpr_fpr(positive_edge, predict_edge, total_unlinked_edges)
        TPR.append(tpr)
        FPR.append(fpr)
        print('done %d / %d iter ' % (i, it_num))

    AUC = numerical_integral(FPR, TPR)

    fig = plt.figure()
    plt.plot([0.0, 1.0], [0.0, 1.0], 'k--')
    label = 'AUC score: %.4f' % (AUC)
    plt.plot(FPR, TPR, 'b-', label=label)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(AUC_file, bbox_inches='tight')



t1 = timeit.default_timer()
evaluate('data/facebook_remained.pkl', 'data/facebook_removed.pkl', 'data/embedding_facebook_remained_2.pkl',
         'data/AUC_facebook_LINE-tensorflow_second-order.png')
print('evaluate in %.2f s' %(timeit.default_timer() - t1))
