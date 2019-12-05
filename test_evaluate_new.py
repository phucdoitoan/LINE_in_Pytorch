
import networkx as nx
import pickle
from evaluate_new import evaluate
import torch
import timeit
import matplotlib.pyplot as plt


def numerical_integral(l_x, l_y):
    integral = 0
    for i in range(len(l_x)-1):
        sum_i = (l_x[i+1]-l_x[i])*(l_y[i+1]+l_y[i]) / 2
        integral += sum_i

    return integral

def main(order=1, AUC_file='AUC_fb_order=1.png', embed_file= 'data/embedding_facebook_remained_1.pkl', G_full_file='data/facebook_combined.pkl', G_remained_file='data/facebook_remained.pkl', G_removed_file='data/facebook_removed.pkl'):
    G_full = nx.read_gpickle(G_full_file)
    #G_remained = nx.read_gpickle(G_remained_file)
    G_removed = nx.read_gpickle(G_removed_file)

    print('Loaded graphs')

    with open(embed_file, 'rb') as file:
        embeds = pickle.load(file)

    print('Loaded embeds')

    negative_set = set()
    while len(negative_set) < len(G_removed.edges):
        while True:
            x = torch.randint(len(G_full.nodes), (1,)).item()
            y = torch.randint(len(G_full.nodes), (1,)).item()
            if (x, y) not in G_full.edges:
                negative_set.add((x, y))
                break

    print('Done randomly choosing negative edges')

    false_edges = list(negative_set)
    true_edges = list(G_removed.edges)

    t1 = timeit.default_timer()
    auc_score, f1_score, auc, fpr, tpr = evaluate(embeds, true_edges, false_edges)
    t2 = timeit.default_timer()
    print('Evaluate in %.2f s' %(t2-t1))

    #auc_manual = numerical_integral(fpr, tpr)

    fig = plt.figure()
    plt.plot([0.0, 1.0], [0.0, 1.0], 'k--')
    label = 'fb-LINE-tensorflow AUC %s: %.4f' % (str(order), auc_score)
    plt.plot(fpr, tpr, 'b-', label=label)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(AUC_file, bbox_inches='tight')


    return auc_score, f1_score, auc