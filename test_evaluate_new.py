
import networkx as nx
import pickle
from evaluate_new import evaluate
import torch
import timeit
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


def numerical_integral(l_x, l_y):
    integral = 0
    for i in range(len(l_x)-1):
        sum_i = (l_x[i+1]-l_x[i])*(l_y[i+1]+l_y[i]) / 2
        integral += sum_i

    return integral

def main(order=1, AUC_file='data/auc/AUC_tensorflow_fb_order=1.png', embed_file= 'data/embedding=tensorflow_fb_remained_order-1.pkl', G_full_file='data/facebook_combined.pkl', G_remained_file='data/facebook_remained.pkl', G_removed_file='data/facebook_removed.pkl'):
    G_full = nx.read_gpickle(G_full_file)
    #G_remained = nx.read_gpickle(G_remained_file)
    G_removed = nx.read_gpickle(G_removed_file)

    print('Loaded graphs')

    with open(embed_file, 'rb') as file:
        embeds = pickle.load(file)

    print('Loaded embeds')

    t1_neg = timeit.default_timer()
    negative_set = set()
    for i in tqdm(range(len(G_removed.edges))):
    #while len(negative_set) < len(G_removed.edges):
        #print(len(negative_set))
        #while True:
        while len(negative_set) < (i+1):
            x = random.sample(list(G_full.nodes),1)[0]
            y = random.sample(list(G_full.nodes),1)[0]
            #print('negative edges (%d, %d)' %(x, y))
            if (x, y) not in G_full.edges:
                negative_set.add((x, y))
               #break

    t2_neg = timeit.default_timer()
    print('Done randomly choosing negative edges: %.2f s' %(t2_neg - t1_neg))

    positive_set = set(G_removed.edges)
    intersec = positive_set.intersection(negative_set)
    uni = positive_set.union(negative_set)
    print('len postive and negative intersec ', len(intersec))
    print('len pos and neg union ', len(uni))

    false_edges = list(negative_set)
    true_edges = list(G_removed.edges)

    print('len false edges: ', len(false_edges))
    print('len true edges: ', len(true_edges))

    t1 = timeit.default_timer()
    auc_score, f1_score, auc, fpr, tpr = evaluate(embeds, true_edges, false_edges)
    t2 = timeit.default_timer()
    print('Evaluate in %.2f s' %(t2-t1))

    print('Order: ', order)
    print('   auc_score: ', auc_score)
    print('   auc: ', auc)

    #auc_manual = numerical_integral(fpr, tpr)

    fig = plt.figure()
    plt.plot([0.0, 1.0], [0.0, 1.0], 'k--')
    label = 'arXiv order-%s AUC: %.4f' % (str(order), auc_score)
    plt.plot(fpr, tpr, 'b-', label=label)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(AUC_file, bbox_inches='tight')


    return auc_score, f1_score, auc


order=1
AUC_file= 'data/arXiv/AUC_arxiv_Adam_order=1-510000-batch-1.png'
embed_file= 'data/arXiv/embedding_Adam-arxiv_remained_1-510000-batch.pkl'
G_full_file='data/arXiv/arxiv_full.pkl'
G_remained_file='data/arXiv/arxiv_remained.pkl'
G_removed_file='data/arXiv/arxiv_removed.pkl'


main(order, AUC_file, embed_file, G_full_file, G_remained_file, G_removed_file)

order=1
AUC_file= 'data/facebook/AUC_Adam_pytorch_facebook_order=1-0-batch.png'
embed_file= 'data/facebook/embedding_Adam=torch-facebook_remained_1-0-batch.pkl'
G_full_file='data/facebook/facebook_combined.pkl'
G_remained_file='data/facebook/facebook_remained.pkl'
G_removed_file='data/facebook/facebook_removed.pkl'