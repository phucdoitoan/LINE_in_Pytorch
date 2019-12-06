# try-implement-LINE

utils.py: contain CustomDataLoader -> load batch_size edges (i,j) choosed randomly from the graph

model.py: contain Line model

main.py: to train model

data directory: facebook dataset (mentioned in node2vec paper) contained in facebook_combined.pkl (4039 nodes, 88234 edges)
The graph is removed 50% edge: remained graph saved in facebook_remained.pkl.
The removed edges is saved in facebook_removed.pkl.
When training in train.py, using only facebook_remained.pkl.

test_evaluate_new.py: testing the model's embeddings with link prediction task. 
Calculate AUC score: positive labels: 44117 edges in facebook_removed.
Negative edges: choose the same amount of edges as in positive label: 44117 unlinked pairs of nodes in the original graph (facebook_combined).

(just change order, AUC_file, embed_file in test_evaluate_new.py to evaluate embeddings)

Embedding by tensorflow version after only 20,000 batches already give AUC close to AUC after run 300,000 batches: 1st-order: 0.76; 2nd-order: 0.87
Embedding by this pytorch version after 300,000 batches give AUC about random: 0.53, 0.55 (tried embedding with smaller number of batches give the same bad AUC scores)
