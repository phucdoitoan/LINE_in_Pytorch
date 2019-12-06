# try-implement-LINE

utils.py: contain CustomDataLoader -> load batch_size edges (i,j) choosed randomly from the graph
model.py: contain Line model
main.py: to train model

data directory: facebook dataset (mentioned in node2vec paper) contained in facebook_combined.pkl (4039 nodes, around 88000 edges)
The graph is removed 50% edge: remained graph saved in facebook_remained.pkl
The removed edges is saved in facebook_removed.pkl
When training in train.py, using only facebook_remained.pkl

test_evaluate_new.py: testing the model's embeddings with link prediction task. 
Calculate AUC score: positive labels: around 44000 edges in facebook_removed
negative edges: choose the same amount of around 44000 unlinked pairs of nodes in the original graph (facebook_combined)
(just change order, AUC_file, embed_file, G_full_file, G_remained_file, G_removed_file in test_evaluate_new.py to evaluate embeddings)
