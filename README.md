# try-implement-LINE

utils.py: contain CustomDataLoader -> load batch_size edges (i,j) choosed randomly from the graph

model.py: contain Line model

main.py: to train model

data directory: facebook dataset (mentioned in node2vec paper) contained in facebook_combined.pkl (4039 nodes, around 88000 edges)
The graph is removed 50% edge: remained graph saved in facebook_remained.pkl.
The removed edges is saved in facebook_removed.pkl.
When training in train.py, using only facebook_remained.pkl.

test_evaluate_new.py: testing the model's embeddings with link prediction task. 
Calculate AUC score: positive labels: around 44000 edges in facebook_removed.
Negative edges: choose the same amount of around 44000 unlinked pairs of nodes in the original graph (facebook_combined).

(just change order, AUC_file, embed_file, G_full_file, G_remained_file, G_removed_file in test_evaluate_new.py to evaluate embeddings)


***********NOTE:*************

copy the whole thing in utils from LINE-tensorflow -> model run correctly with loss reduce to near 0 (0.00..) (for 2nd order)
old utils (the one inside "") => even though after 66000 batches, model's loss still stuck around 0.2 0.3 (dont know why)

=> AUC only around 0.56
while embed with tensorlow -> AUC of 0.77 and 0.87
WHY???

only with 10,000 batches LINE-tensorflow
has already given very good embedding with AUC around 0.75 and 0.89 (even > than convergence AUC for 2nd-order)
(after 36000 batches order-1 also achieve 0.77 AUC as convergence)



TWO MODEL DEFINITELY DIFFERENT:
pytorch is really smelly:

initalial (-1,1): lr = 10 and 0.025
 pytorch gives almost same values of loss (check first several batches)
 while tensorflow do give smaller loss for bigger lr

initial (-100, 100):
  lr = 0.25
    first loss are different (although small) : torch 14577.215820 , tf 14577.213867;
    after that loss are different torch 14245. vs tf 14231; torch 15617 vs 15608

  lr = 10
    first loss same as lr = 0.025: torch 14577.215820 , tf 14577.213867
    1nd batch loss: torch 14245. vs tf 9548.
    2rd batch loss: torch 15610. vs tf 12086.
    3rd: torch 13519. tf 10643.
    4th: 15300. vs tf 13130.
