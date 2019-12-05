

import torch
import numpy as np
from line_model import Line
from utils import CustomDataLoader
import pickle
#import argparse
import time
import torch.optim as optim
import torch.nn.functional as F


def train():

    embedding_dim = 128
    batch_size = 128
    K = 5
    order = 2
    learning_rate = 0.025
    num_batches = 300000
    graph_file = 'data/facebook_remained.pkl'


    data_loader = CustomDataLoader(graph_file=graph_file)
    num_of_nodes = data_loader.num_of_nodes
    print('number of nodes ', num_of_nodes)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = Line(n1=num_of_nodes, dim=embedding_dim, order=order)
    model.to(device)

    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    #def get_lr():
    #    for param_group in optimizer.param_groups:
    #        return param_group['lr']

    sampling_time, training_time = 0, 0

    print('batches\tloss\tsampling time\ttraining_time\tdatetime')

    for b in range(num_batches):
        t1 = time.time()
        source_node, target_node, label = data_loader.fetch_batch(batch_size=batch_size, K=K)
        label = torch.FloatTensor(label).to(device)
        t2 = time.time()
        sampling_time += t2 - t1

        optimizer.zero_grad()
        loss = model(source_node, target_node, label)
        loss.backward()
        optimizer.step()

        training_time += time.time() - t2

        """WARNING: is this the right way to update lr of optimizer?"""
        if b % 100 != 0:
            #print('source_node target_node')
            #print('size ', len(source_node), len(target_node), len(label))
            #print(source_node[:12])
            #print(target_node[:12])
            #print(label[:12])

            for param_group in optimizer.param_groups:
                lr = param_group['lr']

            if lr > learning_rate * 0.0001:
                lr = learning_rate * (1 - b/num_batches)
            else:
                lr = learning_rate * 0.0001

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        else:
            print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            #print('optimizer lr: ', get_lr())
            sampling_time, training_time = 0, 0

        if b % 1000 == 0 or b == (num_batches - 1):
            embedding = model.nodes_embed.data  # embedding.requires_grad : False
            embedding = F.normalize(embedding, p=2, dim=1)
            pickle.dump(embedding.to('cpu'), open('data/embedding=pytorch_fb_remained_order-%s.pkl' % order, 'wb'))


train()

