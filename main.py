

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
    num_batches = 10 #300000
    graph_file = 'data/facebook_remained.pkl'


    data_loader = CustomDataLoader(graph_file=graph_file)
    num_of_nodes = data_loader.num_of_nodes
    #print('number of nodes ', num_of_nodes)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = Line(n1=num_of_nodes, dim=embedding_dim, order=order)
    model.to(device)

    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    #optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #def get_lr():
    #    for param_group in optimizer.param_groups:
    #        return param_group['lr']

    sampling_time, training_time = 0, 0

    print('batches\tloss\tsampling time\ttraining_time\tdatetime')

    for b in range(num_batches):

        print('******************* Batch %d *************************' %b)

        t1 = time.time()
        source_node, target_node, label = data_loader.fetch_batch(batch_size=batch_size, K=K)
        label = torch.FloatTensor(label).to(device)
        t2 = time.time()
        sampling_time += t2 - t1

        if b % 100 != 0:

            optimizer.zero_grad()
            loss = model(source_node, target_node, label)
            loss.backward()
            optimizer.step()

            training_time += time.time() - t2

            print('nodes_embed GRAD VAL ', model.nodes_embed.grad == 0)
            print('context_nodes_embed GRAD VAL ', model.context_nodes_embed.grad == 0)

            """This is for updating learning rate of optimizer: same with tensorflow one: checked"""
            for param_group in optimizer.param_groups:
                lr = param_group['lr']

            if lr > learning_rate * 0.0001:
                lr = learning_rate * (1 - b/num_batches)
            else:
                lr = learning_rate * 0.0001

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        else:
            with torch.no_grad():
                loss1 = model(source_node, target_node, label)
                print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss1, sampling_time, training_time,
                                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            #print('optimizer lr: ', get_lr())
            sampling_time, training_time = 0, 0

        if (b != 0) and (b % 10000 == 0 or b == (num_batches - 1)):
            embedding = model.nodes_embed.data  # embedding.requires_grad : False
            normalized_embedding = F.normalize(embedding, p=2, dim=1)
            pickle.dump(normalized_embedding.to('cpu'), open('data/embedding_Adam=pytorch_fb_remained_order-%s.pkl' % order, 'wb'))


train()

