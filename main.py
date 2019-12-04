

import torch
import numpy as np
from line_model import Line
from utils import CustomDataLoader
import pickle
import argparse
import time
import torch.optim as optim
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--K', default=5, type=int)
    parser.add_argument('--proximity', default=2, help='1 or 2', type=int)
    parser.add_argument('--learning_rate', default=0.025, type=float)
    parser.add_argument('--num_batches', default=35000, type=int)
    parser.add_argument('--graph_file', default='data/facebook_remained.pkl')
    args = parser.parse_args()

    train(args)


def train(args):
    data_loader = CustomDataLoader(graph_file=args.graph_file)
    suffix = args.proximity
    args.num_of_nodes = data_loader.num_of_nodes
    learning_rate = args.learning_rate

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    print('priximity: ', suffix, ' ', type(suffix))

    model = Line(n1 = args.num_of_nodes, dim=args.embedding_dim, order=suffix)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    #optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    sampling_time, training_time = 0, 0

    print('batches\tloss\tsampling time\ttraining_time\tdatetime')

    for b in range(args.num_batches):
        t1 = time.time()
        source_node, target_node, label = data_loader.fetch_batch(batch_size=args.batch_size, K=args.K)
        t2 = time.time()
        sampling_time = t2 - t1

        optimizer.zero_grad()
        loss = model(source_node, target_node, label)
        loss.backward()
        optimizer.step()

        training_time = time.time() - t2

        if b % 100 != 0:
            if learning_rate > args.learning_rate * 0.0001:
                learning_rate = args.learning_rate * (1 - b/args.num_batches)
            else:
                learning_rate = args.learning_rate * 0.0001
        else:
            print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

        if b % 1000 == 0:
            model.nodes_embed.data = F.normalize(model.nodes_embed.data, p=2, dim=1)

        if b == (args.num_batches - 1):
            model.nodes_embed.data = F.normalize(model.nodes_embed.data, p=2, dim=1)
            pickle.dump(model.nodes_embed.data, open('data/embedding_fb_remained_order-%s.pkl' % suffix, 'wb'))

if __name__ == '__main__':
    main()

