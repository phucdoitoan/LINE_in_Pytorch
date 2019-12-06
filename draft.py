
import tensorflow as tf

import torch
from line_model import Line
from utils import CustomDataLoader


batch_size = 128
K = 5


graph_file = 'data/facebook_remained.pkl'
data_loader = CustomDataLoader(graph_file=graph_file)

source_node, target_node, label = data_loader.fetch_batch(batch_size=batch_size, K=K)
torch_label = torch.FloatTensor(label)




class Line_tf:
    def __init__(self, args):
        self.u_i = tf.placeholder(name='u_i', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        self.u_j = tf.placeholder(name='u_j', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        self.label = tf.placeholder(name='label', dtype=tf.float32, shape=[args.batch_size * (args.K + 1)])
        self.embedding = tf.get_variable('target_embedding', [args.num_of_nodes, args.embedding_dim],
                                         initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
        self.u_i_embedding = tf.matmul(tf.one_hot(self.u_i, depth=args.num_of_nodes), self.embedding)
        if args.proximity == 'first-order':
            self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=args.num_of_nodes), self.embedding)
        elif args.proximity == 'second-order':
            self.context_embedding = tf.get_variable('context_embedding', [args.num_of_nodes, args.embedding_dim],
                                                     initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
            self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=args.num_of_nodes), self.context_embedding)

        self.inner_product = tf.reduce_sum(self.u_i_embedding * self.u_j_embedding, axis=1)
        self.loss = -tf.reduce_mean(tf.log_sigmoid(self.label * self.inner_product))
        self.learning_rate = tf.placeholder(name='learning_rate', dtype=tf.float32)
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

import numpy as np
import argparse
import pickle
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', default=128)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--K', default=5)
    parser.add_argument('--proximity', default='second-order', help='first-order or second-order')
    parser.add_argument('--learning_rate', default=0.025)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--num_batches', default=300000)
    #parser.add_argument('--total_graph', default=True)
    parser.add_argument('--graph_file', default='data/facebook_remained.pkl')
    args = parser.parse_args()

    #compare_loss(args)
    return args

def compare_loss(args):
    args.num_of_nodes = data_loader.num_of_nodes

    model_torch = Line(args.num_of_nodes, args.batch_size, 2)
    loss_torch = model_torch(source_node, target_node, torch_label)
    print('loss_torch ', loss_torch)

    model_tf = Line_tf(args)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        feed_dict = {model_tf.u_i: source_node, model_tf.u_j: target_node, model_tf.label: label, model_tf.learning_rate: args.learning_rate}
        loss = sess.run(model_tf.loss, feed_dict=feed_dict)
