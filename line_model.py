

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Line(nn.Module):

    np.random.seed(42)

    def __init__(self, n1, dim, order):
        super(Line, self).__init__()
        self.n1 = n1
        self.dim = dim
        self.order = order

        nodes_init = np.random.uniform(-100, 100, (n1, dim)).astype(np.float32)

        self.nodes_embed = nn.Parameter(torch.from_numpy(nodes_init), requires_grad=True)

        if self.order == 2:
            context_init = np.random.uniform(-100, 100, (n1, dim)).astype(np.float32)

            self.context_nodes_embed = nn.Parameter(torch.from_numpy(context_init), requires_grad=True)



    def forward(self, source_node, target_node, label):
        """

        :param source_node: list of [i,i,i,i,i, ...] of source nodes: each source node repeat K + 1 time: one for target node, K times for K negative nodes
        :param target_node: list of [j,j1,j2,..,jK, ...] of target nodes: j is target node, j1 -> jK is negative nodes
        :param label: FloatTensor([1, -1, -1, -1, -1, -1, 1, ....]) label to indicate which is target node, which is negative nodes
        :return:
        """

        #label = torch.FloatTensor(label)
        #print('size source_node, target_node, label ', len(source_node), len(target_node), len(label))

        #print('SOURCE NODES ', type(source_node), len(source_node), source_node)

        print('nodes_embed grad ', self.nodes_embed.requires_grad)

        source_embed = self.nodes_embed[source_node]
        print('source_embed grad: ', source_embed.requires_grad)
        #results: false => so no grad flow back through source_embed

        if self.order == 1:
            target_embed = self.nodes_embed[target_node]
            print('1st order: target_embed grad ', target_embed.requires_grad)

        elif self.order == 2:  # self.order == 2
            print('context_nodes_embed grad ', self.context_nodes_embed.requires_grad)
            target_embed = self.context_nodes_embed[target_node]
            print('2nd order: target_embed grad ', target_embed.requires_grad)
        else:
            print("ERROR: order has to be 1 or 2")

        inner_product = torch.sum(torch.mul(source_embed, target_embed), dim=1)
        print('inner_product grad ', inner_product.requires_grad)
        pos_neg = torch.mul(label, inner_product)
        print('pos_neg grad ', pos_neg.requires_grad)
        logsig_loss = F.logsigmoid(pos_neg)
        print('logsig_loss grad ', logsig_loss.requires_grad)

        mean_loss = - torch.mean(logsig_loss)
        print('mean_loss grad ', mean_loss.grad)

        #print('   source_node ', source_node)
        #print('   target_node ', target_node)
        #print('   inner product: ', inner_product)
        #print('   pos_neg: ', pos_neg)
        #print('   logsig_loss: ', logsig_loss)

        return mean_loss
