

import torch
import torch.nn as nn
import torch.nn.functional as F


class Line(nn.Module):

    def __init__(self, n1, dim, order):
        super(Line, self).__init__()
        self.n1 = n1
        self.dim = dim
        self.order = order

        self.nodes_embed = nn.Parameter(torch.zeros(n1, dim).uniform_(-1, 1), requires_grad=True)
        #self.nodes_embed = nn.Embedding(n1, dim)
        #self.nodes_embed.weight.data = self.nodes_embed.weight.data.uniform_(-1, 1)

        if self.order == 2:
            self.context_nodes_embed = nn.Parameter(torch.zeros(n1, dim).uniform_(-1, 1), requires_grad=True)
            #self.context_nodes_embed = nn.Embedding(n1, dim)
            #self.context_nodes_embed.weight.data = self.context_nodes_embed.weight.data.uniform_(-1, 1)

    def forward(self, source_node, target_node, label):
        """

        :param source_node: list of [i,i,i,i,i, ...] of source nodes: each source node repeat K + 1 time: one for target node, K times for K negative nodes
        :param target_node: list of [j,j1,j2,..,jK, ...] of target nodes: j is target node, j1 -> jK is negative nodes
        :param label: FloatTensor([1, -1, -1, -1, -1, -1, 1, ....]) label to indicate which is target node, which is negative nodes
        :return:
        """

        #label = torch.FloatTensor(label)
        #print('size source_node, target_node, label ', len(source_node), len(target_node), len(label))

        source_embed = self.nodes_embed[source_node]
        #print('source_embed shape: ', source_embed.shape)

        if self.order == 1:
            target_embed = self.nodes_embed[target_node]
            #print('target_embed shape: ', target_embed.shape)

        elif self.order == 2:  # self.order == 2
            target_embed = self.context_nodes_embed[target_node]
            #print('target_embed shape: ', target_embed.shape)
        else:
            print("ERROR: order has to be 1 or 2")

        inner_product = torch.sum(torch.mul(source_embed, target_embed), dim=1)
        #print('inner_product shape: ', inner_product.shape)
        pos_neg = torch.mul(label, inner_product)
        line_loss = F.logsigmoid(pos_neg)

        mean_loss = - torch.mean(line_loss)
        return mean_loss
