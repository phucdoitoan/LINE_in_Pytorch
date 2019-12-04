

import torch
import torch.nn as nn
import torch.nn.functional as F


class Line(nn.Module):

    def __init__(self, n1, dim, order):
        super(Line, self).__init__()
        self.n1 = n1
        self.dim = dim
        self.order = order

        self.nodes_embed = nn.Parameter(torch.zeros(n1, dim).uniform_(-1, 1))

        if self.order == 2:
            self.context_nodes_embed = nn.Parameter(torch.zeros(n1, dim).uniform_(-1, 1))


    def forward(self, source_node, target_node, label):
        """

        :param source_node: list of [i,i,i,i,i, ...] of source nodes: each source node repeat K + 1 time: one for target node, K times for K negative nodes
        :param target_node: list of [j,j1,j2,..,jK, ...] of target nodes: j is target node, j1 -> jK is negative nodes
        :param label: [1, -1, -1, -1, -1, -1, 1, ....] label to indicate which is target node, which is negative nodes
        :return:
        """

        label = torch.FloatTensor(label)

        source_embed = self.nodes_embed[source_node]

        if self.order == 1:
            target_embed = self.nodes_embed[target_node]

        elif self.order == 2:  # self.order == 2
            target_embed = self.context_nodes_embed[target_node]
        else:
            print("ERROR: order has to be 1 or 2")

        inner_product = torch.sum(torch.mul(source_embed, target_embed), dim=1)
        pos_neg = torch.mul(label, inner_product)
        log_sigmoid = F.logsigmoid(pos_neg)

        line_loss = - torch.mean(log_sigmoid)

        return line_loss
