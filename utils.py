

import networkx as nx
import numpy as np



class CustomDataLoader:

    def __init__(self, graph_file):
        self.G = nx.read_gpickle(graph_file)
        self.num_of_nodes = self.G.number_of_nodes()
        self.num_of_edges = self.G.number_of_edges()
        self.edges_raw = self.G.edges(data=True)
        self.nodes_raw = self.G.nodes(data=True)

        try:
            self.edge_distribution = np.array([attr['weight'] for _, _, attr in self.edges_raw])
        except:
            self.edge_distribution = np.ones(len(self.edges_raw))

        self.edge_distribution /= np.sum(self.edge_distribution)
        self.edge_sampling = AliasSampling(prob=self.edge_distribution)

        self.node_negative_distribution = np.power(np.array([self.G.degree(node, weight='weight') for node, _ in self.nodes_raw]), 0.75)
        self.node_negative_distribution /= np.sum(self.node_negative_distribution)
        self.node_sampling = AliasSampling(prob=self.node_negative_distribution)

        self.edges = list(self.G.edges)
        self.nodes = list(self.G.nodes)

    def fetch_batch(self, batch_size=16, K=10):
        edge_batch_index = self.edge_sampling.sampling(batch_size)

        source_node = []
        target_node = []
        label = []

        for edge_index in edge_batch_index:
            edge = self.edges[edge_index]
            if self.G.__class__ == nx.Graph:
                if np.random.rand() > 0.5:  # important: randomly decide which node is source node and target node for a undirected edge
                    edge = (edge[1], edge[0])
            source_node.append(edge[0])
            target_node.append(edge[1])
            label.append(1)

            for i in range(K):
                while True:
                    negative_node = self.node_sampling.sampling()
                    if not self.G.has_edge(self.nodes[edge[0]], self.nodes[negative_node]):
                        break

                source_node.append(edge[0])
                target_node.append(negative_node)
                label.append(-1)

        return source_node, target_node, label



class AliasSampling:

    # Reference: https://en.wikipedia.org/wiki/Alias_method

    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res












