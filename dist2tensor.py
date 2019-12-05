

import torch
import pickle


def dist2tensor(dist_file, tensor_file):

    with open(dist_file, 'rb') as file:
        dist = pickle.load(file)

    n = len(dist)
    dim = len(dist[0])

    tensor = torch.zeros(n, dim)

    for i in range(n):
        tensor[i] = torch.from_numpy(dist[i])

    with open(tensor_file, 'wb') as file:
        pickle.dump(tensor, file)

