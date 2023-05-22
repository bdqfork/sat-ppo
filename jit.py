from sat.entity import CNF
from sat.spaces import Graph
import torch
import numpy as np
from models.jitgnn import JitGNN
from models.gnn import GNN
from sat.utils import batch_graphs
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Wrapper(nn.Module):
    """Some Information about Wrapper"""

    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.model = model

    def forward(self, vbatched,
                ebatched,
                conn,
                variable_nodes_sizes,
                v_graph_belonging,
                e_graph_belonging,
                gbatched,
                bbatched):
        policy, value = self.model((vbatched,
                                    ebatched,
                                    conn,
                                    variable_nodes_sizes,
                                    v_graph_belonging,
                                    e_graph_belonging,
                                    gbatched,
                                    bbatched))
        return policy.flatten(), value.flatten()


def main(args):
    cnf = CNF('data/satlab/100/sat_000000.npz')
    graph = cnf.to_graph([-69, -359, -247, 199])
    cnf.write("test.cnf")
    graph = [torch.tensor(el) for el in graph]
    graph = batch_graphs([graph])
    model2 = GNN.factory().create_eval(Graph(dtype=np.float, shape=[2, 2, 1]), Graph(
        dtype=np.float, shape=(1,)))
    checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
    model2.load_state_dict(checkpoint['net'])
    print(model2([graph]))
    graph = list(graph)
    graph = graph[:-1]
    embeds = torch.tensor([-69, -359, -247, 199]).long()
    graph[-1] = embeds
    model = JitGNN.factory().create_eval(Graph(dtype=np.float, shape=[2, 2, 1]), Graph(
        dtype=np.float, shape=(1,)))
    model.load_state_dict(checkpoint['net'])
    wrapper = Wrapper(model)
    traced = torch.jit.trace(wrapper, graph)
    print(wrapper(graph[0], graph[1], graph[2], graph[3],
          graph[4], graph[5], graph[6], graph[7]))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    index = args.model_path.rfind('/')
    if index != -1:
        save_path = f'{args.save_path}/{args.model_path[index + 1:]}'
    else:
        save_path = f'{args.save_path}/{args.model_path}'

    traced.save(save_path)
    print('Done')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model_path', type=str, default='model/sat/150/model-17569.pt',
                        help='the model path to load (default: model/sat/150/model-15099.pt)')
    parser.add_argument('-r', '--save_path', type=str, default='model/jit/sat/150',
                        help='the model path to save (default: model/jit/sat/150)')
    args = parser.parse_args()

    print(args)

    main(args)
