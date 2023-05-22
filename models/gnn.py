# The code in this file was originally copied from the Pytorch Geometric library and modified later:
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/meta.html#MetaLayer
# Pytorch geometric license is below

# Copyright (c) 2019 Matthias Fey <matthias.fey@tu-dortmund.de>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LayerNorm
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn.meta import MetaLayer
import numpy as np
from torch import nn, Tensor
from torch.utils.data import Dataset

from envs import Converter
from models.datasets import NonSequentialDataset
from models.model import Model, ModelFactory
import torch.nn.functional as F


class ModifiedMetaLayer(MetaLayer):
    def forward(
        self, x, edge_index, edge_attr=None, u=None, v_indices=None, e_indices=None
    ):
        row, col = edge_index

        if self.edge_model is not None:
            edge_attr = self.edge_model(
                x[row], x[col], edge_attr, u, e_indices)

        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr, u, v_indices)

        if self.global_model is not None:
            u = self.global_model(x, edge_attr, u, v_indices, e_indices)

        return x, edge_attr, u


def get_mlp(
    in_size,
    out_size,
    n_hidden,
    hidden_size,
    activation=nn.LeakyReLU,
    activate_last=True,
    layer_norm=True,
):
    arch = []
    l_in = in_size
    for l_idx in range(n_hidden):
        arch.append(Lin(l_in, hidden_size))
        arch.append(activation())
        l_in = hidden_size

    arch.append(Lin(l_in, out_size))

    if activate_last:
        arch.append(activation())

        if layer_norm:
            arch.append(LayerNorm(out_size))

    return Seq(*arch)


@torch.jit.script
def _scatter_add(src: Tensor, index: Tensor, dim: Tensor, dim_size: Tensor):
    return scatter_add(src, index, dim.item(), None, dim_size=dim_size.item())


@torch.jit.script
def _scatter_mean(src: Tensor, index: Tensor, dim: Tensor, dim_size: Tensor):
    return scatter_mean(src, index, dim.item(), None, dim_size=dim_size.item())


class GraphNet(torch.nn.Module):
    def __init__(
        self,
        in_dims,
        out_dims,
        independent=False,
        e2v_agg="sum",
        n_hidden=1,
        hidden_size=64,
        activation=ReLU,
        layer_norm=True,
    ):
        super().__init__()
        self.e2v_agg = e2v_agg
        if e2v_agg not in ["sum", "mean"]:
            raise ValueError("Unknown aggregation function.")

        v_in = in_dims[0]
        e_in = in_dims[1]
        u_in = in_dims[2]

        v_out = out_dims[0]
        e_out = out_dims[1]
        u_out = out_dims[2]

        if independent:
            self.edge_mlp = get_mlp(
                e_in,
                e_out,
                n_hidden,
                hidden_size,
                activation=activation,
                layer_norm=layer_norm,
            )
            self.node_mlp = get_mlp(
                v_in,
                v_out,
                n_hidden,
                hidden_size,
                activation=activation,
                layer_norm=layer_norm,
            )
            self.global_mlp = get_mlp(
                u_in,
                u_out,
                n_hidden,
                hidden_size,
                activation=activation,
                layer_norm=layer_norm,
            )
        else:
            self.edge_mlp = get_mlp(
                e_in + 2 * v_in + u_in,
                e_out,
                n_hidden,
                hidden_size,
                activation=activation,
                layer_norm=layer_norm,
            )
            self.node_mlp = get_mlp(
                v_in + e_out + u_in,
                v_out,
                n_hidden,
                hidden_size,
                activation=activation,
                layer_norm=layer_norm,
            )
            self.global_mlp = get_mlp(
                u_in + v_out + e_out,
                u_out,
                n_hidden,
                hidden_size,
                activation=activation,
                layer_norm=layer_norm,
            )

        self.independent = independent

        self.op = ModifiedMetaLayer(
            self.edge_model, self.node_model, self.global_model)

    def edge_model(self, src, dest, edge_attr, u=None, e_indices=None):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        if self.independent:
            return self.edge_mlp(edge_attr)

        out = torch.cat([src, dest, edge_attr, u[e_indices]], 1)
        return self.edge_mlp(out)

    def node_model(self, x, edge_index, edge_attr, u=None, v_indices=None):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        if self.independent:
            return self.node_mlp(x)

        row, col = edge_index
        out = None
        dim_size = torch.tensor(x.size(0))
        if self.e2v_agg == "sum":
            out = _scatter_add(
                edge_attr, row, dim=torch.tensor(0), dim_size=dim_size)
        elif self.e2v_agg == "mean":
            out = _scatter_mean(
                edge_attr, row, dim=torch.tensor(0), dim_size=dim_size)
        out = torch.cat([x, out, u[v_indices]], dim=1)
        return self.node_mlp(out)

    def global_model(self, x, edge_attr, u, v_indices, e_indices):
        if self.independent:
            return self.global_mlp(u)
        out = torch.cat(
            [
                u,
                scatter_mean(x, v_indices, dim=0),
                scatter_mean(edge_attr, e_indices, dim=0),
            ],
            dim=1,
        )
        return self.global_mlp(out)

    def forward(
        self, x, edge_index, edge_attr=None, u=None, v_indices=None, e_indices=None
    ):
        return self.op(x, edge_index, edge_attr, u, v_indices, e_indices)


class EncoderCoreDecoder(torch.nn.Module):
    def __init__(
        self,
        in_dims,
        core_out_dims,
        out_dims,
        core_steps=1,
        encoder_out_dims=None,
        dec_out_dims=None,
        e2v_agg="sum",
        n_hidden=1,
        hidden_size=64,
        activation=ReLU,
        independent_block_layers=1,
    ):
        super().__init__()
        # all dims are tuples with (v,e) feature sizes
        self.steps = core_steps
        # if dec_out_dims is None, there will not be a decoder
        self.in_dims = in_dims
        self.core_out_dims = core_out_dims
        self.dec_out_dims = dec_out_dims

        self.layer_norm = True

        self.encoder = None
        if encoder_out_dims is not None:
            self.encoder = GraphNet(
                in_dims,
                encoder_out_dims,
                independent=True,
                n_hidden=independent_block_layers,
                hidden_size=hidden_size,
                activation=activation,
                layer_norm=self.layer_norm,
            )

        core_in_dims = in_dims if self.encoder is None else encoder_out_dims

        self.core = GraphNet(
            (
                core_in_dims[0] + core_out_dims[0],
                core_in_dims[1] + core_out_dims[1],
                core_in_dims[2] + core_out_dims[2],
            ),
            core_out_dims,
            e2v_agg=e2v_agg,
            n_hidden=n_hidden,
            hidden_size=hidden_size,
            activation=activation,
            layer_norm=self.layer_norm,
        )

        if dec_out_dims is not None:
            self.decoder = GraphNet(
                core_out_dims,
                dec_out_dims,
                independent=True,
                n_hidden=independent_block_layers,
                hidden_size=hidden_size,
                activation=activation,
                layer_norm=self.layer_norm,
            )

        pre_out_dims = core_out_dims if self.decoder is None else dec_out_dims

        self.vertex_out_transform = (
            Lin(pre_out_dims[0], out_dims[0]
                ) if out_dims[0] is not None else None
        )
        self.edge_out_transform = (
            Lin(pre_out_dims[1], out_dims[1]
                ) if out_dims[1] is not None else None
        )
        self.global_out_transform = (
            Lin(pre_out_dims[2], out_dims[2]
                ) if out_dims[2] is not None else None
        )

    def get_init_state(self, n_v, n_e, n_u, device):
        return (
            torch.zeros((n_v, self.core_out_dims[0]), device=device),
            torch.zeros((n_e, self.core_out_dims[1]), device=device),
            torch.zeros((n_u, self.core_out_dims[2]), device=device),
        )

    def forward(self, x, edge_index, edge_attr, u, embed, v_indices=None, e_indices=None):
        # if v_indices and e_indices are both None, then we have only one graph without a batch
        if v_indices is None and e_indices is None:
            v_indices = torch.zeros(
                x.shape[0], dtype=torch.long, device=x.device)
            e_indices = torch.zeros(
                edge_attr.shape[0], dtype=torch.long, device=edge_attr.device
            )
        x = x + embed
        if self.encoder is not None:
            x, edge_attr, u = self.encoder(
                x, edge_index, edge_attr, u, v_indices, e_indices
            )

        latent0 = (x, edge_attr, u)
        latent = self.get_init_state(
            x.shape[0], edge_attr.shape[0], u.shape[0], x.device
        )
        for st in range(self.steps):
            latent = self.core(
                torch.cat([latent0[0], latent[0]], dim=1),
                edge_index,
                torch.cat([latent0[1], latent[1]], dim=1),
                torch.cat([latent0[2], latent[2]], dim=1),
                v_indices,
                e_indices,
            )

        if self.decoder is not None:
            latent = self.decoder(
                latent[0], edge_index, latent[1], latent[2], v_indices, e_indices
            )

        v_out = (
            latent[0]
            if self.vertex_out_transform is None
            else self.vertex_out_transform(latent[0])
        )

        e_out = (
            latent[1]
            if self.edge_out_transform is None
            else self.edge_out_transform(latent[1])
        )
        u_out = (
            latent[2]
            if self.global_out_transform is None
            else self.global_out_transform(latent[2])
        )
        return v_out, e_out, u_out


class AttentionPooling(nn.Module):
    """Some Information about AttentionPool"""

    def __init__(self, gate_nn, feat_nn):
        super(AttentionPooling, self).__init__()
        self.gate_nn = gate_nn
        self.feat_nn = feat_nn

    def forward(self, x):
        gate = self.gate_nn(x)
        feat = self.feat_nn(x) if self.feat_nn is not None else x
        gate = F.softmax(gate, -1)
        x = feat * gate
        return x.sum(0)


@torch.jit.script
def t_slice(x, size):
    seq = []
    cur = 0
    for el in size:
        offset = el.item()
        seq.append(x[cur:cur + offset])
        cur += offset
    return seq


class GNN(Model):
    """
    MLP model that is able to handle one dimensional state and action spaces. It does not care abut time series.
    So each example returned by the ``Dataset`` returned from ``dataset` method is a tuple of
    a single eg. state, action and reward

    This MLP model is a actor critic style one with shared input layers for both policy and value.
    """

    def __init__(self, state_space: Converter, action_space: Converter):
        super().__init__(state_space, action_space)
        self.input = EncoderCoreDecoder(
            (state_space.shape[0], state_space.shape[1], state_space.shape[2]),
            core_out_dims=(64, 64, 32),
            out_dims=(None, None, None),
            core_steps=4,
            dec_out_dims=(32, 32, 32),
            encoder_out_dims=(32, 32, 32),
            e2v_agg='mean',
            n_hidden=1,
            hidden_size=64,
            activation=nn.Tanh,
            independent_block_layers=1,
        )
        self.policy_out = self.action_space.policy_out_model(32)
        self.att_pool = AttentionPooling(nn.Linear(32, 32), nn.Linear(32, 32))
        self.value_out = nn.Linear(32, 1)

    def _forward(self, states):
        vout, _, _ = self.input(x=states[0],
                                edge_index=states[2],
                                edge_attr=states[1],
                                v_indices=states[4],
                                e_indices=states[5],
                                u=states[6],
                                embed=states[7])

        x = vout[states[0][:, 0] == 1]

        policy = self.policy_out(x).flatten()

        vertex_sizes = states[3]

        b_policy = t_slice(policy, vertex_sizes)

        x = t_slice(x, vertex_sizes)
        b_value = [self.value_out(self.att_pool(i)) for i in x]

        b_policy = torch.stack(b_policy)
        b_value = torch.stack(b_value)
        return b_policy, b_value

    def forward(self, state):
        b_policy = []
        b_value = []
        for obs in state:
            policy, value = self._forward(obs)
            b_policy.append(policy)
            b_value.append(value)
        b_policy = torch.stack(b_policy)
        b_value = torch.stack(b_value)
        return b_policy, b_value

    @ property
    def recurrent(self) -> bool:
        return True

    def value(self, states: Tensor) -> Tensor:
        _, value = self(states)
        return value

    def policy_logits(self, states: Tensor) -> Tensor:
        policy, _ = self(states)
        return policy

    def dataset(self, *arrays: np.ndarray) -> Dataset:
        return NonSequentialDataset(*arrays)

    @ staticmethod
    def factory() -> 'ModelFactory':
        return GNNFactory()


class GNNFactory(ModelFactory):
    def create(self, state_space: Converter, action_space: Converter) -> Model:
        return GNN(state_space, action_space)

    def create_eval(self, observation_space: Converter, action_space: Converter) -> Model:
        state_converter = Converter.for_space(observation_space)
        action_converter = Converter.for_space(action_space)
        return self.create(state_converter, action_converter)
