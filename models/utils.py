import torch
import numpy as np


def batch_graphs(graphs):
    # we treat a batch of graphs as a one mega graph with several components disconnected from each other
    # we can simply concat the data and adjust the connectivity indices

    vertex_sizes = torch.tensor([el[0].shape[0]
                                 for el in graphs], dtype=torch.long)
    edge_sizes = torch.tensor([el[1].shape[0]
                               for el in graphs], dtype=torch.long)
    vcumsize = np.cumsum(vertex_sizes)
    variable_nodes_sizes = torch.tensor(
        [el[0][el[0][:, 0] == 1].shape[0] for el in graphs],
        dtype=torch.long
    )

    vbatched = torch.cat([el[0] for el in graphs])
    ebatched = torch.cat([el[1] for el in graphs])
    gbatched = torch.cat([el[3] for el in graphs])  # 3 is for global
    bbatched = torch.cat([el[4] for el in graphs])  # 4 is for embeded
    mbatched = torch.cat([el[5] for el in graphs])

    conn = torch.cat([el[2] for el in graphs], dim=1)
    conn_adjuster = vcumsize.roll(1)
    conn_adjuster[0] = 0
    conn_adjuster = torch.tensor(
        np.concatenate(
            [edge_sizes[vidx].item() * [el]
             for vidx, el in enumerate(conn_adjuster)]
        ),
        dtype=torch.long,
    )
    conn = conn + conn_adjuster.expand(2, -1)

    v_graph_belonging = torch.tensor(
        np.concatenate([el.item() * [gidx]
                        for gidx, el in enumerate(vertex_sizes)]),
        dtype=torch.long,
    )
    e_graph_belonging = torch.tensor(
        np.concatenate([el.item() * [gidx]
                        for gidx, el in enumerate(edge_sizes)]),
        dtype=torch.long,
    )

    return (
        vbatched,
        ebatched,
        conn,
        variable_nodes_sizes,
        v_graph_belonging,
        e_graph_belonging,
        gbatched,
        bbatched,
        mbatched
    )
