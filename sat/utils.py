import torch
import os
import pickle
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np


class EarlyStopping(object):
    def __init__(self, save_dir, prefixs, patience=10):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.prefixs = prefixs
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if acc >= self.best_acc:
                self.best_loss = loss
                self.best_acc = acc
                self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def _getFilename(self, prefix):
        return os.path.join(self.save_dir, prefix+'_loss_{:.4f}_acc_{:.4f}.pth'.format(self.best_loss, self.best_acc))

    def save_checkpoint(self, models):
        """Saves model when validation loss decreases."""
        if isinstance(models, tuple):
            for i, model in enumerate(models):
                torch.save(model.state_dict(
                ), self._getFilename(self.prefixs[i]))
        else:
            torch.save(models.state_dict(), self._getFilename(self.prefixs))
        print('Save model loss {:.4f}  acc {:.4f}'.format(
            self.best_loss, self.best_acc))

    def load_checkpoint(self, models):
        """Load the latest checkpoint."""
        if isinstance(models, tuple):
            for i, model in enumerate(models):
                model.load_state_dict(torch.load(
                    self._getFilename(self.prefixs[i])))
        else:
            models.load_state_dict(torch.load(
                self._getFilename(self.prefixs)))


def get_all_files(path, pattern):
    files = []
    lsdir = os.listdir(path)
    dirs = [i for i in lsdir if os.path.isdir(os.path.join(path, i))]
    if dirs:
        for i in dirs:
            files += get_all_files(os.path.join(path, i), pattern)
    files += [os.path.abspath(os.path.join(path, i)) for i in lsdir if os.path.isfile(
        os.path.join(path, i)) and os.path.splitext(i)[1] == pattern and os.path.getsize(os.path.join(path, i))]
    return files


def load_cache(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as cache:
            return pickle.load(cache)


def save_cache(filepath, data):
    if not os.path.exists(os.path.split(filepath)[0]):
        try:
            os.makedirs(os.path.split(filepath)[0])
        except FileExistsError:
            pass
    with open(filepath, 'wb') as cache:
        pickle.dump(data, cache)


def average_relative_error(pred, label):
    p = mean_absolute_percentage_error(label, pred)
    return 1 - p


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


def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        import json
        return json.load(f)


def parse_cnf(filepath: str) -> tuple:
    if filepath.rfind('.npz') > 0:
        return _parse_npz(
            filepath)
    else:
        return _parse_cnf(
            filepath)


def _parse_npz(npz_file: str) -> tuple:
    datas = np.load(npz_file)
    indices, values = datas['indices'], datas[
        'values']
    clause_num, var_num = np.max(indices, axis=0) + 1
    return var_num, clause_num, indices, values


def _parse_cnf(cnf_file: str) -> tuple:
    var_num = 0
    clause_num = 0
    indices = []
    values = []
    with open(cnf_file, 'r') as cnf:
        for line in cnf.readlines():
            if line.strip() == '':
                continue
            splited_line = line.split()
            if splited_line[0] == 'c':
                continue
            if splited_line[0] == 'p':
                var_num = int(splited_line[2])
            if splited_line[0] not in ('c', 'p'):
                for var in splited_line:
                    var = int(var)
                    if var == 0:
                        continue
                    value = [0, 0]
                    if var > 0:
                        var = var - 1
                        value[0] = 1
                    if var < 0:
                        var = -var - 1
                        value[1] = 1
                    indices.append((clause_num, var))
                    values.append(value)
                clause_num += 1
    return var_num, clause_num, indices, values
