import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
import os
import torch_geometric
import numpy as np
import torch
import json

atom_to_idx = {
    "C": 0, "N": 1, "O": 2, "S": 3, "F": 4, "P": 5, "Br": 6, "Cl": 7,
    "Se": 8, "I": 9, "Si": 10, "B": 11, "Sn": 12, "K": 13, "Ru": 14, "Fe": 15
}
bond_to_idx = {1.0: 0, 1.5: 1, 2.0: 2, 3.0: 3}


class HAMDs(torch.nn.Module):
    def __init__(self, data_path, train_prop=0.7, val_prop=0.15):
        super(HAMDs, self).__init__()
        self.graphs, self.fnames = [], []
        for x in os.listdir(data_path):
            if not x.endswith('.json'):
                continue
            self.fnames.append(x)
            with open(os.path.join(data_path, x)) as Fin:
                info = json.load(Fin)

            mol = Chem.MolFromSmiles(info['smiles'])
            weight = Descriptors.MolWt(mol)
            info['weight'] = weight
            block = {}
            for idx, px in enumerate(info['cgnodes']):
                block.update({y: idx for y in px})
            info['block'] = block

            self.graphs.append(info)
        self.graphs.sort(key=lambda x: x['weight'])

        train_idx = int(train_prop * len(self.graphs))
        valid_idx = int(val_prop * len(self.graphs)) + train_idx

        self.data_split = {
            'train_idx': list(range(train_idx)),
            'valid_idx': list(range(train_idx, valid_idx)),
            'test_idx': list(range(valid_idx, len(self.graphs)))
        }

    def get_files_split(self):
        return {
            'train': [self.fnames[x] for x in self.data_split['train_idx']],
            'valid': [self.fnames[x] for x in self.data_split['valid_idx']],
            'test': [self.fnames[x] for x in self.data_split['test_idx']],
        }

    def get_one_item(self, idx):
        num_nodes = len(self.graphs[idx]['nodes'])
        node_feat = np.zeros(num_nodes, dtype=np.int64)
        edge_idx, edge_feat, label = [], [], []
        for nd in self.graphs[idx]['nodes']:
            node_feat[nd['id']] = atom_to_idx[nd['element']]

        for eg in self.graphs[idx]['edges']:
            i_block = self.graphs[idx]['block'][eg['source']]
            j_block = self.graphs[idx]['block'][eg['target']]

            edge_idx.append((eg['source'], eg['target']))
            edge_feat.append(bond_to_idx[eg['bondtype']])
            label.append(i_block != j_block)

            edge_idx.append((eg['target'], eg['source']))
            edge_feat.append(bond_to_idx[eg['bondtype']])
            label.append(j_block != i_block)

        edge_idx = np.array(edge_idx, dtype=np.int64).T
        edge_feat = np.array(edge_feat, dtype=np.int64)

        return {
            'node_feat': node_feat, "num_nodes": num_nodes,
            'edge_index': edge_idx, 'edge_feat': edge_feat,
            'label': np.array(label, dtype=np.float32)
        }

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.get_one_item(x) for x in idx]
        else:
            return self.get_one_item(idx)


def col_fn(batch_data):
    edge_feat, node_feat, edge_idx = [], [], []
    batch, ptr, lstnode, labels = [], [0], 0, []

    for idx, gp in enumerate(batch_data):
        node_feat.append(gp['node_feat'])
        edge_feat.append(gp['edge_feat'])
        edge_idx.append(gp['edge_index'] + lstnode)
        node_cnt = gp['num_nodes']
        batch.append(np.ones(node_cnt, dtype=np.int64) * idx)
        lstnode += node_cnt
        ptr.append(lstnode)
        labels.append(gp['label'])

    return torch_geometric.data.Data(**{
        'x': torch.from_numpy(np.concatenate(node_feat, axis=0)),
        'edge_attr': torch.from_numpy(np.concatenate(edge_feat, axis=0)),
        'edge_index': torch.from_numpy(np.concatenate(edge_idx, axis=1)),
        'num_nodes': lstnode,
        'batch': torch.from_numpy(np.concatenate(batch, axis=0)),
        'ptr': torch.LongTensor(ptr),
        'label': torch.from_numpy(np.concatenate(labels, axis=0))
    })
