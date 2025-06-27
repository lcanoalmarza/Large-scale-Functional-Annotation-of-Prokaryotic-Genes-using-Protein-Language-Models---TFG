#!/usr/bin/env python
"""
Class and functions to load the kegg data and serve it in a convenient
format for pytorch.

Created on Fri Mar 28 12:32:10 2025
@author: lcano
"""

import numpy as np
from torch import Tensor, FloatTensor
from torch.utils.data import Dataset, DataLoader

# Def class KeggDataset
class KeggDataset(Dataset):
    '''Convenience class to manage the kegg data derived from umap over
    ProstT5 embeddings'''

    def __init__(self, xs, ys, labels):
        self.xs = Tensor(xs)
        self.ys = ys
        self.labels = labels

    def __len__(self):
        return len(self.ys)  # number of samples

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

# Def get_dataloaders

def get_dataloaders(umap_file, labels_file,
                    test_size=0.25, batch_size=64, shuffle=False):
    '''Return torch dataloaders for training and test data.'''
    xs=np.load(umap_file)[:, :30]
    y_labels=np.loadtxt(labels_file, delimiter=',', skiprows=0, dtype=str)

    labels = sorted(set(y_labels))
    assert len(labels) > 1, 'Only one label (you selected an invalid label?).'

    label2y = {labels[i]: i for i in range(len(labels))}
    ys = np.array([label2y[label] for label in y_labels])
    
    _, counts = np.unique(y_labels, return_counts=True)
    weights = 1 / counts
    weights /= weights.sum()

    if shuffle:
        indices = np.arange(ys.shape[0])
        np.random.shuffle(indices)
        xs = xs[indices]
        ys = ys[indices]

    n = int(len(ys) * (1 - test_size))  # we take the first n for training
    data_train = KeggDataset(xs[:n], ys[:n], labels)
    data_test = KeggDataset(xs[n:], ys[n:], labels)
    class_weights = FloatTensor(weights)

    return (DataLoader(data_train, batch_size=batch_size),
            DataLoader(data_test, batch_size=batch_size),
            class_weights)
