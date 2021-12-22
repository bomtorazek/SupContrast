from typing import Callable

import random

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def shuffle(*arys):
    """Shuffles multiple input arrays in the same order.
    Args:
        *arys (list[param]): list of arrays
    """
    assert np.mean(list(map(lambda ary: len(ary), arys))) == len(
        arys[0]
    )  # all arrays should have the same length
    perm = np.random.permutation(len(arys[0]))
    shuffled = list(map(lambda ary: np.array(ary)[perm], arys))
    if len(shuffled) == 1:
        return shuffled[0]
    return shuffled



class DomainWeightedSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        datapoints,
        num_domains,
        weights
    ):
        self.datapoints = datapoints
        self.num_domains = num_domains
        self.weights = self.get_weights(weights, num_domains)
        self.subsets = self.datapoints.domain_subsets
        self.weighted_subset_sizes = self.get_weighted_subset_sizes(self.weights, self.subsets)

    def __iter__(self):
        print(self.weighted_subset_sizes)
        domain_indices = [None]* self.num_domains
        for domain, weighted_subset_size in enumerate(self.weighted_subset_sizes):
            domain_indices[domain] = shuffle(self.sample_subset(domain, weighted_subset_size))
        
        domain_pointer, total_pointer = [0]*self.num_domains, 0
        max_unit = self.weighted_subset_sizes[0]//self.weights[0]
        indices = np.array([0]* (max_unit * sum(self.weights)))
        for _ in range(max_unit):
            for domain in range(self.num_domains):
                d_weight = self.weights[domain]
                indices[total_pointer: total_pointer + d_weight] = domain_indices[domain][domain_pointer[domain]: domain_pointer[domain]+ d_weight]
                domain_pointer[domain]+= d_weight
                total_pointer += d_weight

        return iter(indices)

    def __len__(self):
        return sum(self.weighted_subset_sizes)

    def get_weights(self, weights, num_domains):
        if weights is None:
            return [ 1 for i in range(num_domains) ]
        else:
            if len(weights) != num_domains:
                raise Exception("the length of weights must be equal to num_classes")
            return weights

    def get_weighted_subset_sizes(self, weights, subsets):
        # subsets -> [[indices of target images], [indices of source images]]
        reduced_subset_sizes = list(
            map(
                lambda tup: int(np.ceil(tup[0]/tup[1])),
                zip(
                    map(lambda subset: len(subset), subsets),  # [397,64] 
                    weights  # [3,1] 
                )  # [(397,3), (64,1)]
            )  # [133, 64] == [# of target // target weight, # of source // source weight]
        )
        max_unit = max(reduced_subset_sizes)  # 133
        units = [ max_unit if size != 0 else 0 for size in reduced_subset_sizes ]
        return [ int(np.round(w * u)) for w, u in zip(weights, units) ] # 399, 133

    def sample_subset(self, domain, weighted_subset_size):
        # (0, 399) or (1, 133)
        # (0, 216) or (1, 1512)
        subset = self.subsets[domain]
        subset_size = len(subset)
        if weighted_subset_size == 0 or subset_size == 0:
            return []
        if weighted_subset_size < subset_size:
            raise Exception("weighted_subset_size must not be smaller than subset_size")
        q = weighted_subset_size // subset_size
        r = weighted_subset_size % subset_size
        return q * subset + np.random.choice(subset, r, replace=False).tolist()
    
    def apply_domain_weight(self, domain_weight):
        self.weights = domain_weight
        print(self.weights)
        self.weighted_subset_sizes = self.get_weighted_subset_sizes(self.weights, self.subsets)
        


class DomainWeightedSampler_origin(torch.utils.data.Sampler):
    def __init__(
        self,
        datapoints,
        num_domains,
        weights
    ):
        self.datapoints = datapoints
        self.num_domains = num_domains
        self.weights = self.get_weights(weights, num_domains)
        self.subsets = self.datapoints.domain_subsets
        self.weighted_subset_sizes = self.get_weighted_subset_sizes(self.weights, self.subsets)

    def __iter__(self):
        indices = []
        print(self.weighted_subset_sizes)
        for domain, weighted_subset_size in enumerate(self.weighted_subset_sizes):
            indices += self.sample_subset(domain, weighted_subset_size)
        return iter(shuffle(indices))

    def __len__(self):
        return sum(self.weighted_subset_sizes)

    def get_weights(self, weights, num_domains):
        if weights is None:
            return [ 1 for i in range(num_domains) ]
        else:
            if len(weights) != num_domains:
                raise Exception("the length of weights must be equal to num_classes")
            return weights

    def get_weighted_subset_sizes(self, weights, subsets):
        # subsets -> [[indices of target images], [indices of source images]]
        reduced_subset_sizes = list(
            map(
                lambda tup: int(np.ceil(tup[0]/tup[1])),
                zip(
                    map(lambda subset: len(subset), subsets),  # [397,64] 
                    weights  # [3,1] 
                )  # [(397,3), (64,1)]
            )  # [133, 64] == [# of target // target weight, # of source // source weight]
        )
        max_unit = max(reduced_subset_sizes)  # 133
        units = [ max_unit if size != 0 else 0 for size in reduced_subset_sizes ]
        return [ int(np.round(w * u)) for w, u in zip(weights, units) ] # 399, 133

    def sample_subset(self, domain, weighted_subset_size):
        # (0, 399) or (1, 133)
        subset = self.subsets[domain]
        subset_size = len(subset)
        if weighted_subset_size == 0 or subset_size == 0:
            return []
        if weighted_subset_size < subset_size:
            raise Exception("weighted_subset_size must not be smaller than subset_size")
        q = weighted_subset_size // subset_size
        r = weighted_subset_size % subset_size
        return q * subset + np.random.choice(subset, r, replace=False).tolist()
    
    def apply_domain_weight(self, domain_weight):
        self.weights = domain_weight
        print(self.weights)
        self.weighted_subset_sizes = self.get_weighted_subset_sizes(self.weights, self.subsets)
        

class WeightedSampler(torch.utils.data.Sampler):
    def __init__(self, weights=None):
        self.weights = weights

    def __call__(self, dataset):
        self.dataset = dataset
        self.subsets = dataset.subsets
        self.weights = self.set_weights(self.weights, self.dataset.num_classes)
        self.weighted_subset_sizes = self.get_weighted_subset_sizes()
        return self

    def __iter__(self):
        indices = []
        for label, weighted_subset_size in enumerate(self.weighted_subset_sizes):
            indices += self.sample_subset(label, weighted_subset_size)
        return iter(shuffle(indices))

    def __len__(self):
        return sum(self.weighted_subset_sizes)

    def set_weights(self, weights, num_classes):
        if weights is None:
            return [1 for i in range(num_classes)]
        else:
            assert len(weights) == num_classes
            return weights

    def get_weighted_subset_sizes(self):
        reduced_subset_sizes = list(
            map(
                lambda tup: int(np.ceil(tup[0] / tup[1])),
                zip(
                    map(lambda subset: len(subset), self.subsets),  # [397,64]
                    self.weights,  # [3,1]
                ),  # [(397,3), (64,1)]
            )  # [133, 64]
        )
        max_unit = max(reduced_subset_sizes)  # 133
        units = [max_unit if size != 0 else 0 for size in reduced_subset_sizes]
        return [w * u for w, u in zip(self.weights, units)]

    def sample_subset(self, label, weighted_subset_size):
        subset = self.subsets[label]
        subset_size = len(subset)
        if weighted_subset_size == 0 or subset_size == 0:
            return []
        if weighted_subset_size < subset_size:
            raise ValueError(
                "weighted_subset_size should not be smaller than subset_size"
            )
        q = weighted_subset_size // subset_size
        r = weighted_subset_size % subset_size
        return q * subset + random.sample(subset, r)