from __future__ import absolute_import

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class FixSampler(Sampler):
    """Sample the same number of examples from each class."""

    def __init__(self, data_source, sample_order):
        self.data_source = data_source
        self.sample_order = sample_order

    def __len__(self):
        return len(self.sample_order)

    def __iter__(self):
        return iter(self.sample_order)


class RandomIdentitySampler(Sampler):
    """Sample the same number of examples from each class."""

    def __init__(self, data_source, num_instances=1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = data_source.Index
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)
