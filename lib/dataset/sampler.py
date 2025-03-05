import time
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler, Dataset


class CategoriesSampler:
    def __init__(self, label, n_batch, n_cls, n_per, n_extra=32):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per
        self.n_extra = n_extra

        label = np.array(label, dtype=np.int64)  # all data label
        self.m_ind = []  # the data index of each class
        self.j_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            if ind.shape[0] > 0:
                if i < 1000:
                    self.m_ind.append(ind)
                else:
                    self.j_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[: self.n_cls]  # random sample num_class indexs,e.g. 5
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[: self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)

            if self.n_extra > 0:
                batch = self.get_extra_ind(batch)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch

    def get_extra_ind(self, batch):
        for ind in self.j_ind:
            extra_batch = torch.randperm(len(ind))[: self.n_extra]
            batch = torch.cat([batch, extra_batch])
        return batch

    def set_epoch(self, epoch):
        pass


class DDP_CategoriesSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        label,
        episode=1000,
        class_other_turn=(8, 8),
        n_per=2,
        seed: int = 0,
    ):
        """
        Different process should have different seed
        """
        super().__init__(dataset=dataset, drop_last=True, seed=seed)

        label = np.array(label, dtype=np.int64)

        num_class_sample = 0
        class_ind_ls = []
        other_ind_ls = []
        for label_i in range(max(label) + 1):
            label_i_ind = np.argwhere(label == label_i).reshape(-1)
            label_i_ind = torch.from_numpy(label_i_ind)
            if label_i_ind.shape[0] > 0:
                if label_i < 1000:
                    class_ind_ls.append(label_i_ind)
                    num_class_sample += len(label_i_ind)
                else:
                    other_ind_ls.append(label_i_ind)

        self.label = label
        self.class_ind_ls = class_ind_ls
        self.other_ind_ls = other_ind_ls
        self.n_per = n_per
        self.n_way = class_other_turn[0] // n_per
        self.class_other_turn = class_other_turn
        self.num_class = len(class_ind_ls)
        self.num_other = len(other_ind_ls)
        self.num_class_sample = num_class_sample
        self.num_samples = episode * (class_other_turn[0] + class_other_turn[1])
        self.episode = episode

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch + 100000 * self.rank)
        # g.manual_seed(self.seed + self.epoch)  #  + 100000 * self.rank

        other_ind = [torch.randperm(len(self.other_ind_ls[0]), generator=g)]

        indices = []
        for i in range(self.episode):
            selected_classes = torch.randperm(self.num_class, generator=g)[: self.n_way]

            for cls in selected_classes:
                class_ind = self.class_ind_ls[cls]
                class_selected_ind = torch.randperm(len(class_ind), generator=g)[: self.n_per]
                class_selected_ind = class_ind[class_selected_ind]
                indices.extend(class_selected_ind.tolist())

            for cls in range(self.num_other):
                class_ind = self.other_ind_ls[cls]
                _shot = self.class_other_turn[1] // self.num_other
                class_selected_ind = other_ind[cls][_shot * i : _shot * (i + 1)]
                class_selected_ind = class_ind[class_selected_ind]
                indices.extend(class_selected_ind.tolist())

        return iter(indices)
