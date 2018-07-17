from abc import abstractmethod

from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):

    @abstractmethod
    def count(self): pass

    @abstractmethod
    def get(self, index): pass

    def __getitem__(self, index):
        return self.get(index)

    def __len__(self):
        return self.count()
