from torch.utils.data import Dataset


class MergedSet(Dataset):
    def __init__(self, d1, d2):
        self.d1 = d1
        self.d2 = d2
        self.d1_len = self.d1.__len__()
        self.d2_len = self.d2.__len__()

    def __getitem__(self, index):
        if index >= self.d1_len:
            index = index - self.d1_len
            return self.d2.__getitem__(index)
        else:
            return self.d1.__getitem__(index)

    def __len__(self):
        return self.d1_len + self.d2_len

    def get_corpus(self):
        return self.d1.get_corpus() + " " + self.d2.get_corpus()
