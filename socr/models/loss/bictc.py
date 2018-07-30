import torch
import numpy as np

from socr.models.loss import Loss


class BiCTC(Loss):

    def __init__(self, labels):
        super().__init__()
        self.labels = labels
        self.nll = torch.nn.NLLLoss()

    def forward(self, output, label_matrix):
        num_path = label_matrix.size()[1]

        output = torch.stack([output] * num_path).transpose(0, 1) # batch_size x num_path x num_label x width

        m = label_matrix * output
        m = torch.sum(m, axis=2) # batch_size x num_path x width
        m = torch.mul(m, axis=2) # batch_size x num_path
        m = torch.sum(m, axis=2) # batch_size

        target = m.new()
        target[:] = 1.0

        return self.nll(m, target)

    def text_to_label(self, text):

        text_label = []

        for i in range(1, len(text)):
            c1 = text[i - 1]
            c2 = text[i]
            index = self.labels[c1 + c2]
            text_label.append(index)

        return text_label

    def label_to_paths(self, label, remaining_width):
        if len(label) == 0:
            if remaining_width == 0:
                return [[]]
            else:
                return []

        if remaining_width == 0:
            return []

        current_label = label[0]
        blank_label = self.labels[""]
        current_paths = []

        # For current_label path
        for i in range(1, remaining_width + 1):
            next_paths = self.label_to_paths(label[1:], remaining_width - i)

            for path in next_paths:
                current_paths.append(i * [current_label] + path)

        # For blank path
        for i in range(1, remaining_width + 1):
            next_paths = self.label_to_paths(label, remaining_width - i)

            for path in next_paths:
                current_paths.append(i * [blank_label] + path)

        return current_paths

    def paths_to_matrix(self, paths):
        matrix = np.zeros((len(paths), len(self.labels), len(paths[0])), dtype='float32')

        for j in range(0, len(paths)):
            for i in range(0, len(paths[j])):
                matrix[j][paths[j][i]][i] = 1.0

        return matrix

    def preprocess_label(self, text, width):
        label = self.text_to_label(text)
        paths = self.label_to_paths(label, width)
        return torch.from_numpy(self.paths_to_matrix(paths))

    def process_labels(self, labels, is_cuda=True):
        var = torch.autograd.Variable(labels).float()
        if is_cuda:
            var = var.cuda()
        return var

    def collate(self, batch):
        data = [item[0] for item in batch]
        max_width = max([d.size()[2] for d in data])

        data = [torch.nn.functional.pad(d, (0, max_width - d.size()[2], 0, 0)) for d in data]
        data = torch.stack(data)

        target = [item[1] for item in batch]
        # TODO : First pad width axes with blank label
        # TODO : Then, pad num paths axis with zero

        return [data, target]
