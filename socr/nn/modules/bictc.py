import torch
import numpy as np


class BiCTC:

    def __init__(self):
        self.nll = torch.nn.NLLLoss()

    def forward(self, output, label_matrix):
        batch_size = output.size()[0]
        num_path = label_matrix.size()[0]

        # TODO : Duplicate label_matrix batch_size times
        # TODO : Duplicate output num_path times
        # TODO : So we want batch_size x num_path x num_label x width

        label_matrix = torch.stack([label_matrix] * batch_size)
        output = torch.stack([output] * num_path).transpose(0, 1)

        m = label_matrix * output
        m = torch.sum(m, axis=2) # batch_size x num_path x width
        m = torch.mul(m, axis=2) # batch_size x num_path
        m = torch.sum(m, axis=2) # batch_size

        target = m.new()
        target[:] = 1.0

        return self.nll(m, target)

    def text_to_label(self, text, labels):

        text_label = []

        for i in range(1, len(text)):
            c1 = text[i - 1]
            c2 = text[i]
            index = labels[c1 + c2]
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
        current_paths = []

        for i in range(1, remaining_width + 1):
            next_paths = self.label_to_paths(label[1:], remaining_width - i)

            for path in next_paths:
                current_paths.append(i * [current_label] + path)

        return current_paths

    def paths_to_matrix(self, paths, labels):
        matrix = np.zeros((len(paths), len(labels), len(paths[0])), dtype='float32')

        for j in range(0, len(paths)):
            for i in range(0, len(paths[j])):
                matrix[j][paths[j][i]][i] = 1.0

        return matrix
