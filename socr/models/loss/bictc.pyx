from functools import reduce

import numpy as np
import torch

from socr.utils.logging.logger import print_normal, print_warning

cdef class BiCTC:

    cdef object nll
    cdef dict labels
    cdef object width_transform

    def __init__(self, labels, width_transform):
        super().__init__()
        self.labels = labels
        self.width_transform = width_transform
        self.nll = torch.nn.PoissonNLLLoss()

    def forward(self, output, label_matrix):
        num_path = label_matrix.size()[1]
        width = output.size()[1]

        output = torch.stack([output] * num_path)


        # print(output.size())
        # print(label_matrix.size())

        # OUTPUT : num_path x width x batch_size x num_label
        # LABEL : batch_size x num_path x num_label x width
        # WHAT WE WANT : num_label x width x num_path x batch_size

        output = output.permute(3, 1, 0, 2)
        label_matrix = label_matrix.permute(2, 3, 1, 0)

        m = label_matrix * output
        m = m.sum(dim=0) # num_label : width x num_path x batch_size
        m = reduce(lambda x, y: x * y, [m[i] for i in range(0, width)])
        m = m.sum(dim=0) # num_path  : batch_size
        m = m.unsqueeze(0)

        target = m.new_ones(m.size())

        print(m.size)
        print(target.size())

        return self.nll(m, target)

    def cuda(self, **kwargs):
        print_normal("Using CTCLoss with CUDA")
        self.nll.cuda()
        return self

    def cpu(self):
        print_warning("Using CTCLoss with CPU")
        self.nll.cpu()
        return self

    def text_to_label(self, text):

        text_label = []

        for i in range(1, len(text)):
            c1 = text[i - 1]
            c2 = text[i]
            try:
                index = self.labels[c1 + c2]
                text_label.append(index)
            except KeyError:
                print_warning("Invalid key : " + c1 + c2)

        return text_label

    cpdef label_to_paths(self, list label, int remaining_width, int total_width, int div=1):
        cdef int blank_label = self.labels[""]

        if len(label) == 0:
            if remaining_width == 0:
                return [[]]
            else:
                return [[blank_label] * remaining_width]

        if remaining_width < len(label):
            return []

        cdef int current_label = label[0]
        cdef list current_paths = []

        # For current_label path
        for i in range(1, remaining_width + 1):
            next_paths = self.label_to_paths(label[1:], remaining_width - i, total_width, div)

            for path in next_paths:
                current_paths.append(i * [current_label] + path)

        if remaining_width == total_width:
            # For blank path
            for i in range(1, remaining_width + 1):
                next_paths = self.label_to_paths(label, remaining_width - i, total_width, div)

                for path in next_paths:
                    current_paths.append(i * [blank_label] + path)

        return current_paths

    def paths_to_matrix(self, paths):
        if len(paths) == 0:
            return np.zeros((len(paths), len(self.labels), 0), dtype='float32')
        matrix = np.zeros((len(paths), len(self.labels), len(paths[0])), dtype='float32')

        for j in range(0, len(paths)):
            for i in range(0, len(paths[j])):
                matrix[j][paths[j][i]][i] = 1.0

        return matrix

    def preprocess_label(self, text, width):
        width = int(self.width_transform(width))
        label = self.text_to_label(text)
        paths = self.label_to_paths(label, width, width, div=4)
        return torch.from_numpy(self.paths_to_matrix(paths))

    def process_labels(self, labels, is_cuda=True):
        var = torch.autograd.Variable(labels).float()
        if is_cuda:
            var = var.cuda()
        return var

    # def collate(self, batch):
    #     data = [item[0] for item in batch]
    #     max_width = max([d.size()[2] for d in data])
    #
    #     data = [torch.nn.functional.pad(d, (0, max_width - d.size()[2], 0, 0)) for d in data]
    #     data = torch.stack(data)
    #
    #     target = [item[1] for item in batch]
    #
    #     # TODO : First pad width axes with blank label
    #     # TODO : Then, pad num paths axis with zero
    #
    #     return [data, target]
