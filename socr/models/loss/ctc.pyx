from functools import reduce

import numpy as np
import torch

from socr.utils.logging.logger import print_normal, print_warning, print_error

cdef class CTC:

    cdef object nll
    cdef dict labels
    cdef object width_transform
    cdef bint blank_as_separator

    def __init__(self, labels, width_transform, blank_as_separator=False):
        super().__init__()

        # todo : blank as separator

        self.labels = labels
        self.width_transform = width_transform
        self.nll = torch.nn.BCELoss()
        self.blank_as_separator = blank_as_separator

    def forward(self, output, label_matrix):
        # print(output.size())
        # print(label_matrix.size())

        # OUTPUT : width x batch_size x num_label
        # LABEL : batch_size x num_label x width

        # output = output.permute(1,0,2)

        assert not torch.isnan(output).any()
        assert not torch.isnan(label_matrix).any()

        return self.nll(output, label_matrix)

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


    cpdef label_to_path_matrix(self, list label, int width):
        cdef int blank_label_start = -1
        cdef int blank_label_end = -1
        label = [blank_label_start] + label + [blank_label_end]

        cdef float[:,:] matrix = np.zeros((width, len(label)), dtype='float32')
        cdef float[:] time_sum = np.zeros((width), dtype='float32')

        cdef int i
        cdef int j
        cdef int start_range = 0
        cdef int end_range = 0

        for i in range(0, width):

            # todo : check start and end range
            # todo : start with blank and first, end with blank end last (idea : increment width by 2)
            end_range = min(end_range + 1, len(label) - 1)
            if width - i <= len(label):
                start_range = min(start_range + 1, len(label) - 1)

            for j in range(start_range, end_range + 1):
                if i == 0:
                    matrix[i][j] += 1.0
                    time_sum[i] += 1.0
                else:
                    matrix[i][j] += matrix[i - 1][j] / time_sum[i - 1]
                    time_sum[i] += matrix[i - 1][j] / time_sum[i - 1]

                    if j != 0:
                        matrix[i][j] += matrix[i - 1][j - 1] / time_sum[i - 1]
                        time_sum[i] += matrix[i - 1][j - 1] / time_sum[i - 1]

    
        cdef float[:,:] char_matrix = np.zeros((width, len(self.labels)), dtype='float32')
        cdef int blank_label = self.labels[""]
        cdef int label_id

        for i in range(0, width):
            for j in range(0, len(label)):
                label_id = label[j]
                if label_id < 0:
                    label_id = blank_label

                if time_sum[i] == 0:
                    print("wrong time sum for " + str(i))

                char_matrix[i][label_id] += matrix[i][j] / time_sum[i]

                if np.isnan(char_matrix[i][label_id]):
                    print_error("Numerical stability error at CTC for time " + str(i) + " with values : " + str(matrix[i][j]) + " " + str(time_sum[i]))
                    assert False

        return char_matrix

    def preprocess_label(self, text, width):
        width = int(self.width_transform(width))
        label = self.text_to_label(text)
        matrix = self.label_to_path_matrix(label, width)
        return torch.from_numpy(np.array(matrix))

    def process_labels(self, labels, is_cuda=True):
        var = torch.autograd.Variable(labels).float()
        if is_cuda:
            var = var.cuda()
        return var

    def ytrue_to_lines(self, sequence):
        # OUTPUT : width x batch_size x num_label
        width = sequence.shape[0]
        batch_size = sequence.shape[1]

        text = ""
        last_label = -1

        for time in range(0, width):

            max_label = 0
            for i in range(1, len(self.labels)):
                if sequence[time][0][i] > sequence[time][0][max_label]:
                    max_label = i
            
            if max_label != last_label:
                text = text + self.labels[max_label]
                last_label = max_label

        return text

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
