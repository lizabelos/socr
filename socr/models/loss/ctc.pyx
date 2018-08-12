from functools import reduce

import numpy as np
import torch

from socr.utils.logging.logger import print_normal, print_warning, print_error

cdef class CTC:

    cdef object nll
    cdef dict labels
    cdef dict inv_labels
    cdef object width_transform
    cdef bint blank_as_separator
    cdef int label_len

    def __init__(self, labels, width_transform, blank_as_separator=False):
        super().__init__()

        # todo : blank as separator
        self.labels = labels
        self.inv_labels = {v: k for k, v in self.labels.items()}
        self.width_transform = width_transform
        self.nll = torch.nn.BCELoss()
        self.blank_as_separator = blank_as_separator
        self.label_len = max(labels.values()) + 1

    def forward(self, output, label):
        # batch_size x probs x width

        # assert not torch.isnan(output).any()
        # assert not torch.isnan(label_matrix).any()

        # print(torch.sum(output, dim=1))
        # print(torch.sum(label_matrix, dim=2))
        label_matrix = label[0]

        prob_matrix = output * label_matrix
        prob_matrix = torch.sum(prob_matrix, dim=2)

        # print("COUCOU")
        # print(prob_matrix)

        prob_matrix = -torch.log(prob_matrix)

        # print("AU REVOIR")
        # print(prob_matrix)

        return torch.sum(prob_matrix)

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

            if c1 in self.labels:
                index = self.labels[c1]
                text_label.append(index)
            elif c1 + c2 in self.labels:
                index = self.labels[c1 + c2]
                text_label.append(index)
            else:
                print_warning("Invalid key : " + c1 + c2)

        if len(text_label) == 0:
            print_warning("Text label of length 0")

        return text_label


    cpdef label_to_path_matrix(self, list label, int width):
        cdef int blank_label_start = -1
        cdef int blank_label_end = -2
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

    
        cdef float[:,:] char_matrix = np.zeros((width, self.label_len), dtype='float32')
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
        var = torch.autograd.Variable(labels[0]).float()
        if is_cuda:
            var = var.cuda()
        return var, labels[1]

    cpdef ytrue_to_lines(self, float[:,:,:] sequence):
        # OUTPUT : batch_size x width x num_label
        cdef int width = sequence.shape[1]
        cdef int batch_size = sequence.shape[0]

        cdef str text = ""
        cdef int last_label = -1
        
        cdef int time
        cdef int max_label
        cdef int i

        for time in range(0, width):

            max_label = 0
            for i in range(1, self.label_len):
                if sequence[0][time][i] > sequence[0][time][max_label]:
                    max_label = i
            
            if max_label != last_label:
                text = text + self.inv_labels[max_label]
                last_label = max_label

        return text

    def collate(self, batch):
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
