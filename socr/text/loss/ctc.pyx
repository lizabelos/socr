import numpy as np
np.seterr(divide='raise',invalid='raise')
import torch

from socr.utils.logger import print_normal, print_warning, print_error
from socr.text.codecs.ctc_decoder import CTCDecoder
from socr.text.modules.ctc import CTCFunction

cpdef text_to_label(dict labels, str text):
    cdef list text_label = []
    cdef int i
    cdef int j
    cdef int index
    cdef int remaining
    cdef int last_length = 0

    for i in range(0, len(text)):
        index = -1

        remaining = len(text) - i + 1
        for j in range(1, remaining):
            crop = text[i:i+j]

            if crop in labels:
                index = labels[crop]
                last_length = len(crop)

        if index!= -1:
            text_label.append(index)
        else:
            if len(text[i:]) > last_length:
                raise Exception
            last_length = last_length - 1

    if len(text_label) == 0:
        print_warning("Text label of length 0")
        return [labels[""]]

    return text_label

class CTC(torch.nn.Module):

    def __init__(self, labels, width_transform):
        super().__init__()

        self.labels = labels
        self.inv_labels = {v: k for k, v in self.labels.items()}
        self.label_len = max(labels.values()) + 1

        for i in range(0, self.label_len):
            if not i in self.inv_labels:
                print_error("CTC Key error : " + str(i))
                assert False

        self.width_transform = width_transform
        self.softmax = torch.nn.Softmax(dim=2)
        self.ctc = CTCFunction()
        self.is_gpu = False
        self.decoder = CTCDecoder(self.inv_labels, self.label_len)

    def forward(self, output, label):
        return self.ctc.apply(output, label, self.labels[""], self.is_gpu, self.width_transform)

    def cuda(self, **kwargs):
        self.is_gpu = True
        print_warning("CTC do not support GPU and will be processed on CPU")
        return self

    def cpu(self):
        self.is_gpu = False
        print_normal("Using CTCLoss with CPU")
        return self

    def text_to_label(self, str text):
        return text_to_label(self.labels, text)

    def preprocess_label(self, text, width):
        width = int(self.width_transform(width))
        label = self.text_to_label(text)
        return torch.from_numpy(np.array(label))

    def process_labels(self, labels, is_cuda=True):
        return labels

    def ytrue_to_lines(self, sequence):
        return self.decoder.decode(sequence)

    def collate(self, batch):
        data = [item[0] for item in batch]
        max_width = max([d.size()[2] for d in data])

        data = [torch.nn.functional.pad(d, (0, max_width - d.size()[2], 0, 0)) for d in data]
        data = torch.stack(data)

        target = [item[1] for item in batch]

        return [data, target]
