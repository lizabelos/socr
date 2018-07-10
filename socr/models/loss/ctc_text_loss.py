import torch

from socr.utils.language.word_beam_search import wordBeamSearch
from socr.utils.setup.build import install_and_import_wrapctc

warpctc = install_and_import_wrapctc()

from . import Loss


class CTCTextLoss(Loss):
    """A YoloV1 simplified Loss"""

    def __init__(self, labels):
        """

        :param s: grid division, assuming we have only 1 bounding box per cell
        """
        super().__init__()

        self.labels = labels
        self.labels_len = len(self.labels) + 1
        self.loss = warpctc.CTCLoss(size_average=False, length_average=False).cuda()

    def forward(self, inputs, truth):
        labels, labels_length = truth
        labels = torch.autograd.Variable(torch.IntTensor(labels), requires_grad=False)
        labels_length = torch.autograd.Variable(torch.IntTensor(labels_length), requires_grad=False)

        # inputs = inputs.transpose(0, 1)
        inputs_width = inputs.size(0)
        batch_size = inputs.size(1)
        inputs_width = torch.autograd.Variable(torch.IntTensor([inputs_width] * batch_size), requires_grad=False)

        return self.loss(inputs.contiguous(), inputs_width, labels, labels_length)

    def process_labels(self, s_list, is_cuda=True):
        labels = []
        labels_length = []

        for s in s_list:
            s_result = []
            for c in s:
                pos = self.labels.find(c)
                if pos == -1:
                    raise RuntimeError("Invalid Character : " + str(c) + "!")
                # Assuming that space is that position 0
                s_result.append(pos + 1)
            labels = labels + s_result
            labels_length = labels_length + [len(s_result)]

        return labels, labels_length

    def ytrue_to_lines(self, lm, sequence):
        result = wordBeamSearch(sequence[0].data.cpu().numpy(), 5, lm, False)
        return result

    def convert_to_string(self, tokens, vocab, seq_len):
        return ''.join([vocab[x] for x in tokens[0:seq_len]])


