from libc cimport math
import cython
from cython.parallel import parallel, prange
import numpy as np
np.seterr(divide='raise',invalid='raise')
import torch

from socr.utils.logger import print_normal, print_warning, print_error
from socr.text.codecs.ctc_decoder import CTCDecoder


# From https://raw.githubusercontent.com/amaas/stanford-ctc/master/ctc/ctc.py

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np_ctc(float[:,:] prediction, int[::1] sequence, unsigned int blank, float[:,:] grad, unsigned int width):
    """
    CTC loss function.
    params - n x m matrix of n-D probability distributions over m frames. Must
    be in Fortran order in memory.
    seq - sequence of phone id's for given example.
    Returns objective and gradient.
    """

    cdef unsigned int sequence_length = sequence.shape[0] # Length of label sequence (# phones)
    cdef unsigned int numphones = prediction.shape[0] # Number of labels
    cdef unsigned int sequence_with_blank_length = 2*sequence_length + 1 # Length of label sequence with blanks
    # cdef unsigned int T = params.shape[1] # Length of utterance (time)

    cdef float[:,:] alphas = np.zeros((sequence_with_blank_length,width), dtype='float32')
    cdef float[:,:] betas = np.zeros((sequence_with_blank_length,width), dtype='float32')
    cdef float[:,:] ab = np.empty((sequence_with_blank_length,width), dtype='float32')
    # cdef float[:,:] grad = np.zeros((numphones,T), dtype='float32')
    cdef float[:,:] grad_v = grad
    cdef float[:] absum = np.empty(width, dtype='float32')

    cdef unsigned int t, s, l
    cdef float c, llForward, llBackward, llDiff, tmp

    try:
        # Initialize alphas and forward pass
        alphas[0,0] = prediction[blank,0]
        alphas[1,0] = prediction[sequence[0],0]
        c = alphas[0,0] + alphas[1,0]
        alphas[0,0] = alphas[0,0] / c
        alphas[1,0] = alphas[1,0] / c
        llForward = math.log(c)

        for t in range(1, width):
            for s in range(0,sequence_with_blank_length):
                l = (s-1)/2
                # blank
                if s%2 == 0:
                    if s==0:
                        alphas[s,t] = alphas[s,t-1] * prediction[blank,t]
                    else:
                        alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * prediction[blank,t]
                # same label twice
                elif s == 1 or sequence[l] == sequence[l-1]:
                    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * prediction[sequence[l],t]
                else:
                    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1] + alphas[s-2,t-1]) \
                                  * prediction[sequence[l],t]

            # normalize at current time (prevent underflow)
            c = 0.0
            for s in range(0,sequence_with_blank_length):
                c += alphas[s,t]
            for s in range(0,sequence_with_blank_length):
                alphas[s,t] = alphas[s,t] / c
            llForward += math.log(c)

        # Initialize betas and backwards pass
        betas[sequence_with_blank_length-1,width-1] = prediction[blank,width-1]
        betas[sequence_with_blank_length-2,width-1] = prediction[sequence[sequence_length-1],width-1]
        c = betas[sequence_with_blank_length-1,width-1] + betas[sequence_with_blank_length-2,width-1]
        betas[sequence_with_blank_length-1,width-1] = betas[sequence_with_blank_length-1,width-1] / c
        betas[sequence_with_blank_length-2,width-1] = betas[sequence_with_blank_length-2,width-1] / c
        llBackward = math.log(c)
        for t in range(width-1,0,-1):
            t = t - 1
            for s in range(sequence_with_blank_length,0,-1):
                s = s-1
                l = (s-1)/2
                # blank
                if s%2 == 0:
                    if s == sequence_with_blank_length-1:
                        betas[s,t] = betas[s,t+1] * prediction[blank,t]
                    else:
                        betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * prediction[blank,t]
                # same label twice
                elif s == sequence_with_blank_length-2 or sequence[l] == sequence[l+1]:
                    betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * prediction[sequence[l],t]
                else:
                    betas[s,t] = (betas[s,t+1] + betas[s+1,t+1] + betas[s+2,t+1]) \
                                 * prediction[sequence[l],t]

            c = 0.0
            for s in range(0,sequence_with_blank_length):
                c += betas[s,t]
            for s in range(0,sequence_with_blank_length):
                betas[s,t] = betas[s,t] / c
            llBackward += math.log(c)

        # Compute gradient with respect to unnormalized input parameters
        for t in range(width):
            for s in range(sequence_with_blank_length):
                ab[s,t] = alphas[s,t]*betas[s,t]
        for s in range(sequence_with_blank_length):
            # blank
            if s%2 == 0:
                for t in range(width):
                    grad_v[blank,t] += ab[s,t]
                    if ab[s,t] != 0:
                        ab[s,t] = ab[s,t]/prediction[blank,t]
            else:
                for t in range(width):
                    grad_v[sequence[(s-1)/2],t] += ab[s,t]
                    if ab[s,t] != 0:
                        ab[s,t] = ab[s,t]/(prediction[sequence[(s-1)/2],t])

        for t in range(width):
            absum[t] = 0
            for s in range(sequence_with_blank_length):
                absum[t] += ab[s,t]

        # grad = params - grad / (params * absum)
        for t in range(width):
            for s in range(numphones):
                tmp = (prediction[s,t]*absum[t])
                if tmp > 0:
                    grad_v[s,t] = prediction[s,t] - grad_v[s,t] / tmp
                else:
                    grad_v[s,t] = prediction[s,t]

    except (FloatingPointError,ZeroDivisionError) as e:
        print_error("Zero Division in CTC")
        return -llForward


    return -llForward


cpdef parallal_np_ctc(ctx, t_output, list label, blank, is_gpu, width_transform):
    output = t_output.cpu().detach().numpy().astype('float32')

    cdef int batch_size = output.shape[0]

    cdef float[:] cost = np.zeros((batch_size), dtype='float32')
    cdef float[:,:,:] grad = np.zeros(output.shape, dtype='float32')

    cdef int i
    cdef int w

    for i in range(0, batch_size):
        w = width_transform(label[i][2])
        cost[i] = np_ctc(output[i], label[i][0].cpu().detach().numpy().astype('int32'), blank, grad[i], width_transform(label[i][2])) / w

    total_cost = torch.autograd.Variable(torch.from_numpy(np.array(cost))).float()
    total_gradient = torch.autograd.Variable(torch.from_numpy(np.array(grad))).float()

    if is_gpu:
        total_cost = total_cost.cuda()
        total_gradient = total_gradient.cuda()

    ctx.save_for_backward(total_gradient)

    return total_cost.mean()

class CTCFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, t_output, label, blank, is_gpu, width_transform):
        return parallal_np_ctc(ctx, t_output, label, blank, is_gpu, width_transform)

    @staticmethod
    def backward(ctx, grad_output):
        gradient, = ctx.saved_tensors
        return gradient, None, None, None, None


class CTC(torch.nn.Module):

    def __init__(self, labels, width_transform):
        super().__init__()

        self.labels = labels
        self.inv_labels = {v: k for k, v in self.labels.items()}
        self.width_transform = width_transform
        self.label_len = max(labels.values()) + 1
        self.softmax = torch.nn.Softmax(dim=2)
        self.ctc = CTCFunction()
        self.is_gpu = False
        self.decoder = CTCDecoder(self.inv_labels, self.label_len)

    def forward(self, output, label):
        output = self.softmax(output)
        output = output.transpose(1, 2)

        return self.ctc.apply(output, label, self.labels[""], self.is_gpu, self.width_transform)

    def cuda(self, **kwargs):
        self.is_gpu = True
        print_warning("CTC do not support GPU and will be processed on CPU")
        return self

    def cpu(self):
        self.is_gpu = False
        print_normal("Using CTCLoss with CPU")
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
            return [self.labels[""]]

        return text_label

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
