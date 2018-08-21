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
cdef np_ctc(double[:,:] prediction, int[::1] sequence, unsigned int blank, double[:,:] grad, unsigned int width):
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

    cdef double[:,:] alphas = np.zeros((sequence_with_blank_length,width), dtype='float64')
    cdef double[:,:] betas = np.zeros((sequence_with_blank_length,width), dtype='float64')
    cdef double[:,:] ab = np.empty((sequence_with_blank_length,width), dtype='float64')
    # cdef double[:,:] grad = np.zeros((numphones,T), dtype='float64')
    cdef double[:,:] grad_v = grad
    cdef double[:] absum = np.empty(width, dtype='float64')

    cdef unsigned int t, s, l
    cdef double c, llForward, llBackward, llDiff, tmp

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
                    grad_v[s,t] = (prediction[s,t] - grad_v[s,t] / tmp) / sequence_length
                else:
                    grad_v[s,t] = prediction[s,t] / sequence_length

    except (FloatingPointError,ZeroDivisionError) as e:
        print_error("Zero Division in CTC")
        return -llForward


    return -llForward



cdef inline log_add2(double a, double b):
    result = math.log(math.exp(a) + math.exp(b))
    if math.isnan(result):
        raise FloatingPointError("Logarithm addition (2) failed")
    return result

cdef inline log_sub2(double a, double b):
    result = math.log(math.exp(a) - math.exp(b))
    if math.isnan(result):
        raise FloatingPointError("Logarithm substraction (2) failed")
    return result

cdef inline log_add3(double a, double b, double c):
    result = math.log(math.exp(a) + math.exp(b) + math.exp(c))
    if math.isnan(result):
        raise FloatingPointError("Logarithm addition (3) failed")
    return result

cdef inline log_mul2(double a, double b):
    return a + b

cdef inline log_div2(double a, double b):
    if b == 0:
        raise ZeroDivisionError("Division by zero")
    return a - b

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np_ctc_log(double[:,:] prediction, int[::1] sequence, unsigned int blank, double[:,:] grad, unsigned int width):
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

    cdef double[:,:] alphas = np.full((sequence_with_blank_length,width), -np.inf, dtype='float64')
    cdef double[:,:] betas = np.full((sequence_with_blank_length,width), -np.inf, dtype='float64')
    cdef double[:,:] ab = np.full((sequence_with_blank_length,width), -np.inf, dtype='float64')
    # cdef double[:,:] grad = np.zeros((numphones,T), dtype='float64')
    cdef double[:,:] grad_v = grad
    cdef double[:] absum = np.full(width, -np.inf, dtype='float64')

    cdef unsigned int t, s, l
    cdef double c, llForward, llBackward, llDiff, tmp

    try:
        # Initialize alphas and forward pass
        alphas[0,0] = prediction[blank,0]
        alphas[1,0] = prediction[sequence[0],0]
        c = log_add2(alphas[0,0],alphas[1,0])
        alphas[0,0] = log_div2(alphas[0,0], c)
        alphas[1,0] = log_div2(alphas[1,0], c)
        llForward = c

        for t in range(1, width):
            for s in range(0,sequence_with_blank_length):
                l = (s-1)/2
                # blank
                if s%2 == 0:
                    if s==0:
                        alphas[s,t] = log_mul2(alphas[s,t-1], prediction[blank,t])
                    else:
                        alphas[s,t] = log_mul2(log_add2(alphas[s,t-1],alphas[s-1,t-1]),prediction[blank,t])
                # same label twice
                elif s == 1 or sequence[l] == sequence[l-1]:
                    alphas[s,t] = log_mul2(log_add2(alphas[s,t-1],alphas[s-1,t-1]),prediction[sequence[l],t])
                else:
                    alphas[s,t] = log_mul2(log_add3(alphas[s,t-1],alphas[s-1,t-1],alphas[s-2,t-1]),prediction[sequence[l],t])

            # normalize at current time (prevent underflow)
            c = alphas[0,t]
            for s in range(1,sequence_with_blank_length):
                c = log_add2(c,alphas[s,t])
            for s in range(0,sequence_with_blank_length):
                alphas[s,t] = log_div2(alphas[s,t], c)
            llForward += c

    except (FloatingPointError,ZeroDivisionError) as e:
        print_error("During alphas computation, " + str(e))
        return -llForward

    try:

        # Initialize betas and backwards pass
        betas[sequence_with_blank_length-1,width-1] = prediction[blank,width-1]
        betas[sequence_with_blank_length-2,width-1] = prediction[sequence[sequence_length-1],width-1]
        c = log_add2(betas[sequence_with_blank_length-1,width-1], betas[sequence_with_blank_length-2,width-1])
        betas[sequence_with_blank_length-1,width-1] = log_div2(betas[sequence_with_blank_length-1,width-1], c)
        betas[sequence_with_blank_length-2,width-1] = log_div2(betas[sequence_with_blank_length-2,width-1], c)
        llBackward = c
        for t in range(width-1,0,-1):
            t = t - 1
            for s in range(sequence_with_blank_length,0,-1):
                s = s-1
                l = (s-1)/2
                # blank
                if s%2 == 0:
                    if s == sequence_with_blank_length-1:
                        betas[s,t] = log_mul2(betas[s,t+1], prediction[blank,t])
                    else:
                        betas[s,t] = log_mul2(log_add2(betas[s,t+1], betas[s+1,t+1]), prediction[blank,t])
                # same label twice
                elif s == sequence_with_blank_length-2 or sequence[l] == sequence[l+1]:
                    betas[s,t] = log_mul2(log_add2(betas[s,t+1], betas[s+1,t+1]), prediction[sequence[l],t])
                else:
                    betas[s,t] = log_mul2(log_add3(betas[s,t+1], betas[s+1,t+1], betas[s+2,t+1]),prediction[sequence[l],t])

            c = betas[0,t]
            for s in range(1,sequence_with_blank_length):
                c = log_add2(c,betas[s,t])
            for s in range(0,sequence_with_blank_length):
                betas[s,t] = log_div2(betas[s,t],c)
            llBackward += c

    except (FloatingPointError,ZeroDivisionError) as e:
        print_error("During betas computation, " + str(e))
        return -llForward

    try:

        # Compute gradient with respect to unnormalized input parameters
        for t in range(width):
            for s in range(sequence_with_blank_length):
                ab[s,t] = log_mul2(alphas[s,t],betas[s,t])

        for t in range(width):
            for s in range(numphones):
                grad_v[s,t] = -np.inf

        for s in range(sequence_with_blank_length):
            # blank
            if s%2 == 0:
                for t in range(width):
                    grad_v[blank,t] = log_add2(grad_v[blank,t],ab[s,t])
                    if ab[s,t] != -np.inf:
                        ab[s,t] = log_div2(ab[s,t],prediction[blank,t])
            else:
                for t in range(width):
                    grad_v[sequence[(s-1)/2],t] = log_add2(grad_v[sequence[(s-1)/2],t], ab[s,t])
                    if ab[s,t] != -np.inf:
                        ab[s,t] = log_div2(ab[s,t],(prediction[sequence[(s-1)/2],t]))

        for t in range(width):
            absum[t] = 0
            for s in range(sequence_with_blank_length):
                if ab[s,t] != -np.inf:
                    absum[t] = log_add2(absum[t], ab[s,t])

        # grad = params - grad / (params * absum)
        for t in range(width):
            for s in range(numphones):
                tmp = log_mul2(prediction[s,t],absum[t])
                if tmp != -np.inf:
                    grad_v[s,t] = (math.exp(prediction[s,t]) - math.exp(log_div2(grad_v[s,t], tmp))) / sequence_length
                else:
                    grad_v[s,t] = math.exp(prediction[s,t]) / sequence_length


    except (FloatingPointError,ZeroDivisionError) as e:
        print_error("During gradient computation, " + str(e))
        return -llForward


    return -llForward


cpdef parallal_np_ctc(ctx, t_output, list label, blank, bint is_gpu, width_transform, bint log_space):
    output = t_output.cpu().detach().numpy().astype('float64')

    cdef int batch_size = output.shape[0]

    cdef double[:] cost = np.zeros((batch_size), dtype='float64')
    cdef double[:,:,:] grad = np.zeros(output.shape, dtype='float64')

    cdef int i
    cdef int w

    for i in range(0, batch_size):
        w = width_transform(label[i][2])
        if log_space:
            cost[i] = np_ctc_log(output[i], label[i][0].cpu().detach().numpy().astype('int32'), blank, grad[i], w) / w
        else:
            cost[i] = np_ctc(output[i], label[i][0].cpu().detach().numpy().astype('int32'), blank, grad[i], w) / w

    total_cost = torch.autograd.Variable(torch.from_numpy(np.array(cost))).float()
    total_gradient = torch.autograd.Variable(torch.from_numpy(np.array(grad))).float()

    if is_gpu:
        total_cost = total_cost.cuda()
        total_gradient = total_gradient.cuda()

    total_gradient = total_gradient / batch_size

    ctx.save_for_backward(total_gradient)

    return total_cost.mean()

class CTCFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, output, label, blank, is_gpu, width_transform, log_space):
        if log_space:
            output = torch.nn.functional.log_softmax(output, dim=2)
        else:
            output = torch.nn.functional.softmax(output, dim=2)
        output = output.transpose(1, 2)
        return parallal_np_ctc(ctx, output, label, blank, is_gpu, width_transform, log_space)

    @staticmethod
    def backward(ctx, grad_output):
        gradient, = ctx.saved_tensors
        gradient = gradient.transpose(1, 2)
        if torch.isnan(gradient).any():
            raise FloatingPointError()
        return gradient, None, None, None, None, None


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
        return self.ctc.apply(output, label, self.labels[""], self.is_gpu, self.width_transform, False)

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
