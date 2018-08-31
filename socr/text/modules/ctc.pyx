from libc cimport math
import cython
import numpy as np
np.seterr(divide='raise',invalid='raise')
import torch

from socr.utils.logger import print_error

# The original code comes from https://raw.githubusercontent.com/amaas/stanford-ctc/master/ctc/ctc.py

@cython.boundscheck(False)
@cython.wraparound(False)
cdef compute_ctc(double[:,:] prediction, int[::1] sequence, unsigned int blank, double[:,:] grad, unsigned int width):
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

cpdef batch_compute_ctc(ctx, t_output, list label, blank, bint is_gpu, width_transform):
    output = t_output.cpu().detach().numpy().astype('float64')

    cdef int batch_size = output.shape[0]

    cdef double[:] cost = np.zeros((batch_size), dtype='float64')
    cdef double[:,:,:] grad = np.zeros(output.shape, dtype='float64')

    cdef int i
    cdef int w

    for i in range(0, batch_size):
        w = width_transform(label[i][2])
        cost[i] = compute_ctc(output[i], label[i][0].cpu().detach().numpy().astype('int32'), blank, grad[i], w) / w

    total_cost = torch.autograd.Variable(torch.from_numpy(np.array(cost))).float()
    total_gradient = torch.autograd.Variable(torch.from_numpy(np.array(grad))).float()

    if is_gpu:
        total_cost = total_cost.cuda()
        total_gradient = total_gradient.cuda()

    total_gradient = total_gradient / batch_size

    ctx.save_for_backward(total_gradient)

    return total_cost

class CTCFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, output, label, blank, is_gpu, width_transform):
        output = torch.nn.functional.softmax(output, dim=2)
        output = output.transpose(1, 2)
        return batch_compute_ctc(ctx, output, label, blank, is_gpu, width_transform)

    @staticmethod
    def backward(ctx, grad_output):

        gradient, = ctx.saved_tensors
        gradient = gradient.transpose(1, 2)

        if torch.isnan(gradient).any():
            raise FloatingPointError()

        return gradient * grad_output.view(-1, 1, 1), None, None, None, None, None