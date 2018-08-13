import numpy as np
import torch

from socr.utils.logging.logger import print_normal, print_warning, print_error


# From https://raw.githubusercontent.com/amaas/stanford-ctc/master/ctc/ctc.py
def np_ctc(params, seq, blank=0, is_prob=True):
    """
    CTC loss function.
    params - n x m matrix of n-D probability distributions over m frames.
    seq - sequence of phone id's for given example.
    is_prob - whether params have already passed through a softmax
    Returns objective and gradient.
    """
    params = np.transpose(params, (1, 0))

    seqLen = seq.shape[0]  # Length of label sequence (# phones)
    numphones = params.shape[0]  # Number of labels
    L = 2 * seqLen + 1  # Length of label sequence with blanks
    T = params.shape[1]  # Length of utterance (time)

    alphas = np.zeros((L, T))
    betas = np.zeros((L, T))

    # Keep for gradcheck move this, assume NN outputs probs
    if not is_prob:
        params = params - np.max(params, axis=0)
        params = np.exp(params)
        params = params / np.sum(params, axis=0)

    # Initialize alphas and forward pass
    alphas[0, 0] = params[blank, 0]
    alphas[1, 0] = params[seq[0], 0]
    c = np.sum(alphas[:, 0])
    alphas[:, 0] = alphas[:, 0] / c
    llForward = np.log(c)
    for t in range(1, T):
        start = max(0, L - 2 * (T - t))
        end = min(2 * t + 2, L)
        for s in range(start, L):
            l = (s - 1) // 2
            # blank
            if s % 2 == 0:
                if s == 0:
                    alphas[s, t] = alphas[s, t - 1] * params[blank, t]
                else:
                    alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[blank, t]
            # same label twice
            elif s == 1 or seq[l] == seq[l - 1]:
                alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * params[seq[l], t]
            else:
                alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1] + alphas[s - 2, t - 1]) \
                               * params[seq[l], t]

        # normalize at current time (prevent underflow)
        c = np.sum(alphas[start:end, t])
        alphas[start:end, t] = alphas[start:end, t] / c
        llForward += np.log(c)

    # Initialize betas and backwards pass
    betas[-1, -1] = params[blank, -1]
    betas[-2, -1] = params[seq[-1], -1]
    c = np.sum(betas[:, -1])
    betas[:, -1] = betas[:, -1] / c
    llBackward = np.log(c)
    for t in range(T - 2, -1, -1):
        start = max(0, L - 2 * (T - t))
        end = min(2 * t + 2, L)
        for s in range(end - 1, -1, -1):
            l = (s - 1) // 2
            # blank
            if s % 2 == 0:
                if s == L - 1:
                    betas[s, t] = betas[s, t + 1] * params[blank, t]
                else:
                    betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1]) * params[blank, t]
            # same label twice
            elif s == L - 2 or seq[l] == seq[l + 1]:
                betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1]) * params[seq[l], t]
            else:
                betas[s, t] = (betas[s, t + 1] + betas[s + 1, t + 1] + betas[s + 2, t + 1]) \
                              * params[seq[l], t]

        c = np.sum(betas[start:end, t])
        betas[start:end, t] = betas[start:end, t] / c
        llBackward += np.log(c)

    # Compute gradient with respect to unnormalized input parameters
    grad = np.zeros(params.shape)
    ab = alphas * betas
    for s in range(L):
        # blank
        if s % 2 == 0:
            grad[blank, :] += ab[s, :]
            ab[s, :] = ab[s, :] / params[blank, :]
        else:
            grad[seq[(s - 1) // 2], :] += ab[s, :]
            ab[s, :] = ab[s, :] / (params[seq[(s - 1) // 2], :])
    absum = np.sum(ab, axis=0)

    # Check for underflow or zeros in denominator of gradient
    llDiff = np.abs(llForward - llBackward)
    if llDiff > 1e-5 or np.sum(absum == 0) > 0:
        print("Diff in forward/backward LL : %f" % llDiff)
        print("Zeros found : (%d/%d)" % (np.sum(absum == 0), absum.shape[0]))
        return -llForward, np.transpose(grad, (1, 0)), True

    grad = params - grad / (params * absum)

    return -llForward, np.transpose(grad, (1, 0)), False


class CTCFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, t_output, label, blank, is_gpu):
        output = t_output.cpu().detach().numpy()
        label = label.cpu().detach().numpy().astype(int)

        gradients = []
        costs = []

        for i in range(0, output.shape[0]):
            cost, gradient, _ = np_ctc(output[i], label[i], blank)
            gradients.append(gradient)
            costs.append(cost)

        total_cost = np.stack(costs)
        total_gradient = np.stack(gradients)

        total_cost = torch.autograd.Variable(torch.from_numpy(total_cost)).float()
        total_gradient = torch.autograd.Variable(torch.from_numpy(total_gradient)).float()

        if is_gpu:
            total_cost = total_cost.cuda()
            total_gradient = total_gradient.cuda()

        ctx.save_for_backward(total_gradient)

        return total_cost.mean()

    @staticmethod
    def backward(ctx, grad_output):
        gradient, = ctx.saved_tensors
        return gradient, None, None, None


class CTC(torch.autograd.Function):

    def __init__(self, labels, width_transform):
        super().__init__()

        self.labels = labels
        self.inv_labels = {v: k for k, v in self.labels.items()}
        self.width_transform = width_transform
        self.label_len = max(labels.values()) + 1
        self.softmax = torch.nn.Softmax(dim=2)
        self.ctc = CTCFunction()
        self.is_gpu = False

    def forward(self, output, label):
        label = label[0]
        output = self.softmax(output)

        return self.ctc.apply(output, label, self.labels[""], self.is_gpu)

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
        var = torch.autograd.Variable(labels[0]).float()
        if is_cuda:
            var = var.cuda()
        return var, labels[1]

    def ytrue_to_lines(self, sequence):
        # OUTPUT : batch_size x width x num_label
        width = sequence.shape[1]
        batch_size = sequence.shape[0]

        text = ""
        last_label = -1

        for time in range(0, width):

            max_label = 0
            for i in range(1, self.label_len):
                if sequence[0][time][i] > sequence[0][time][max_label]:
                    max_label = i

            if max_label != last_label:
                if max_label not in self.inv_labels:
                    print_warning("Invalid label during decoding : " + str(max_label))
                else:
                    text = text + self.inv_labels[max_label]
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
