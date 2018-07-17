import torch
from torch.autograd import Function, Variable


class Binarize(Function):

    @staticmethod
    def forward(ctx, x):
        positives = torch.ge(x, 0)
        negatives = torch.le(x, 0)
        le0xmin1 = torch.mul(negatives.float(), -1)
        binary_output = positives.float() + le0xmin1.float()
        ctx.save_for_backward(x)
        return binary_output

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, = grad_outputs
        input, = ctx.saved_tensors
        grad_input = torch.nn.functional.tanh(Variable(input)).data * grad_output
        return grad_input