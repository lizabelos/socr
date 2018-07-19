import torch


class CPUParallel(torch.nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *input):
        return self.module(*input)