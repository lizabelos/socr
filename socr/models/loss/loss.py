import torch
from abc import ABCMeta, abstractmethod


class Loss(torch.nn.Module):
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, predicted, y_true): pass

    @abstractmethod
    def process_labels(self, labels, is_cuda=True): pass