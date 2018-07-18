import torch

from .resSru import resSru
from .conv_lstm_network import ConvLSTMNetwork
from .dhSegment import dhSegment
from .fasterSegment import fasterSegment


def get_model_by_name(name):
    return {
        "resSru": resSru,
        "dhSegment": dhSegment,
        "fasterSegment": fasterSegment
    }[name]


def get_optimizer_by_name(name):
    return {
        "SGD": torch.optim.SGD,
        "RMSProp": torch.optim.RMSprop,
        "Adam": torch.optim.Adam
    }[name]
