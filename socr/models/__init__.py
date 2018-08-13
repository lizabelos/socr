import torch

from .resSru import resSru
from .dhSegment import dhSegment


def get_model_by_name(name):
    return {
        "resSru": resSru,
        "dhSegment": dhSegment,
    }[name]


def get_optimizer_by_name(name):
    return {
        "SGD": torch.optim.SGD,
        "RMSProp": torch.optim.RMSprop,
        "Adam": torch.optim.Adam
    }[name]
