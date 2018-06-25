from abc import abstractmethod

import torch


class Model(torch.nn.Module):

    @abstractmethod
    def get_input_image_width(self): pass

    @abstractmethod
    def get_input_image_height(self): pass

    def get_input_image_channels(self):
        return 3

    @abstractmethod
    def create_loss(self): pass