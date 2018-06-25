import torch
from abc import abstractmethod

from socr.models.model import Model


class ConvolutionalModel(Model):

    def get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def forward_cnn(self, input): pass

    @abstractmethod
    def forward_fc(self, input): pass

    def forward(self, input):
        return self.forward_fc(self.forward_cnn(input))

    def get_cnn_output_size(self):
        shape = [1, 3, self.get_input_image_height(), self.get_input_image_width()]
        if shape[3] is None:
            shape[3] = shape[2]
        input = torch.autograd.Variable(torch.rand(*shape))
        return self.forward_cnn(input).size()
