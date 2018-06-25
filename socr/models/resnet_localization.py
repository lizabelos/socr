from collections import OrderedDict

import torch

from socr.nn.modules.resnet import resnet34
from socr.models.convolutional_model import ConvolutionalModel
from socr.models.loss import AbsoluteBoxLoss


class ResnetLocalization(ConvolutionalModel):

    def __init__(self, output_numbers):
        super().__init__()

        self.convolutions = resnet34(bn=False)

        cnn_output_size = self.get_cnn_output_size()

        self.linears = torch.nn.Sequential(OrderedDict([
            # Output
            ('fc1', torch.nn.Linear(cnn_output_size[1] * cnn_output_size[2] * cnn_output_size[3], 512)),
            ('fc2', torch.nn.Linear(512, 512)),
            ('fc3', torch.nn.Linear(512, output_numbers))
        ]))

        print(self.convolutions)
        print(self.linears)

    def forward_cnn(self, input):
        x, _ = self.convolutions(input)
        return x

    def forward_fc(self, input):
        return self.linears(input.view(input.data.size(0), -1))

    def get_input_image_width(self):
        return 512

    def get_input_image_height(self):
        return 512

    def create_loss(self):
        return AbsoluteBoxLoss()
