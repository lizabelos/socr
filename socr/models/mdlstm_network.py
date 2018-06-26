from collections import OrderedDict

import torch

from socr.models.convolutional_model import ConvolutionalModel
from socr.nn import RNNLayer, Lstm2D
from socr.models.loss import CTCTextLoss


class MDLSTMNetwork(ConvolutionalModel):

    def __init__(self, labels):
        super().__init__()

        self.labels = labels
        self.output_numbers = len(labels) + 1

        self.activation = torch.nn.ReLU()

        self.convolutions = torch.nn.Sequential(OrderedDict([

            ('lstm3', Lstm2D(3, 64, 64, 64)),
            ('activation3', self.activation),
        ]))

        self.fc = torch.nn.Linear(48 * 128, self.output_numbers)

        print(self.convolutions)
        print(self.fc)

    def forward_cnn(self, x):
        return self.convolutions(x)

    def forward_fc(self, x):
        batch_size = x.data.size()[0]
        channel_num = x.data.size()[1]
        height = x.data.size()[2]
        width = x.data.size()[3]

        x = x.view(batch_size, channel_num * height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = self.fc(x)

        if not self.training:
            x = torch.nn.functional.softmax(x, dim=2)

        return x

    def get_input_image_width(self):
        return None

    def get_input_image_height(self):
        return 48

    def create_loss(self):
        return CTCTextLoss(self.labels)