from collections import OrderedDict

import torch

from socr.models.convolutional_model import ConvolutionalModel
from socr.nn import RNNLayer
from socr.models.loss import CTCTextLoss


class ConvLSTMNetwork(ConvolutionalModel):

    def __init__(self, labels):
        super().__init__()

        self.labels = labels
        self.output_numbers = len(labels) + 1

        self.activation = torch.nn.ReLU()

        self.convolutions = torch.nn.Sequential(OrderedDict([
            ('conv1', torch.nn.Conv2d(3, 64, kernel_size=7, bias=False)),
            ('maxpool1', torch.nn.MaxPool2d(kernel_size=2, stride=(2, 1))),
            ('activation1', self.activation),

            ('conv2', torch.nn.Conv2d(64, 128, kernel_size=5, bias=False)),
            ('maxpool2', torch.nn.MaxPool2d(kernel_size=2, stride=(2, 1))),
            ('activation2', self.activation),

            ('conv3', torch.nn.Conv2d(128, 256, kernel_size=3, bias=False)),
            ('maxpool3', torch.nn.MaxPool2d(kernel_size=2, stride=(2, 1))),
            ('activation3', self.activation),

            ('conv4', torch.nn.Conv2d(256, 512, kernel_size=3, bias=False)),
            ('activation4', self.activation),
        ]))
        self.convolutions_output_size = self.get_cnn_output_size()

        self.lstm = RNNLayer(self.convolutions_output_size[1] * self.convolutions_output_size[2], 256, rnn_type=torch.nn.GRU, bidirectional=False, batch_norm=False, dropout=0.5)
        self.fc = torch.nn.Linear(256, self.output_numbers)

        print(self.convolutions_output_size)
        print(self.convolutions)
        print(self.lstm)
        print(self.fc)

    def forward_cnn(self, x):
        return self.convolutions(x)

    def forward_fc(self, x):
        batch_size = x.data.size()[0]
        channel_num = x.data.size()[1]
        height = x.data.size()[2]
        width = x.data.size()[3]

        x = x.view(batch_size, channel_num * height, width)
        # x is (batch_size x hidden_size x width)
        x = torch.transpose(x, 1, 2)
        # x is (batch_size x width x hidden_size)
        x = torch.transpose(x, 0, 1).contiguous()

        x = self.lstm(x)
        x = self.activation(x)

        x = self.fc(x)

        x = x.transpose(0, 1)

        if not self.training:
            x = torch.nn.functional.softmax(x, dim=2)

        return x

    def get_input_image_width(self):
        return None

    def get_input_image_height(self):
        return 48

    def create_loss(self):
        return CTCTextLoss(self.labels)