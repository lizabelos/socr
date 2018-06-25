from collections import OrderedDict

import torch

from socr.models.convolutional_model import ConvolutionalModel
from socr.nn import RNNLayer
from socr.models.loss import CTCTextLoss


class BiggerConvLSTMNetwork(ConvolutionalModel):

    def __init__(self, labels):
        super().__init__()

        self.labels = labels
        self.output_numbers = len(labels) + 1

        self.activation = torch.nn.ReLU()

        self.convolutions = torch.nn.Sequential(OrderedDict([
            ('conv1-1', torch.nn.Conv2d(3, 64, kernel_size=3, bias=False)),
            ('activation1-1', self.activation),
            ('conv1-2', torch.nn.Conv2d(64, 64, kernel_size=3, bias=False)),
            ('activation1-2', self.activation),
            ('maxpool1', torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))),

            ('conv2-1', torch.nn.Conv2d(64, 128, kernel_size=3, bias=False)),
            ('activation2-1', self.activation),
            ('conv2-2', torch.nn.Conv2d(128, 128, kernel_size=3, bias=False)),
            ('activation2-2', self.activation),
            ('maxpool2', torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))),

            ('conv3-1', torch.nn.Conv2d(128, 256, kernel_size=3, bias=False)),
            ('activation3-1', self.activation),
            ('conv3-2', torch.nn.Conv2d(256, 256, kernel_size=3, bias=False)),
            ('activation3-2', self.activation),
            ('maxpool3', torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))),

            ('conv4-1', torch.nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)),
            ('activation4-1', self.activation),
            ('conv4-2', torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)),
            ('activation4-2', self.activation),
            ('maxpool4', torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))),
        ]))

        self.last_conv = torch.nn.Sequential(OrderedDict([
            ('conv5', torch.nn.Conv2d(512, self.output_numbers, kernel_size=(3, 1))),
            ('activation5', self.activation)
        ]))

        self.convolutions_output_size = self.get_cnn_output_size()

        self.lstm1 = RNNLayer(self.convolutions_output_size[1] * self.convolutions_output_size[2], self.output_numbers, rnn_type=torch.nn.GRU, bidirectional=False, batch_norm=False)
        # self.fc = torch.nn.Linear(self.output_numbers * 2, self.output_numbers)

        print(self.convolutions_output_size)
        print(self.convolutions)
        print(self.last_conv)
        print(self.lstm1)
        # print(self.lstm2)
        # print(self.fc)

    def forward_cnn(self, x):
        return self.convolutions(x)

    def forward_fc(self, x):
        batch_size = x.data.size()[0]
        channel_num = x.data.size()[1]
        height = x.data.size()[2]
        width = x.data.size()[3]

        x_conv = self.last_conv(x)
        x_conv = x_conv.view(batch_size, self.output_numbers, width)
        x_conv = torch.transpose(x_conv, 1, 2)

        x = x.view(batch_size, channel_num * height, width)
        x = torch.transpose(x, 1, 2)
        x_lstm = torch.transpose(x, 0, 1).contiguous()

        x_lstm = self.lstm1(x_lstm)
        x_lstm = self.activation(x_lstm)
        # x_lstm = self.lstm2(x_lstm)
        # x_lstm = self.activation(x_lstm)

        x_lstm = x_lstm.transpose(0, 1).contiguous()

        x = x_lstm + x_conv


        if not self.training:
            x = torch.nn.functional.softmax(x, dim=2)

        return x

    def get_input_image_width(self):
        return None

    def get_input_image_height(self):
        return 48

    def create_loss(self):
        return CTCTextLoss(self.labels)