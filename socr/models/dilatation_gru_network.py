from collections import OrderedDict

import torch

from socr.models.convolutional_model import ConvolutionalModel
from socr.nn import RNNLayer
from socr.models.loss import CTCTextLoss


class DilatationGruNetwork(ConvolutionalModel):

    def __init__(self, labels):
        super().__init__()

        self.labels = labels
        self.output_numbers = len(labels) + 1

        self.activation = torch.nn.ReLU()

        self.convolutions = torch.nn.Sequential(OrderedDict([
            ('conv1-1', torch.nn.Conv2d(3, 64, kernel_size=3)),
            ('activation1-1', self.activation),
            ('conv1-2', torch.nn.Conv2d(64, 64, kernel_size=3)),
            ('activation1-2', self.activation),
            ('maxpool1', torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))),

            ('conv2-1', torch.nn.Conv2d(64, 128, kernel_size=3)),
            ('activation2-1', self.activation),
            ('conv2-2', torch.nn.Conv2d(128, 128, kernel_size=3)),
            ('activation2-2', self.activation),
            ('maxpool2', torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))),

            ('conv3-1', torch.nn.Conv2d(128, 256, kernel_size=3)),
            ('activation3-1', self.activation),
            ('conv3-2', torch.nn.Conv2d(256, 256, kernel_size=3)),
            ('activation3-2', self.activation),
            ('conv3-3', torch.nn.Conv2d(256, 256, kernel_size=3)),
            ('activation3-3', self.activation),
            ('maxpool3', torch.nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)))
        ]))
        self.convolutions_output_size = self.get_cnn_output_size()

        self.rnn = RNNLayer(self.convolutions_output_size[1] * self.convolutions_output_size[2], 256, num_layers=4, rnn_type=torch.nn.GRU, bidirectional=True, batch_norm=False, biadd=True, dropout=0.5)

        self.fc = torch.nn.Linear(256, self.output_numbers)

        print(self.convolutions_output_size)
        print(self.convolutions)
        print(self.rnn)
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

        x = self.rnn(x)
        x = self.activation(x)

        x = self.fc(x)
        x = self.activation(x)

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