from collections import OrderedDict

import torch

from socr.text.loss.ctc import CTC
from socr.text.modules.indrnn import IndRNN
from socr.text.modules.resnet import ResNet, Bottleneck, BasicBlock
from socr.utils.logger import print_normal


class resRnn(torch.nn.Module):

    def __init__(self, labels):
        super().__init__()

        self.labels = labels
        self.output_numbers =  max(labels.values()) + 1
        self.rnn_size = self.output_numbers

        print_normal("Creating resSru with " + str(self.output_numbers) + " labels")

        self.convolutions = torch.nn.Sequential(OrderedDict([
            ('conv1', torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', torch.nn.BatchNorm2d(64)),
            ('activation', torch.nn.ReLU(inplace=True)),
            ('maxpool', torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=(1, 1))),
            ('resnet', ResNet(BasicBlock, [2, 2, 2, 2], strides=[1, (2, 1), (2, 1), (2, 1)], bn=True)),
        ]))
        self.convolutions_output_size = self.get_cnn_output_size()

        self.rnn = IndRNN(self.convolutions_output_size[1] * self.convolutions_output_size[2], self.rnn_size, n_layer=3, bidirectional=True, batch_norm=True, batch_first=True, dropout=0.2, nonlinearity='tanh')
        self.fc = torch.nn.Linear(2 * self.rnn_size, self.output_numbers)

        self.softmax = torch.nn.Softmax(dim=2)

    def get_cnn_output_size(self):
        shape = [1, 3, self.get_input_image_height(), self.get_input_image_width()]
        if shape[3] is None:
            shape[3] = shape[2]
        input = torch.autograd.Variable(torch.rand(*shape))
        return self.forward_cnn(input).size()

    def forward(self, input):
        input = self.forward_cnn(input)
        input = self.forward_fc(input)
        return input

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

        x, _ = self.rnn(x)
        x = self.fc(x)
        # x = x.view(width, batch_size, self.output_numbers, 2)
        # x = torch.sum(x, dim=3)

        if not self.training:
            x = self.softmax(x)

        return x

    def get_input_image_width(self):
        return None

    def get_input_image_height(self):
        return 48

    def create_loss(self):
        return CTC(self.labels, lambda x: self.conv_output_size(self.conv_output_size(x, 7, 3, 2), 3, 1, 2))

    def conv_output_size(self, width, filter_size, padding, stride):
        return (width - filter_size + 2 * padding) / stride + 1

    def adaptative_learning_rate(self, optimizer):
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98)
        # return torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
