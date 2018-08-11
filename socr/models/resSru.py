from collections import OrderedDict

import torch

from socr.models.loss.ctc import CTC
from socr.nn import IndRNN
from socr.nn.modules.resnet import ResNet, Bottleneck, BasicBlock
from socr.utils.setup.build import install_and_import_sru

sru = install_and_import_sru()

from socr.models.convolutional_model import ConvolutionalModel
from socr.models.loss import CTCTextLoss


class resSru(ConvolutionalModel):

    def __init__(self, labels):
        super().__init__()

        self.labels = labels
        self.output_numbers = len(labels)

        self.activation = torch.nn.ReLU()

        self.convolutions = torch.nn.Sequential(OrderedDict([
            ('conv1', torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', torch.nn.BatchNorm2d(64)),
            ('activation', torch.nn.ReLU(inplace=True)),
            ('maxpool', torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=(1, 1))),
            ('resnet', ResNet(BasicBlock, [2, 2, 2, 2], strides=[1, (2, 1), (2, 1), (2, 1)], bn=True)),
        ]))
        self.convolutions_output_size = self.get_cnn_output_size()

        self.rnn = torch.nn.GRU(self.convolutions_output_size[1] * self.convolutions_output_size[2], 256, num_layers=2, bidirectional=True, dropout=0.3)
        # self.rnn = IndRNN(self.convolutions_output_size[1] * self.convolutions_output_size[2], 128, n_layer=3, bidirectional=True, batch_norm=True)

        # print(self.convolutions_output_size)

        # self.rnn = sru.SRU(self.convolutions_output_size[1] * self.convolutions_output_size[2], 256, num_layers=6,
        #                    bidirectional=True, rnn_dropout=0.3, use_tanh=1, use_relu=0, layer_norm=False, weight_norm=True)

        self.fc = torch.nn.Linear(2 * 256, self.output_numbers)

        self.softmax = torch.nn.Softmax(dim = 2)

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

        x, _ = self.rnn(x)
        x = self.fc(x)

        # if not self.training:
        x = x.transpose(0, 1)
        x = self.softmax(x)

        return x

    def get_input_image_width(self):
        return None

    def get_input_image_height(self):
        return 64

    def create_loss(self):
        return CTC(self.labels, lambda x: self.conv_output_size(self.conv_output_size(x, 7, 3, 2), 3, 1, 2))

    def conv_output_size(self, width, filter_size, padding, stride):
        return (width - filter_size + 2 * padding) / stride + 1

    def adaptative_learning_rate(self, optimizer):
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98)
        # return torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
