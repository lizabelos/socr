from collections import OrderedDict

import torch

from socr.models.convolutional_model import ConvolutionalModel
from socr.nn import ConvLayer
from socr.nn import MultiDimensionalLSTM
from socr.models.loss.line_semantic_segmentation_loss import LineSemanticSegmentationLoss


class LSTM2DLocalizationNetwork(ConvolutionalModel):

    def __init__(self):
        super().__init__()

        self.tanh = torch.nn.Tanh()

        self.convolutions = torch.nn.Sequential(OrderedDict([
            # ConvLayers
            ('convlayer1', ConvLayer(3, 16, 7, bn=False, maxpool=True)),
            ('convlayer2', ConvLayer(16, 32, 5, bn=False, maxpool=False)),
            ('convlayer3', ConvLayer(32, 64, 3, bn=False, maxpool=False)),
            ('convlayer4', ConvLayer(64, 64, 3, bn=False, maxpool=False)),
            ('mdlstm', MultiDimensionalLSTM(channels=64, win_dim=(4, 4), rnn_size=16))
        ]))
        self.convolutions_output_size = self.get_cnn_output_size()

        self.fc = torch.nn.Linear(self.convolutions_output_size[3], self.get_output_feature_len())

        print(self.convolutions)
        print(self.convolutions_output_size)
        print(self.fc)

    def forward_cnn(self, x):
        x = self.convolutions(x)
        # x = x.permute(0, 2, 3, 1)
        return x

    def forward_fc(self, x):
        x = self.fc(x)
        x = self.tanh(x)
        return x

    def get_input_image_width(self):
        return 512

    def get_input_image_height(self):
        return 512

    def get_output_width(self):
        return 64

    def get_output_height(self):
        return 64

    def get_output_feature_len(self):
        return 3

    def create_loss(self):
        return LineSemanticSegmentationLoss(sw=self.get_output_width(), sh=self.get_output_height())
