from collections import OrderedDict

import torch

from socr.models.convolutional_model import ConvolutionalModel
from socr.models.loss.x_height_cc_loss import XHeightCCLoss
from socr.nn import Lstm2D


class XHeightLabelingModel(ConvolutionalModel):
    """
    This network architecture comes from 'Handwritten text line segmentation using Fully Convolutional Network' (2017)
    http://www.teklia.com/wp-content/uploads/2018/01/3586f005.pdf
    """

    def __init__(self):
        super().__init__()

        self.activation = torch.nn.ReLU(inplace=True)

        self.convolutions = torch.nn.Sequential(OrderedDict([
            ('conv1', torch.nn.Conv2d(3, 64, kernel_size=(3, 3), dilation=(1, 1), padding=1)),
            ('activation1', self.activation),

            ('conv2', torch.nn.Conv2d(64, 64, kernel_size=(3, 3), dilation=(1, 1), padding=1)),
            ('activation2', self.activation),

            ('conv3', torch.nn.Conv2d(64, 128, kernel_size=(3, 3), dilation=(2, 2), padding=2)),
            ('activation3', self.activation),

            ('conv4', torch.nn.Conv2d(128, 128, kernel_size=(3, 3), dilation=(2, 2), padding=2)),
            ('activation4', self.activation),

            ('conv5', torch.nn.Conv2d(128, 256, kernel_size=(3, 3), dilation=(4, 4), padding=4)),
            ('activation5', self.activation),

            ('conv6', torch.nn.Conv2d(256, 256, kernel_size=(3, 3), dilation=(4, 4), padding=4)),
            ('activation6', self.activation),

            ('lstm', Lstm2D(256, 2, 256, 256, ksize=1)),
            ('activation7', self.activation),

            ('conv7', torch.nn.Conv2d(4, 2, kernel_size=(1, 1), dilation=(1, 1), padding=0)),
            # ('activation8', torch.nn.Sigmoid())
        ]))

        print(self.convolutions)

    def forward_cnn(self, x):
        x = self.convolutions(x)
        return x

    def forward_fc(self, x):
        return x

    def get_input_image_width(self):
        return None

    def get_input_image_height(self):
        return None

    def create_loss(self):
        return XHeightCCLoss()