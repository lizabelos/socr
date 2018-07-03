from collections import OrderedDict

import torch

from socr.models.convolutional_model import ConvolutionalModel
from socr.models.loss.x_height_cc_loss import XHeightCCLoss
from socr.nn.modules.resnet import resnet50, ResNet, Bottleneck, PSPModule, PSPUpsample, BasicBlock


class XHeightResnetModel(ConvolutionalModel):

    def __init__(self):
        super().__init__()

        self.activation = torch.nn.ReLU(inplace=True)

        self.resnet = resnet50()

        self.convolutions = torch.nn.Sequential(OrderedDict([
            ('first', torch.nn.Conv2d(3, 64, kernel_size=7, padding=3, bias=False)),
            ('resnet', ResNet(BasicBlock, [2, 2, 2, 2], bn=False)),
            # ('pspmodule', PSPModule(512, 512)),
            ('up1', PSPUpsample(512, 256, bn=False)),
            ('drop1', torch.nn.Dropout2d(p=0.1)),
            ('up2', PSPUpsample(256, 64, bn=False)),
            ('drop2', torch.nn.Dropout2d(p=0.1)),
            ('up3', PSPUpsample(64, 64, bn=False)),
            ('drop3', torch.nn.Dropout2d(p=0.1)),
            ('final', torch.nn.Conv2d(64, 2, kernel_size=1)),
        ]))

        print(self.convolutions)

    def forward_cnn(self, x):
        h, w = x.size(2), x.size(3)
        x = self.convolutions(x)
        x = torch.nn.functional.upsample(input=x, size=(h, w), mode='bilinear')
        return x

    def forward_fc(self, x):
        return x

    def get_input_image_width(self):
        return None

    def get_input_image_height(self):
        return None

    def create_loss(self):
        return XHeightCCLoss()
