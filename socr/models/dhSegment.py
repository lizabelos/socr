import copy

import torch
import torch.utils.model_zoo as model_zoo
from torch.nn import Parameter

from socr import print_normal
from socr.models.convolutional_model import ConvolutionalModel
from socr.models.loss.x_height_cc_loss import XHeightCCLoss
from socr.nn.modules.resnet import PSPUpsample, BasicBlock


class dhSegment(ConvolutionalModel):

    def __init__(self):
        super(dhSegment, self).__init__()

        self.inplanes = 64

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.up1 = PSPUpsample(512 + 256, 512, bn=True)
        self.up2 = PSPUpsample(512 + 128, 256, bn=True)
        self.up3 = PSPUpsample(256 + 64, 128, bn=True)
        self.up4 = PSPUpsample(128 + 3, 64, bn=True)

        print(self)

        print_normal("Downloading pretrained model from pytorch model zoo...")
        pretrained_model = model_zoo.load_url("https://download.pytorch.org/models/resnet18-5c106cde.pth")

        print_normal("Loading pretrained resnet...")
        self.load_my_state_dict(pretrained_model)

    def load_my_state_dict(self, state_dict):

        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride,
                                         bias=False)

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, bn=True))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn=True))

        return torch.nn.Sequential(*layers)

    def forward_cnn(self, x_1):

        # x_0 = S

        x_2 = self.conv1(x_1)  # S / 2

        x_3 = self.layer1(x_2)
        x_3 = self.layer2(x_3)
        x_4 = self.layer3(x_3)
        x_5 = self.layer4(x_4)

        x_4 = self.up1(x_5, addition=x_4)
        x_3 = self.up2(x_4, addition=x_3)
        x_2 = self.up3(x_3, addition=x_2)
        x_1 = self.up4(x_2, addition=x_1)

        return x_1

    def forward_fc(self, x):
        return x

    def get_input_image_width(self):
        return None

    def get_input_image_height(self):
        return None

    def create_loss(self):
        return XHeightCCLoss()
