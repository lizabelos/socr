import copy
from random import randint

import torch
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.nn import Parameter

from socr.models.convolutional_model import ConvolutionalModel
from socr.models.loss.x_height_cc_loss import XHeightCCLoss
from socr.nn.modules.binarize import Binarize
from socr.nn.modules.lstm import SRU2D, Lstm2D
from socr.nn.modules.resnet import PSPUpsample, BasicBlock, Bottleneck
from socr.utils.logging.logger import print_normal


class fasterSegment(ConvolutionalModel):

    def __init__(self):
        super(fasterSegment, self).__init__()

        self.inplanes = 64

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.act1 = torch.nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)

        self.up1 = PSPUpsample(384, 256, bn=True)
        self.up2 = PSPUpsample(320, 128, bn=True)
        self.up3 = PSPUpsample(128 + 64, 64, bn=True)
        self.up4 = PSPUpsample(64 + 3, 32, bn=True)

        self.last_conv_prob_2 = torch.nn.Conv2d(32, 2, kernel_size=1)
        self.last_act_prob_2 = torch.nn.ReLU(inplace=True)

        # self.thresh = torch.nn.Parameter(torch.Tensor([0.0]), requires_grad=False)

        print_normal("Applying xavier initialization...")
        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight.data)

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
        x_2 = self.bn1(x_2)
        x_2 = self.act1(x_2)

        x_3 = self.layer1(x_2)
        x_4 = self.layer2(x_3)
        x_5 = self.layer3(x_4)

        x_4 = self.up1(x_5, addition=x_4)
        x_3 = self.up2(x_4, addition=x_3)
        x_2 = self.up3(x_3, addition=x_2)
        x_1 = self.up4(x_2, addition=x_1)

        x_prob = self.last_conv_prob_2(x_1)
        x_prob = self.last_act_prob_2(x_prob)

        return x_prob

    def forward_fc(self, x):
        return x

    def get_input_image_width(self):
        return None

    def get_input_image_height(self):
        return None

    def create_loss(self):
        return XHeightCCLoss()

    def adaptative_learning_rate(self, optimizer):
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98)

    def collate(self, batch):
        data = [item[0] for item in batch]  # just form a list of tensor
        label = [item[1] for item in batch]

        min_width = min([d.size()[1] for d in data])
        min_height = min([d.size()[0] for d in data])

        min_width = min(min_width, 512)
        min_height = min(min_height, 512)

        new_data = []
        new_label = []

        for i in range(0, len(data)):
            d = data[i]

            crop_x = randint(0, d.size()[1] - min_width)
            crop_y = randint(0, d.size()[0] - min_height)
            d = d[crop_y:crop_y + min_height, crop_x:crop_x + min_width]
            d = torch.transpose(d, 0, 2)
            d = torch.transpose(d, 1, 2)
            new_data.append(d)

            d = label[i]
            d = d[crop_y:crop_y + min_height, crop_x:crop_x + min_width]
            d = torch.transpose(d, 0, 2)
            d = torch.transpose(d, 1, 2)
            new_label.append(d)

        data = torch.stack(new_data)
        label = torch.stack(new_label)

        return [data, label]
