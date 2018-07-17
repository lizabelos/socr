import copy
from random import randint

import torch
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.nn import Parameter

from socr.models.convolutional_model import ConvolutionalModel
from socr.models.loss.x_height_cc_loss import XHeightCCLoss
from socr.nn.modules.binarize import Binarize
from socr.nn.modules.resnet import PSPUpsample, BasicBlock, Bottleneck
from socr.utils.logging.logger import print_normal


class dhSegment(ConvolutionalModel):

    def __init__(self):
        super(dhSegment, self).__init__()

        self.inplanes = 64

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.act1 = torch.nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        self.layer4_reduce = torch.nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.layer4_reduce_bn = torch.nn.BatchNorm2d(512)
        self.layer4_reduce_act = torch.nn.ReLU(inplace=True)

        self.layer3_reduce = torch.nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.layer3_reduce_bn = torch.nn.BatchNorm2d(512)
        self.layer3_reduce_act = torch.nn.ReLU(inplace=True)

        self.up1 = PSPUpsample(512 + 512, 512, bn=True)
        self.up2 = PSPUpsample(512 + 512, 256, bn=True)
        self.up3 = PSPUpsample(256 + 64, 128, bn=True)
        self.up4 = PSPUpsample(128 + 3, 64, bn=True)

        self.last_conv_prob =  torch.nn.Conv2d(64, 2, kernel_size=(1, 1), dilation=(1, 1), padding=0, bias=True)
        self.last_act_prob = torch.nn.ReLU(inplace=True)

        # self.thresh = torch.nn.Parameter(torch.Tensor([0.0]), requires_grad=False)

        print_normal("Applying xavier initialization...")
        self.apply(self.weights_init)

        print_normal("Downloading pretrained model from pytorch model zoo...")
        pretrained_model = model_zoo.load_url("https://download.pytorch.org/models/resnet50-19c8e357.pth")

        print_normal("Loading pretrained resnet...")
        self.load_my_state_dict(pretrained_model)

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
        x_3 = self.layer2(x_3)
        x_4 = self.layer3(x_3)
        x_5 = self.layer4(x_4)

        x_5 = self.layer4_reduce(x_5)
        x_5 = self.layer4_reduce_bn(x_5)
        x_5 = self.layer4_reduce_act(x_5)

        x_4 = self.layer3_reduce(x_4)
        x_4 = self.layer3_reduce_bn(x_4)
        x_4 = self.layer3_reduce_act(x_4)

        x_4 = self.up1(x_5, addition=x_4)
        x_3 = self.up2(x_4, addition=x_3)
        x_2 = self.up3(x_3, addition=x_2)
        x_1 = self.up4(x_2, addition=x_1)

        x_prob = self.last_conv_prob(x_1)
        x_prob = self.last_act_prob(x_prob)

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
            # d = torch.transpose(d, 0, 2)
            # d = torch.transpose(d, 0, 1)
            d = d[crop_y:crop_y + min_height, crop_x:crop_x + min_width]
            d = torch.transpose(d, 0, 2)
            d = torch.transpose(d, 1, 2)
            new_data.append(d)

            d = label[i]

            # d = torch.transpose(d, 0, 2)
            # d = torch.transpose(d, 0, 1)
            d = d[crop_y:crop_y + min_height, crop_x:crop_x + min_width]
            d = torch.transpose(d, 0, 2)
            d = torch.transpose(d, 1, 2)
            new_label.append(d)

        data = torch.stack(new_data)
        label = torch.stack(new_label)


        # max_width = max([d.size()[2] for d in data])
        # max_height = max([d.size()[1] for d in data])
        #
        # data = [torch.nn.functional.pad(d, (0, max_width - d.size()[2], 0, max_height - d.size()[1])) for d in data]
        # data = torch.stack(data)
        #
        # label = [torch.nn.functional.pad(d, (0, max_width - d.size()[2], 0, max_height - d.size()[1])) for d in label]
        # label = torch.stack(label)

        return [data, label]