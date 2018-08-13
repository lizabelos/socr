from random import randint

import torch

from socr.line.codecs.baseline_decoder import BaselineDecoder
from socr.line.codecs.baseline_encoder import BaselineEncoder
from socr.utils.image import show_numpy_image
from socr.utils.image import show_connected_components, connected_components
from socr.utils.logger import print_normal


class XHeightCCLoss(torch.nn.Module):
    """An absolute position Loss"""

    def __init__(self, loss_type="mse", hysteresis_minimum=0.5, hysteresis_maximum=0.5, thicknesses=2,
                 height_importance=1.0):
        """

        :param s: grid division, assuming we have only 1 bounding box per cell
        """
        super().__init__()

        self.add_activation = None
        self.loss_type = loss_type
        if loss_type == "mse":
            print_normal("Using MSE Loss with Hysteresis=(" + str(hysteresis_minimum) + "," + str(hysteresis_maximum) + "), thicknesses=" + str(thicknesses) + ", height_importance=" + str(height_importance))
            self.mse = torch.nn.MSELoss()
        elif loss_type == "bce":
            print_normal("Using Binary Cross Entropy Loss Hysteresis=(" + str(hysteresis_minimum) + "," + str(hysteresis_maximum) + "), thicknesses=" + str(thicknesses) + ", height_importance=" + str(height_importance))
            self.mse = torch.nn.BCELoss()
            # self.mse = torch.nn.BCEWithLogitsLoss()
        elif loss_type == "norm":
            self.mse = None
        else:
            raise AssertionError
        self.mseh = torch.nn.MSELoss()

        self.hysteresis_minimum = hysteresis_minimum
        self.hysteresis_maximum = hysteresis_maximum
        self.thicknesses = thicknesses

        self.height_factor = 1.0
        self.height_importance = height_importance
        self.decoder = BaselineDecoder(self.height_factor, self.hysteresis_minimum, self.hysteresis_maximum)
        self.encoder = BaselineEncoder(self.height_factor, self.thicknesses)

    def forward(self, predicted, y_true):
        predicted = predicted.permute(1, 0, 2, 3).contiguous()
        y_true = y_true.permute(1, 0, 2, 3).contiguous()

        if self.height_importance == 0:
            if self.loss_type == "norm":
                return torch.mean(torch.abs(predicted[0] - y_true[0]))
            else:
                return self.mse(predicted[0], y_true[0])
        else:
            raise NotImplementedError()
            # return self.mse(predicted[0], y_true[0]) + (self.height_importance * self.mseh(predicted[1], y_true[1]))

    def document_to_ytrue(self, image_size, base_lines):
        return self.encoder.encode(image_size, base_lines)

    def show_ytrue(self, image, y_true):
        y_true = y_true[0]
        show_numpy_image(image, invert_axes=True)
        show_connected_components(connected_components(y_true))
        show_numpy_image(y_true, invert_axes=False)

    def process_labels(self, labels, is_cuda=True):
        var = torch.autograd.Variable(labels).float()
        if is_cuda:
            var = var.cuda()
        return var

    def ytrue_to_lines(self, image, predicted, with_images=True):
        return self.decoder.decode(image, predicted, with_images, degree=3, brut_points=True)

    def collate(self, batch):
        data = [item[0] for item in batch]  # just form a list of tensor
        label = [item[1] for item in batch]

        min_width = min([d.size()[1] for d in data])
        min_height = min([d.size()[0] for d in data])

        min_width = min(min_width, 300)
        min_height = min(min_height, 300)

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
