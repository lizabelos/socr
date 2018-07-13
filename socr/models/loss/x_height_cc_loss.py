import numpy as np
import torch

from socr.models.decoders.baseline_decoder import BaselineDecoder
from socr.models.encoders.baseline_encoder import BaselineEncoder
from socr.utils.image import show_numpy_image
from socr.utils.image.connected_components import show_connected_components, connected_components
from . import Loss


class XHeightCCLoss(Loss):
    """An absolute position Loss"""

    def __init__(self):
        """

        :param s: grid division, assuming we have only 1 bounding box per cell
        """
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.height_factor = 1.0 / 32.0
        self.decoder = BaselineDecoder(self.height_factor)
        self.encoder = BaselineEncoder(self.height_factor)

    def forward(self, predicted, y_true):
        return self.mse(predicted, y_true)
        #
        # batch_size = predicted.size()[0]
        # width = predicted.size()[3]
        # height = predicted.size()[2]
        #
        # predicted = predicted.permute(1, 0, 2, 3).contiguous()
        # y_true = y_true.permute(3, 0, 1, 2).contiguous()
        #
        # probs_error = (predicted[0] - y_true[0])
        # probs_error = probs_error * probs_error
        #
        # height_error = (predicted[1] - y_true[1]) * y_true[0]
        # height_error = height_error * height_error
        #
        # return torch.sum(probs_error.view(-1) + height_error.view(-1)) / (batch_size * width * height)

    def document_to_ytrue(self, image_size, base_lines):
        return self.encoder.encode(image_size, base_lines)

    def show_ytrue(self, image, y_true):
        # y_true = np.swapaxes(y_true, 0, 2)
        # y_true = np.swapaxes(y_true, 1, 2)
        y_true = y_true[0]

        print("Show image...")
        show_numpy_image(image, invert_axes=True)

        print("Show cc...")
        show_connected_components(connected_components(y_true))

        print("Show prediction...")
        show_numpy_image(y_true, invert_axes=False)

    def process_labels(self, labels, is_cuda=True):
        var = torch.autograd.Variable(labels).float()
        if is_cuda:
            var = var.cuda()
        return var

    def ytrue_to_lines(self, image, predicted, with_images=True):
        return self.decoder.decode(image, predicted, with_images, degree=3)

