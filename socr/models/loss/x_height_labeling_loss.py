import numpy as np
import torch

from socr.utils.image import show_numpy_image
from . import Loss


class XHeightLabelingLoss(Loss):
    """An absolute position Loss"""

    def __init__(self):
        """

        :param s: grid division, assuming we have only 1 bounding box per cell
        """
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.height_factor = 1.0 / 32.0

    def forward(self, predicted, y_true):
        batch_size = predicted.size()[0]
        width = predicted.size()[3]
        height = predicted.size()[2]

        predicted = predicted.permute(1, 0, 2, 3).contiguous()
        y_true = y_true.permute(3, 0, 1, 2).contiguous()

        probs_error = (predicted[0] - y_true[0])
        probs_error = probs_error * probs_error

        height_error = (predicted[1] - y_true[1]) * y_true[0]
        height_error = height_error * height_error

        return torch.sum(probs_error.view(-1) + height_error.view(-1)) / (batch_size * width * height)

        # return self.mse(predicted, y_true)

    def plot(self, y_true, x, y, height):
        # TODO : Multiply with image_size ??
        if x >= y_true.shape[1] or y >= y_true.shape[0] or x < 0 or y < 0:
            return

        # y_true[y][x] = height
        y_true[y][x][0] = 1.0
        y_true[y][x][1] = height * self.height_factor

    def draw_vertical_line(self, y_true, line):
        x0, y0, x1, y1, height = line

        for y in range(y0, y1):
            self.plot(y_true, x0, y, height)

    def draw_line(self, y_true, line):
        x0, y0, x1, y1, height = line

        deltax = x1 - x0
        deltay = y1 - y0
        if deltax == 0:
            return self.draw_vertical_line(y_true, line)
        deltaerr = abs(deltay / deltax)
        error = 0.0
        y = y0

        for x in range(x0, x1):
            self.plot(y_true, x, y, height)
            error = error + deltaerr
            while error >= 0.5:
                sign = -1 if deltay < 0 else 1
                y = y + sign
                error = error - 1.0

    def document_to_ytrue(self, image_size, base_lines):
        image_width, image_height = image_size
        y_true = np.zeros((image_height, image_width, 2), dtype='float')

        for line in base_lines:
            self.draw_line(y_true, line)

        return y_true

    def show_ytrue(self, image, y_true):
        y_true = y_true[0]
        image = np.stack((y_true, image[0], image[1]), axis=-1)
        show_numpy_image(image, invert_axes=False)
        show_numpy_image(y_true, invert_axes=False)

    def ytrue_to_lines_link(self, y_true, links, x_begin, y_begin):
        pos_list = [(x_begin, y_begin)]
        result = []

        while len(pos_list) > 0:
            x, y = pos_list.pop()

            if x < 0 or y < 0 or x >= y_true.shape[2] or y >= y_true.shape[1]:
                continue

            h = y_true[1][y][x] / self.height_factor
            p = y_true[0][y][x]

            if p < 0.25 or links[y][x][0] == -1:
                continue

            links[y][x][0] = -1
            links[y][x][1] = -1

            result.append((x, y, h))

            for x_off in range(-3, 3):
                for y_off in range(-3, 3):
                    pos_list.append((x + x_off, y + y_off))

        return result

    def ytrue_to_lines_extract_image(self, image, link_list):
        # todo : derivate ?
        x_min = int(min([link[0] for link in link_list]))
        x_max = int(max([link[0] for link in link_list]))
        y_min = int(min([link[1] - link[2] / 2 for link in link_list]))
        y_max = int(max([link[1] + link[2] / 2 for link in link_list]))

        width = x_max - x_min + 1
        height = y_max - y_min + 3

        if width < 10 or height < 10:
            return None

        line = np.ones((3, height, width))

        for x in range(0, width):
            for y in range(0, height):
                line[0][y][x] = image[0][y + y_min][x + x_min]
                line[1][y][x] = image[1][y + y_min][x + x_min]
                line[2][y][x] = image[2][y + y_min][x + x_min]

        return line, (x_min, y_min, x_max, y_max)

    def ytrue_to_lines(self, image, y_true):
        height, width = y_true.shape[1], y_true.shape[2]

        links = np.zeros((height, width, 4), dtype='int')
        link_list_array = []

        for x in range(0, width):
            for y in range(0, height):
                link_list = self.ytrue_to_lines_link(y_true, links, x, y)
                if len(link_list) > 0:
                    link_list_array.append(link_list)

        lines = []
        for link_list in link_list_array:
            line = self.ytrue_to_lines_extract_image(image, link_list)
            if line is not None:
                lines.append(line)

        return lines

    def process_labels(self, labels, is_cuda=True):
        var = torch.autograd.Variable(labels).float()
        if is_cuda:
            var = var.cuda()
        return var
