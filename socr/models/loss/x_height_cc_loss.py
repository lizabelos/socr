import math
import threading
from multiprocessing.pool import ThreadPool

import numpy as np
import torch
from skimage.transform import resize

from socr.utils.image import show_numpy_image, foreach_point_in_line
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
        self.pool = ThreadPool(processes=4)

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
        if x >= y_true.shape[1] or y >= y_true.shape[0] or x < 0 or y < 0:
            return

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
        show_connected_components(connected_components(y_true))
        show_numpy_image(y_true, invert_axes=False)

    def process_labels(self, labels, is_cuda=True):
        var = torch.autograd.Variable(labels).float()
        if is_cuda:
            var = var.cuda()
        return var

    def ytrue_to_lines(self, image, predicted):
        components = connected_components(predicted[0])
        num_components = np.max(components)
        results = []

        asyncs = []

        for i in range(1, num_components + 1):
            asyncs.append(self.pool.apply_async(self.process_components, (image, predicted, components, i)))

        for async in asyncs:
            result = async.get()
            if result is not None:
                results.append(result)


        return results

    def process_components(self, image, prediction, components, index, degree=3, line_height=64, baseline_resolution=16):
        x_train = []
        y_data = []

        max_height = 10

        for y in range(0, components.shape[0]):
            for x in range(0, components.shape[1]):
                if components[y][x] == index:
                    x_train.append(x)
                    y_data.append(y)
                    max_height = max(max_height, prediction[1][y][x] / self.height_factor)

        if len(x_train) == 0:
            return None

        min_x = min(x_train)
        max_x = max(x_train)
        line_components_width = max_x - min_x
        line_components_height = int(max_height)
        line_height = line_height
        line_width = line_components_width * line_height // line_components_height

        if line_components_width < 10 and line_components_height < 10:
            # print("Warning : too small image width and height for index : " + str(index) + ". Skipping.")
            return None

        if line_components_width < 10:
            # print("Warning : too small image width for index : " + str(index) + ". Skipping.")
            return None

        if line_components_height < 10:
            # print("Warning : too small image height for index : " + str(index) + ". Skipping.")
            return None

        x_train = np.array(x_train)
        y_data = np.array(y_data)

        coefs = np.polynomial.polynomial.polyfit(x_train, y_data, degree)
        # coefs = np.polyfit(x_train, y_data, degree)
        ffit = np.polynomial.Polynomial(coefs)
        fderiv = np.polynomial.Polynomial(np.polynomial.polynomial.polyder(coefs))

        fnorm  = lambda x, a: -(a - x) / fderiv(x) + ffit(x)

        line = np.ones((3, line_height, line_width))
        baseline_points = []

        # line_x = np.arange(0, line_width)
        # x = line_x * line_components_width / line_width + min_x
        # bs_x = x * image.shape[2] / prediction.shape[2]
        # bs_y = ffit(x) * image.shape[1] / prediction.shape[1]
        #
        # tanx0 = x + 1
        # tany0 = fnorm(x, x + 1)
        # tanx1 = x - 1
        # tany1 = fnorm(x, x - 1)
        #
        # dist0 = np.sqrt((tanx0 - x) ** 2 + (tany0 - ffit(x)) ** 2)
        # dist1 = np.sqrt((tanx1 - x) ** 2 + (tany1 - ffit(x)) ** 2)
        #
        # tany0 = (tany0 - ffit(x)) * line_components_height / (dist0 * 2) + ffit(x)
        # tany1 = (tany1 - ffit(x)) * line_components_height / (dist1 * 2) + ffit(x)
        #
        # c = np.arange(0, 3).astype(int)
        # line_x = line_x.astype(int)
        # line_y = np.arange(0, line_height).astype(int)
        # line_positions = np.array(np.meshgrid(c, line_y, line_x)).T.reshape(-1,3)
        #
        #
        # image_positions = np.swapaxes(line_positions, 0, 1)
        # image_positions[2] = x[image_positions[2]] * image.shape[2] / prediction.shape[2]
        # image_positions[1] = image_positions[1] * line_components_height / line_height
        # image_positions[1] = tany0[image_positions[2]] * (image_positions[1] / line_components_height) + tany1[image_positions[2]] * ((-image_positions[1] + line_components_height) / line_components_height)
        # image_positions = np.swapaxes(image_positions, 0, 1)
        #
        # for i in range(0, line_positions.shape[0]):
        #     print(str(line_positions[i]) + " " + str(image_positions[i]))
        #     line[line_positions[i][0]][line_positions[i][1]][line_positions[i][2]] = image[image_positions[i][0]][image_positions[i][1]][image_positions[i][2]]
        #

        for i in range(0, baseline_resolution + 1):
            x = min_x * (i / baseline_resolution) + max_x * ((baseline_resolution - i) / baseline_resolution)
            baseline_points.append(int(x * image.shape[2] / prediction.shape[2]))
            baseline_points.append(int(ffit(x) * image.shape[1] / prediction.shape[1]))

        # for line_x in range(0, line_width):
            # x = line_x * line_components_width / line_width + min_x

            # deriv = fderiv(x)
            # baseline_points.append(int(x * image.shape[2] / prediction.shape[2]))
            # baseline_points.append(int(ffit(x) * image.shape[1] / prediction.shape[1]))

            # if deriv == 0:
            #     tanx0 = x
            #     tany0 = ffit(x) + 1
            #     tanx1 = x
            #     tany1 = ffit(x) - 1
            # else:
            #     tanx0 = x + 1
            #     tany0 = fnorm(x, x + 1)
            #     tanx1 = x - 1
            #     tany1 = fnorm(x, x - 1)
            #     if tany0 < tany1:
            #         tanx2, tany2 = tanx0, tany0
            #         tanx0, tany0 = tanx1, tany1
            #         tanx1, tany1 = tanx2, tany2
            #
            # dist0 = math.sqrt((tanx0 - x) ** 2 + (tany0 - ffit(x)) ** 2)
            # dist1 = math.sqrt((tanx1 - x) ** 2 + (tany1 - ffit(x)) ** 2)
            #
            # tanx0 = (tanx0 - x) * line_components_height / (dist0 * 2) + x
            # tany0 = (tany0 - ffit(x)) * line_components_height / (dist0 * 2) + ffit(x)
            # tanx1 = (tanx1 - x) * line_components_height / (dist1 * 2) + x
            # tany1 = (tany1 - ffit(x)) * line_components_height / (dist1 * 2) + ffit(x)
            #
            # for line_y in range(0, line_height):
            #     y = line_y * line_components_height / line_height
            #
            #     image_x = x * image.shape[2] / prediction.shape[2]
            #     image_y = tany0 * (y / line_components_height) + tany1 * ((line_components_height - y) / line_components_height)
            #     image_y = image_y * image.shape[1] / prediction.shape[1]
            #
            #     if image_x < 0 or image_y < 0 or image_x >= image.shape[2] or image_y >= image.shape[1]:
            #         continue
            #     # image_y = y * (tany0 - tany1) / line_height + tany1
            #
            #     for c in range(0, 3):
            #         line[c][int(line_y)][int(line_x)] = image[c][int(image_y)][int(image_x)]


        # show_numpy_image(line, pause=0.1)
        # line = resize(line, (3, line_height // 2, line_width // 2))

        return line, baseline_points