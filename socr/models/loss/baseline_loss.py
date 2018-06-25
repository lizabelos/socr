import math

import numpy as np
import torch

from socr.utils.image import show_numpy_image, foreach_point_in_line
from . import Loss


class BaselineLoss(Loss):
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
        show_numpy_image(y_true, invert_axes=False)

    def process_labels(self, labels, is_cuda=True):
        var = torch.autograd.Variable(labels).float()
        if is_cuda:
            var = var.cuda()
        return var

    def hough_draw_lines(self, x, y, proba, hough_tab):
        for a in range(0, 180):
            p = int(x * math.cos(math.radians(a)) + y * math.sin(math.radians(a)))
            hough_tab[p][a] += proba

    def hough_draw(self, predicted):
        height, width = predicted.shape[1], predicted.shape[2]

        hough_max = int(math.sqrt(height ** 2 + width ** 2))
        hough_tab = np.zeros((hough_max, 180), dtype='float')

        for x in range(0, width):
            for y in range(0, height):
                self.hough_draw_lines(x, y, predicted[0][y][x], hough_tab)

        return hough_tab, hough_max

    def hough_get_min_and_max(self, hough_tab, hough_max):
        min_value = hough_tab[0][0]
        max_value = hough_tab[0][0]

        for x in range(0, hough_max):
            for y in range(0, 180):
                min_value = min(min_value, hough_tab[x][y])
                max_value = max(max_value, hough_tab[x][y])

        return min_value, max_value

    def hough_is_local_max(self, hough_tab, x, y, radius=5):
        current_value = hough_tab[x][y]
        for rx in range(-radius, radius):
            for ry in range(-radius, radius):
                if rx == 0 and ry == 0:
                    continue
                compx = x + rx
                compy = y + ry
                if compx < 0 or compx >= hough_tab.shape[0] or compy < 0 or compy >= hough_tab.shape[1]:
                    continue
                if hough_tab[compx][compy] > current_value:
                    return False

        return True

    def hough_find_local_max(self, hough_tab, hough_max, radius=5, seuil=50):
        lines = []
        for p in range(0, hough_max):
            for a in range(0, 180):
                if hough_tab[p][a] < seuil:
                    continue
                if self.hough_is_local_max(hough_tab, p, a, radius):
                    c = math.cos(math.radians(a))
                    s = math.sin(math.radians(a))
                    x = 0 if c == 0 else int(p / c)
                    y = 0 if s == 0 else int(p / s)
                    lines.append((0, y, x, 0))
        return lines

    def hough_extract_lines(self, hough_tab, hough_max, threshold):
        lines = []

        for p in range(0, hough_max):
            for a in range(0, 180):
                if hough_tab[p][a] > threshold:
                    x = int(p / math.cos(math.radians(a)))
                    y = int(p / math.sin(math.radians(a)))
                    lines.append((0, y, x, 0))

        return lines

    def hough_lines_to_segment(self, lines, predicted):
        segment = []

        for x0, y0, x1, y1 in lines:
            begin = None
            end = None

            if x1 == x0:
                print("Warning : vertical line")
                continue
            # TODO : Bug x1 == x0 ?
            coef = (y1 - y0) / (x1 - x0)
            at0 = y0 - x0 * coef

            for x in range(0, predicted.shape[2]):
                y = x * coef + at0
                xi, yi = int(x), int(y)
                if xi >= 0 and yi >= 0 and xi < predicted.shape[2] and yi < predicted.shape[1]:
                    if predicted[0][yi][xi] > 0.2:
                        if begin is None:
                            begin = (xi, yi)
                        else:
                            end = (xi, yi)
                    elif begin is not None and end is not None:
                        segment.append((begin[0], begin[1], end[0], end[1]))
                        begin = None
                        end = None

            if begin is not None and end is not None:
                segment.append((begin[0], begin[1], end[0], end[1]))

        return segment

    def rotate(self, origin, point, angle):
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

    def lines_to_tangents(self, lines):
        results = []

        for x0, y0, x1, y1 in lines:
            xC, yC = (x0 + x1) / 2, (y0 + y1) / 2

            x0, y0 = x0 - xC, y0 - yC
            x1, y1 = x1 - xC, y1 - yC

            xt0, yt0 = self.rotate((0, 0), (x0, y0), math.radians(90))
            xt1, yt1 = self.rotate((0, 0), (x1, y1), math.radians(90))

            d0 = math.sqrt(xt0 ** 2 + yt0 ** 2)
            d1 = math.sqrt(xt1 ** 2 + yt1 ** 2)

            xt0 = xt0 / d0
            yt0 = yt0 / d0
            xt1 = xt1 / d1
            yt1 = yt1 / d1

            results.append((xt0, yt0, xt1, yt1))

        return results

    def extract_lines(self, image, predicted, lines, tangents):

        results = []

        for i in range(0, len(lines)):
            x0, y0, x1, y1 = lines[i]

            xt0, yt0, xt1, yt1 = tangents[i]
            xtC, ytC = (xt0 + xt1) / 2, (yt0 + yt1) / 2
            xt0, yt0, xt1, yt1 = xt0 - xtC, yt0 - ytC, xt1 - xtC, yt1 - ytC

            width = int(math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2) + 0.5)
            height = int(foreach_point_in_line(lines[i], lambda x, y, v: predicted[1][y][x] if v is None else max(v, predicted[1][y][x])) / self.height_factor + 0.5)

            print(str(width) + " " + str(height))

            if width < 10 or height < 10:
                continue

            line_image = np.ones((3, height, width), dtype='float')

            for line_x in range(0, width):
                predicted_x = x0 * (line_x / width) + x1 * ((width - line_x) / width)
                predicted_y = y0 * (line_x / width) + y1 * ((width - line_x) / width)
                h = height
                # h = int(predicted[1][int(predicted_y)][int(predicted_x)] / self.height_factor)
                cxt0, cyt0, cxt1, cyt1 = xt0 * h, yt0 * h, xt1 * h, yt1 * h
                for line_y in range(0, height):
                    # TODO : interpolation
                    image_x = int(predicted_x + (cxt0 * line_y / h) + (cxt1 * (h - line_y) / h))
                    image_y = int(predicted_y + (cyt0 * line_y / h) + (cyt1 * (h - line_y) / h))
                    if image_x > 0 and image_y > 0 and image_x < image.shape[2] and image_y < image.shape[1]:
                        line_image[0][height - line_y - 1][line_x] = image[0][image_y][image_x]
                        line_image[1][height - line_y - 1][line_x] = image[1][image_y][image_x]
                        line_image[2][height - line_y - 1][line_x] = image[2][image_y][image_x]

            results.append((line_image, lines[i]))

        return results

    def ytrue_to_lines(self, image, predicted):
        show_numpy_image(predicted[0], invert_axes=False)

        hough_tab, hough_max = self.hough_draw(predicted)
        show_numpy_image(hough_tab, invert_axes=False)

        # min_value, max_value = self.hough_get_min_and_max(hough_tab, hough_max)
        # median_value = (min_value + max_value) / 2
        # lines = self.hough_extract_lines(hough_tab, hough_max, threshold=median_value)
        lines = self.hough_find_local_max(hough_tab, hough_max)
        print("Detected " + str(len(lines)) + " lines")
        print(lines)
        segments = self.hough_lines_to_segment(lines, predicted)
        print("Found " + str(len(segments)) + " segments")
        tangents = self.lines_to_tangents(segments)
        extracted = self.extract_lines(image, predicted, segments, tangents)
        print("Extracted " + str(len(extracted)) + " images")
        return extracted
