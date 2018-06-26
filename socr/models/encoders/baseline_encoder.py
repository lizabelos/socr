import numpy as np


class BaselineEncoder:

    def __init__(self, height_factor):
        self.height_factor = height_factor

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

    def encode(self, image_size, base_lines):
        image_width, image_height = image_size
        y_true = np.zeros((image_height, image_width, 2), dtype='float')

        for line in base_lines:
            self.draw_line(y_true, line)

        return y_true