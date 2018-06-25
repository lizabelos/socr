import numpy as np
import torch

from . import Loss


class LineSemanticSegmentationLoss(Loss):
    """An absolute position Loss"""

    def __init__(self, sw, sh, lambda_prob=1.0, lambda_position=1.0):
        """

        :param s: grid division, assuming we have only 1 bounding box per cell
        """
        super().__init__()
        self.sw = sw
        self.sh = sh
        self.s = self.sw * self.sh
        self.lambda_prob = lambda_prob
        self.lambda_position = lambda_position
        self.mse = torch.nn.MSELoss()

    def narrow_coord(self, data, id):
        return data.narrow(1, self.s * id, self.s)

    def narrow_coords(self, data, id):
        id1 = id * 2
        id2 = id1 + 1
        return data.narrow(1, self.s * id1, self.s), \
               data.narrow(1, self.s * id2, self.s)

    def narrow_data(self, data):
        if data.size()[1] != self.prediction_size:
            raise ValueError("invalid data size : " + str(data.size()) + "!=" + str([None, self.prediction_size]))

        probabilities = data.narrow(1, 0, self.s)
        coords = []

        for i in range(0, 4):
            coords.append(self.narrow_coords(data, i))

        return probabilities, coords

    def forward(self, predicted, y_true):
        return self.mse(predicted, y_true)

    def box_at(self, pos):
        x, y = pos
        return int(x * self.sw), int(y * self.sh)

    def document_to_ytrue_plot_point(self, y_true, p, higher_y, left=False):
        x, y = p
        y_true[y][x][0] = 1
        y_true[y][x][2] = 0 if left else 1
        y_true[y][x][1] = 0

        if higher_y[x] is None or y < higher_y[x]:
            y_true[y][x][1] = 1
            if higher_y[x] is not None:
                y_true[higher_y[x]][x][1] = 0
            higher_y[x] = y

    def document_to_ytrue_draw_line(self, y_true, p1, p2, higher_y):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1

        x = x1
        while x < x2:
            y = int(y1 + dy * (x - x1) / dx)
            self.document_to_ytrue_plot_point(y_true, (x, y), higher_y, x == x1)
            x = x + 1

    def document_to_ytrue(self, coords):
        y_true = np.zeros((self.sh, self.sw, 3), dtype='float')

        for coord in coords:
            polygon = [self.box_at(coord[i]) for i in range(0, 4)]
            y_start = min([p[1] for p in polygon])
            y_end = max([p[1] for p in polygon])

            higher_y = [None] * self.sw

            y = y_start
            while y < y_end:

                points = []

                for i in range(0, len(polygon)):
                    p1 = polygon[i]
                    p2 = polygon[(i + 1) % 4]

                    min_x, max_x = min(p1[0], p2[0]), max(p1[0], p2[0])
                    min_y, max_y = min(p1[1], p2[1]), max(p1[1], p2[1])

                    if min_y <= y <= max_y:
                        dx = max_x - min_x
                        dy = max_y - min_y
                        if dy == 0:
                            points.append(min_x)
                            points.append(max_x)
                        else:
                            points.append(int(min_x + dx * (y - min_y) / dy))

                points.sort()
                for i in range(0, len(points) // 2):
                    p1_x = points[i * 2]
                    p2_x = points[i * 2 + 1]
                    self.document_to_ytrue_draw_line(y_true, (p1_x, y), (p2_x, y), higher_y)

                y = y + 1

        return y_true

    def binarize_result(self, ytrue):
        for i in range(0, ytrue.size(0)):
            for j in range(0, ytrue.size(1)):
                for k in range(0, 3):
                    if ytrue[i][j][k] > 0.5:
                        ytrue[i][j][k] = 1
                    else:
                        ytrue[i][j][k] = 0

        return ytrue

    def ytrue_to_lines(self, ytrue):
        # TODO : think about a better decoder
        boxes = np.zeros((self.sh, self.sw), dtype='int')
        num_boxes = 0

        for i in range(1, ytrue.size(0)):
            for j in range(1, ytrue.size(1)):
                for k in range(0, 3):
                    if ytrue[i][j][0] > 0.5:

                        if ytrue[i][j][2] < 0.5 and boxes[i][j - 1] != 0: # LEFT
                            boxes[i][j] = boxes[j][j - 1]
                        elif ytrue[i][j][1] < 0.5 and boxes[i - 1][j] != 0: # RIGHT
                            boxes[i][j] = boxes[i - 1][j]
                        elif ytrue[i][j][2] > 0.5 and ytrue[i][j][1] > 0.5:
                            boxes[i][j] = num_boxes + 1
                            num_boxes = num_boxes + 1
                        else:
                            if ytrue[i][j][2] < ytrue[i][j][1]:
                                boxes[i][j] = boxes[j][j - 1]
                            else:
                                boxes[i][j] = boxes[i - 1][j]
                            print("Warning : invalid prediction")

        return boxes, num_boxes

    def process_labels(self, labels, is_cuda=True):
        var = torch.autograd.Variable(labels).float()
        if is_cuda:
            var = var.cuda()
        return var
