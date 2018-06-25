import numpy as np
import torch

from . import Loss

class WhereToStartLoss(Loss):
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

    def forward(self, predicted, y_true):
        # B H W C
        # C B H W
        batch_size = predicted.size()[0]

        predicted = predicted.permute(3, 0, 1, 2).contiguous()
        y_true = y_true.permute(1, 0, 2, 3).contiguous()

        probs_error = (predicted[0] - y_true[0])
        probs_error = probs_error * probs_error

        height_error = (predicted[1] - y_true[1]) * y_true[0]
        height_error = height_error * height_error

        return torch.sum(probs_error.view(-1) + height_error.view(-1)) / batch_size

    def box_at(self, pos):
        x, y = pos
        return int(x * (self.sw - 1)), int(y * (self.sh - 1))

    def document_to_ytrue(self, coords):
        y_true = np.zeros((2, self.sh, self.sw), dtype='float')

        for coord in coords:
            polygon = [self.box_at(coord[i]) for i in range(0, 4)]

            x_start = min([p[0] for p in polygon])
            x_stop = max([p[0] for p in polygon])

            y_start = min([p[1] for p in polygon])
            y_end = max([p[1] for p in polygon])

            # print(str(y_end) + " " + str(x_start))

            y_true[0][y_end][x_start] = 1
            y_true[1][y_end][x_start] = (y_end - y_start) / self.sw

        return y_true

    def ytrue_to_lines(self, ytrue):
        ytrue = ytrue.cpu().detach().numpy()
        predicted = []

        for x in range(0, self.sw):
            for y in range(0, self.sh):
                if ytrue[y][x][0] > 0.40:
                    predicted.append((ytrue[y][x][0], x / self.sw, y / self.sh, ytrue[y][x][1]))

        return predicted

    def process_labels(self, labels, is_cuda=True):
        var = torch.autograd.Variable(labels).float()
        if is_cuda:
            var = var.cuda()
        return var
