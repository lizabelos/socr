import numpy as np
import torch

from socr.utils.image import iou
from . import Loss


class AbsoluteBoxLoss(Loss):
    """An absolute position Loss"""

    def __init__(self, sw=8, sh=128, lambda_prob=1.0, lambda_position=1.0, lambda_size=1.0):
        """

        :param s: grid division, assuming we have only 1 bounding box per cell
        """
        super().__init__()
        self.sw = sw
        self.sh = sh
        self.s = self.sw * self.sh
        self.lambda_prob = lambda_prob
        self.lambda_position = lambda_position
        self.lambda_size = lambda_size
        self.num_classes = 1

        # For each box, we predict one probability, one 2D coordinate for left-top corner, and one 2D size
        self.prediction_size = self.s * (self.num_classes + 2 + 2)

    def get_prediction_size(self):
        return self.prediction_size

    def narrow_data(self, data):
        if data.size()[1] != self.prediction_size:
            raise ValueError("invalid data size : " + str(data.size()) + "!=" + str([None, self.prediction_size]))

        probabilities = data.narrow(1, 0, self.s * self.num_classes)
        x = data.narrow(1, self.s * (self.num_classes + 0), self.s)
        y = data.narrow(1, self.s * (self.num_classes + 1), self.s)
        widths = data.narrow(1, self.s * (self.num_classes + 2), self.s)
        heights = data.narrow(1, self.s * (self.num_classes + 3), self.s)

        return probabilities, x, y, widths, heights

    def forward(self, predicted, y_true):
        if predicted.size()[1] != self.prediction_size:
            raise ValueError("invalid prediction size : " + str(predicted.size()) + "!=" + str([None, self.prediction_size]))
        if y_true.size()[1] != self.prediction_size:
            raise ValueError("invalid truth size : " + str(y_true.size()) + "!=" + str([None, self.prediction_size]))

        # Slice y_true values
        truth_probabilities, truth_x, truth_y, truth_widths, truth_heights = self.narrow_data(y_true)
        num_objects = torch.sum(truth_probabilities, dim=1)

        # Slice predicted values
        pred_probabilities, pred_x, pred_y, pred_widths, pred_heights = self.narrow_data(predicted)

        # Probabilities error
        probabilities_error = truth_probabilities - pred_probabilities
        probabilities_error = torch.mul(probabilities_error, probabilities_error)
        probabilities_error = probabilities_error * self.lambda_prob

        # Position error (only for true box)
        x_error = truth_x - pred_x
        y_error = truth_y - pred_y
        position_error = (torch.mul(x_error, x_error) + torch.mul(y_error, y_error)) * truth_probabilities
        position_error = position_error * self.lambda_position

        # Size error (only for true box)
        width_error = truth_widths - pred_widths
        height_error = truth_heights - pred_heights
        size_error = (torch.mul(width_error, width_error) + torch.mul(height_error,
                                                                      height_error)) * truth_probabilities
        size_error = size_error * self.lambda_size

        error_vector = (torch.sum(probabilities_error, dim=1) + torch.sum(position_error, dim=1) + torch.sum(size_error, dim=1)) / num_objects
        return torch.sum(error_vector, dim=0) / error_vector.size()[0]

    def box_at(self, x, y):
        return int(x * self.sw), int(y * self.sh)

    def document_to_ytrue(self, document):
        line_list = document.line_list()
        y_true = np.zeros(self.prediction_size, dtype='float')
        for line in line_list:
            line_x, line_y = line.get_position()
            line_width, line_height = line.get_size()

            box_x, box_y = self.box_at(line_x, line_y)
            box_id = box_x * self.sh + box_y

            if y_true[box_id] == 1:
                print("Warning : Box collision box_x=" + str(box_x) + ", box_y=" + str(box_y))

            y_true[box_id] = 1
            y_true[box_id + (self.s * 1)] = line_x
            y_true[box_id + (self.s * 2)] = line_y
            y_true[box_id + (self.s * 3)] = line_width
            y_true[box_id + (self.s * 4)] = line_height

        return y_true

    def ytrue_to_lines(self, ytrue):
        line_list = []

        probs = []
        boxes = []
        for i in range(0, self.s):
            if ytrue[i] > 0.5:
                p = ytrue[i]
                x = ytrue[i + (self.s * 1)]
                y = ytrue[i + (self.s * 2)]
                w = ytrue[i + (self.s * 3)] + x
                h = ytrue[i + (self.s * 4)] + y
                boxes.append(np.array([x, y, w, h]))
                probs.append(p)

        for i in range(0, len(boxes) - 1):
            no_intersection = True
            for j in range(i + 1, len(boxes)):
                intersection = iou(boxes[i], boxes[j])
                if intersection > 1.0:
                    print("IOU is " + str(intersection) + ", merging 2 boxes")
                    boxes[j] = (boxes[i] * probs[i] + boxes[j] * probs[j]) / (probs[i] + probs[j])
                    probs[j] = max(probs[i], probs[j])
                    no_intersection = False

            if no_intersection:
                p = probs[i]
                x = boxes[i][0]
                y = boxes[i][1]
                w = boxes[i][2] - x
                h = boxes[i][3] - y
                line_list.append((p, x, y, w, h))

        return line_list

    def process_labels(self, labels, is_cuda=True):
        var = torch.autograd.Variable(labels).float()
        if is_cuda:
            var = var.cuda()
        return var