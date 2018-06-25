import math

import numpy as np
import glymur
import matplotlib
from PIL import Image

matplotlib.use('tkagg')
from matplotlib import pyplot


def image_pillow_to_numpy(image):
    image = np.array(image, dtype='float') / 255.0
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 1, 2)
    return image


def image_numpy_to_pillow(image):
    image = np.swapaxes(image, 1, 2)
    image = np.swapaxes(image, 0, 2)
    image = image * 255.0
    return Image.fromarray(image.astype('uint8'), 'RGB')


def image_pytorch_to_pillow(image):
    return image_numpy_to_pillow(image.cpu().detach().numpy())


def load_jp2_numpy(image_path):
    image = glymur.Jp2k(image_path)
    image = np.array(image[:], dtype='float') / 255.0
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 1, 2)
    return image


def show_numpy_image(image, invert_axes=True, pause=3):
    if invert_axes:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 0, 2)
    print("Showing image using tkagg..." + str(image.shape))
    pyplot.imshow(image)
    pyplot.pause(pause)


def show_pytorch_image(image, invert_axes=True, pause=3):
    return show_numpy_image(image.cpu().detach().numpy(), invert_axes, pause)


def is_between_1d(a, b, v):
    return max(a, b) >= v >= min(a, b)


def is_between_2d(p1, p2, p):
    return is_between_1d(p1[0], p2[0], p[0]) and is_between_1d(p1[1], p2[1], p[1])


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def segment_intersection(seg1, seg2):
    intersection = line_intersection(seg1, seg2)
    if intersection is None:
        return None

    if is_between_2d(seg1[0], seg1[1], intersection) and is_between_2d(seg2[0], seg2[1], intersection):
        return intersection

    return None


def linesegment_intersection(seg1, line2):
    intersection = line_intersection(seg1, line2)
    if intersection is None:
        return None

    if is_between_2d(seg1[0], seg1[1], intersection):
        return intersection

    return None


def foreach_point_in_polygon(polygon, function):
    """Return the numbers of points"""
    min_y = min([y for x, y in polygon])
    max_y = max([y for x, y in polygon])
    results = []
    for i in range(min_y, max_y):
        intersections = []
        for j in range(0, len(polygon)):
            intersection = linesegment_intersection((polygon[j], polygon[(j + 1) % len(polygon)]), (0, i, 1, i))
            if intersection is not None:
                intersections.append(intersection)

        intersections = sorted(intersections, key=lambda x: x[0])
        for intersection_i in range(0, len(intersections) // 2):
            min_x = int(intersections[intersection_i * 2])
            max_x = int(intersections[intersection_i * 2 + 1])
            for x in range(min_x, max_x):
                results.append(function(x, i))

    return results


def foreach_point_in_vertical_line(line, fn, value=None):
    x0, y0, x1, y1 = line

    for y in range(y0, y1):
        value = fn(x0, y, value)

    return value


def foreach_point_in_line(line, fn, value=None):
    x0, y0, x1, y1 = line

    deltax = x1 - x0
    deltay = y1 - y0
    if deltax == 0:
        return foreach_point_in_vertical_line(line, fn)
    deltaerr = abs(deltay / deltax)
    error = 0.0
    y = y0

    for x in range(x0, x1):
        value = fn(x, y, value)
        error = error + deltaerr
        while error >= 0.5:
            sign = -1 if deltay < 0 else 1
            y = y + sign
            error = error - 1.0

    return value


def mIoU(polygons, segmentated, seuil=0.5):
    polygons_area = 0
    intersections_and_area = 0
    segmentated_area = 0

    for line in segmentated:
        for c in line:
            if c > seuil:
                segmentated_area = segmentated_area + 1

    for polygon in polygons:
        results = foreach_point_in_polygon(polygon, lambda x, y: segmentated[y][x])

        polygons_area = polygons_area + len(segmentated)
        intersections_and_area = intersections_and_area + sum([0 if n < seuil else 1 for n in results])

    intersections_or_area = polygons_area + segmentated_area - intersections_and_area
    return intersections_and_area / intersections_or_area


def iou(bboxes1, bboxes2):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou
