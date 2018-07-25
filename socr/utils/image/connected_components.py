import math

import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import convolve, gaussian_filter
from skimage import filters

from socr.utils.image import show_numpy_image


def interpolation(array, x, y):
    s = array.shape
    i = math.floor(x)
    j = math.floor(y)
    t = x - i
    u = y - j
    u1 = 1.0 - u
    t1 = 1.0 - t
    if j == s[0] - 1:
        if i == s[1] - 1:
            return array[j][i]
        return t * array[j][i] + t1 * array[j + 1][i]
    if i == s[1] - 1:
        return u * array[j][i] + u1 * array[j][i + 1]
    return t1 * u1 * array[j][i] + t * u1 * array[j][i + 1] + \
           t * u * array[j + 1][i + 1] + t1 * u * array[j + 1][i]


def connected_components(image, hist_min=0.5, hist_max=0.97):
    image = np.array(image)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))


    # show_numpy_image(image, invert_axes=False)
    # image = cv2.erode(image, kernel, iterations=1)
    # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # show_numpy_image(image, invert_axes=False)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # image = cv2.erode(np.array(image), kernel, iterations=1)
    #
    # sobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # sobelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # derivX = convolve(image, sobelX)
    # derivY = convolve(image, sobelY)
    #
    # gradient = derivX + derivY * 1j
    #
    # # gradient = np.gradient(image)
    #
    # # show_numpy_image(gradient[0], invert_axes=False)
    # # show_numpy_image(gradient[1], invert_axes=False)
    #
    # # gradient = gradient[1] + gradient[0] * 1j
    #
    # # print(gradient)
    #
    # # print(gradient.shape)
    #
    # # show_numpy_image(gradient, invert_axes=False)
    #
    # G = np.absolute(gradient)
    # theta = np.angle(gradient)
    #
    # Gmax = G.copy()
    #
    # for i in range(1, image.shape[1] - 1):
    #     for j in range(1, image.shape[0] - 1):
    #         if G[j][i] != 0:
    #             cos = math.cos(theta[j][i])
    #             sin = math.sin(theta[j][i])
    #             g1 = interpolation(G, i + cos, j + sin)
    #             g2 = interpolation(G, i - cos, j - sin)
    #             if (G[j][i] < g1) or (G[j][i] < g2):
    #                 Gmax[j][i] = 0.0

    # thresh = filters.apply_hysteresis_threshold(np.array(image), 0.4, 0.99)

    thresh = filters.apply_hysteresis_threshold(np.array(image), 0.5, 0.5)

    thresh = (np.clip(thresh, 0, 1) * 255).astype(np.uint8)

    # kernel = np.ones((10, 10), np.uint8)
    # image = cv2.dilate(image, kernel, iterations=1)

    # ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # kernel = np.ones((4, 4), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # thresh = cv2.erode(thresh, kernel, iterations=1)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Marker labelling
    ret, markers = cv2.connectedComponents(thresh)
    markers = markers + 1
    markers[thresh == 0] = 0
    return markers


def show_connected_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    cv2.imshow('labels', labeled_img)
    cv2.waitKey(100)


def save_connected_components(labels, path):
    # Map component labels to hue val
    labels = np.array(labels)
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    cv2.imwrite(path, labeled_img)
