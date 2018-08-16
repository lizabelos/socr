import math

import numpy as np
import glymur
import matplotlib
from PIL import Image

matplotlib.use('tkagg')
from matplotlib import pyplot
from skimage import filters
import cv2

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

def image_numpy_to_pillow_bw(image):
    image = image * 255.0
    return Image.fromarray(image.astype('uint8'), 'L')


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


def connected_components(image, hist_min, hist_max):
    image = np.array(image)

    thresh = filters.apply_hysteresis_threshold(np.array(image), hist_min, hist_max)
    thresh = (np.clip(thresh, 0, 1) * 255).astype(np.uint8)

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




