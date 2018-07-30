import cv2
import numpy as np
from skimage import filters


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
