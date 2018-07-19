import numpy as np
import cv2


def connected_components(image):
    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    # kernel = np.ones((10, 10), np.uint8)
    # image = cv2.dilate(image, kernel, iterations=1)

    ret, thresh = cv2.threshold(image, 192, 255, cv2.THRESH_BINARY)

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
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    cv2.imwrite(path, labeled_img)