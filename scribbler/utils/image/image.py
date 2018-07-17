import numpy as np
from PIL import Image
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


def show_numpy_image(image, invert_axes=True, pause=3):
    if invert_axes:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 0, 2)
    print("Showing image using tkagg..." + str(image.shape))
    pyplot.imshow(image)
    pyplot.pause(pause)
