from random import randint

from PIL import ImageDraw
from skimage.draw import line_aa

from scribbler.generator import LineGenerator, DocumentGenerator
from scribbler.utils.image.image import show_numpy_image, image_numpy_to_pillow

if __name__ == '__main__':
    line_generator = DocumentGenerator()


    for i in range(0,4):
        image, baselines = line_generator.get(randint(0, line_generator.count()))
        image = image_numpy_to_pillow(image)

        image_drawer = ImageDraw.Draw(image)

        for bl in baselines:
            image_drawer.line((bl[0], bl[1], bl[2], bl[3]), fill=(256, 0, 0))

        print(baselines)
        image.show()