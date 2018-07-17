from random import randint

from PIL import ImageOps

from scribbler.resources.resources_helper import peak_random_preloaded_image
from scribbler.transformation import AbstractTransformation


class PhantomTransformation(AbstractTransformation):

    def transform_image(self, image):
        image = image.copy()
        image_width, image_height = image.size
        n = randint(0, 10)
        for i in range(0, n):
            phantomPatterns = peak_random_preloaded_image("phantomPatterns")
            image.paste(phantomPatterns, (randint(0, image_width), randint(0, image_height)), ImageOps.invert(phantomPatterns.convert("L")))
        return image

    def transform_position(self, x, y, width, height):
        return x, y, width, height

    def generate_random(self):
        pass
