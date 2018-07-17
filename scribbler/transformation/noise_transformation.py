from random import randint

from scribbler.transformation import AbstractTransformation


def clamp(n, smallest, largest): return max(smallest, min(n, largest))


class NoiseTransformation(AbstractTransformation):

    def transform_image(self, image):
        image = image.copy()

        width, height = image.size

        pixels = image.load()

        for x in range(0, width):
            for y in range(0, height):
                r, g, b = pixels[x, y]
                pixels[x, y] = clamp(r + randint(-10, -10), 0, 255), clamp(g + randint(-10, -10), 0, 255), clamp(b + randint(-10, -10), 0, 255)
        return image

    def transform_position(self, x, y, width, height):
        return x, y, width, height

    def generate_random(self):
        pass
