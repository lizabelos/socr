import random
from random import randint

from PIL import ImageDraw

from socr.dataset.generator.document_generator_helper import DocumentGeneratorHelper
from socr.dataset.generator.generator import Generator


class CharacterGenerator(Generator):

    def __init__(self, helper, labels, width=32, height=32):
        super().__init__()
        self.helper = helper
        self.lst = labels
        self.width = width
        self.height = height

    def generate(self, index):

        font_height = self.height
        font_color = randint(0, 255), randint(0, 255), randint(0, 255)

        text = self.generate_text()
        font = self.helper.get_font(index, font_height)
        # font_width = font.getsize(text)[0]

        image = self.helper.get_random_background(self.width, self.height)
        image_draw = ImageDraw.Draw(image)
        image_draw.text((0, 0), text, font=font, fill=font_color)

        return image, text

    def get_labels(self):
        return self.lst

    def generate_text(self):
        return random.choice(self.lst)
