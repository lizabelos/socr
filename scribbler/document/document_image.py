from PIL import Image

from scribbler.document import AbstractDocument
from scribbler.resources.resources_helper import peak_random_preloaded_image


class DocumentImage(AbstractDocument):

    def __init__(self, width=None, height=None, parent=None):
        super().__init__(width, height, parent)
        self.image = peak_random_preloaded_image("backgrounds")

    def to_image(self):
        return self.image.copy().convert("RGBA")

    def generate_random(self, index=-1):
        self.image = self.get_random_image()

    def get_random_image(self):
        image = peak_random_preloaded_image("backgrounds")
        return image.resize(self.get_size(), Image.ANTIALIAS)
