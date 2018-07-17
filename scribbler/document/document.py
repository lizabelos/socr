from math import cos, sin
from random import randint

from PIL import Image

from scribbler.document import AbstractDocument
from scribbler.resources.resources_helper import peak_random_preloaded_image
from scribbler.transformation.abstract_transformation import AbstractTransformation


class Document(AbstractDocument):

    def __init__(self, width=None, height=None, parent=None, random_rotation=10):
        super().__init__(width, height, parent)

        self.documents = []
        self.random_rotation = random_rotation
        self.transformations = []
        self.background = peak_random_preloaded_image("backgrounds")

    def append_document(self, document, position=(0, 0)):
        if not isinstance(document, AbstractDocument):
            raise TypeError("this is not a document")

        x, y = position
        assert x >= 0
        assert y >= 0

        self.documents.append((document, position))

    def add_transformation(self, transformation):
        if not isinstance(transformation, AbstractTransformation):
            raise TypeError("the text is not a string")
        self.transformations.append(transformation)

    def to_image(self):
        image = self.background.copy()

        for document, position in self.documents:
            document_image = document.to_image()
            image.paste(document_image, position, document_image)

        for transformation in self.transformations:
            transformation.transform_image(image)

        return image

    def generate_random(self, index=-1):
        for document, position in self.documents:
            document.generate_random(index)

        for transformation in self.transformations:
            transformation.generate_random()

        self.background = self.get_random_background()

    def get_random_background(self):
        image = peak_random_preloaded_image("backgrounds")
        width, height = self.get_size()
        if width is None:
            width = self.get_maximum_width()
        image_width, image_height = image.size
        x_crop = randint(0, max(1, image_width - width))
        y_crop = randint(0, max(1, image_height - height))
        return image.crop((x_crop, y_crop, x_crop + width, y_crop + height))

    def get_maximum_width(self):
        max_width = 0
        for document, position in self.documents:
            x, y = position
            max_width = max(max_width, x + document.get_maximum_width())
        return max_width

    def get_baselines(self):
        baselines = []
        for document, position in self.documents:
            document_baselines = document.get_baselines()
            for document_baseline in document_baselines:
                document_baseline[0] += position[0]
                document_baseline[1] += position[1]
                document_baseline[2] += position[0]
                document_baseline[3] += position[1]

            baselines = baselines + document_baselines
        return baselines
