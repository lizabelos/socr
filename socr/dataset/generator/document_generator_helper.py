import skimage
from os import listdir
from os.path import join, isfile
from random import randint
from abc import ABCMeta
import numpy as np

from PIL import Image, ImageFont, ImageOps
from lxml import etree

from socr.dataset.generator.generator import Generator
from socr.utils.image.degrade import gauss_distort


class DocumentGeneratorHelper(Generator):
    __metaclass__ = ABCMeta

    def __init__(self):
        with open("resources/characters.txt", "r") as content_file:
            self.labels = content_file.read() + " "

        self.fonts = self.list_resources("fonts")
        self.backgrounds = self.list_resources("backgrounds")
        self.border_holes = self.list_resources("borderHoles")
        self.corner_holes = self.list_resources("cornerHoles")
        self.phantom_patterns = self.list_resources("phantomPatterns")

        self.preloaded_background = [Image.open(background_path).convert('RGB') for background_path in self.backgrounds]
        self.preloaded_phantoms = [Image.open(phantom_path).convert("RGBA") for phantom_path in self.phantom_patterns]
        self.preloaded_phantoms_l = [ImageOps.invert(phantom.convert("L")) for phantom in self.preloaded_phantoms]
        self.preloaded_texts = []
        self.preload_text_ressources("texts")

    def preload_text_ressources(self, name):
        resources = self.list_resources(name)
        for resource in resources:
            tree = etree.parse(resource)
            root = tree.getroot()
            self.recursive_preload_text_ressources(name, root)

    def recursive_preload_text_ressources(self, name, root):
        title = root.tag.title()
        if title == "Text":
            self.preloaded_texts.append(root.text)
        else:
            for children in root.getchildren():
                self.recursive_preload_text_ressources(name, children)

    def get_random_text(self, min_size=10, max_size=50):
        text = self.preloaded_texts[randint(0, len(self.preloaded_texts) - 1)]
        text = "".join([c if self.labels.find(c) != -1 else "" for c in text])

        p = randint(0, len(text) // 2)
        return text[p:p+randint(min_size, max_size)]

    def get_number_font(self):
        return len(self.fonts)

    def list_resources(self, name):
        dir_name = "resources/" + name
        return [join(dir_name, f) for f in listdir(dir_name) if isfile(join(dir_name, f))]

    def peak_random(self, resource):
        return resource[randint(0, len(resource) - 1)]

    def get_random_font(self, font_height):
        font_name = self.peak_random(self.fonts)
        return ImageFont.truetype(font_name, font_height)

    def get_font(self, index, font_height):
        font_name = self.fonts[index]
        return ImageFont.truetype(font_name, font_height)

    def get_random_background(self, width, height):
        background_id = randint(0, len(self.backgrounds) - 1)
        image = self.preloaded_background[background_id]
        image_width, image_height = image.size
        x_crop = randint(0, max(1, image_width - width))
        y_crop = randint(0, max(1, image_height - height))
        return image.crop((x_crop, y_crop, x_crop + width, y_crop + height))

    def paster_into_random_background(self, image):
        width, height = image.size
        background = self.get_random_background(width, height)
        background.paste(image, (0, 0), ImageOps.invert(image.convert("L")))
        return background

    def add_random_phantom_patterns(self, image):
        image_width, image_height = image.size
        n = randint(0, 10)
        for i in range(0, n):
            phantom_id = randint(0, len(self.phantom_patterns) - 1)
            image.paste(self.preloaded_phantoms[phantom_id], (randint(0, image_width), randint(0, image_height)),
                        self.preloaded_phantoms_l[phantom_id])
        return image

    def augment(self, image):
        c = gauss_distort([image[0], image[1], image[2]])
        image = np.stack(c, axis=0)
        image = skimage.util.random_noise(image)
        return image

    def generate(self, index):
        raise NotImplementedError()
