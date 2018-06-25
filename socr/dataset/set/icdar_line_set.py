import os
from random import randint

import torch
from os.path import isfile, join

from PIL import Image
from lxml import etree
from torch.utils.data.dataset import Dataset

from socr.dataset.generator.document_generator import DocumentGenerator
from socr.utils.image import image_pillow_to_numpy


class ICDARLineSet(Dataset):

    def __init__(self, helper, path, height):
        self.height = height
        self.labels = []
        self.document_generator = DocumentGenerator(helper)
        self.recursive_list(os.path.join(path, "page"))

    def recursive_list(self, path):
        for file_name in os.listdir(path):
            if isfile(join(path, file_name)):
                if file_name.endswith(".xml"):
                    self.parse_xml(path, file_name)
            else:
                self.recursive_list(join(path, file_name))

    def parse_xml(self, path, file_name):
        tree, root = None, None

        try:
            tree = etree.parse(join(path, file_name))
            root = tree.getroot()
        except:
            print("Warning : invalid xml file '" + file_name + "'")
            return

        for children in root.getchildren():
            if children.tag.title().endswith("Page"):
                self.parse_page(path, children)

    def parse_page(self, path, root):
        root_dict = {}
        for name, value in root.attrib.items():
            root_dict[name] = value

        imageFilename = root_dict["imageFilename"]
        imageFullpath = join(path, "../" + imageFilename)

        for children in root.getchildren():
            print(children.tag.title())
            if children.tag.title().endswith("Textregion"):
                text, region = self.parse_region(children)
                if text is not None and text != "":
                    self.labels.append((imageFullpath, region, text))

    def parse_region(self, root):
        region = None
        text = None

        for children in root.getchildren():
            print("---" + children.tag.title())
            if children.tag.title().endswith("Coords"):
                region = self.parse_coords(children)
            if children.tag.title().endswith("Textequiv"):
                text = self.parse_text(children)

        return text, region

    def parse_coords(self, root):
        root_dict = {}
        for name, value in root.attrib.items():
            root_dict[name] = value

        points_s = root_dict["points"].split(" ")

        min_x = 9999
        max_x = 0

        min_y = 9999
        max_y = 0

        for s in points_s:
            x, y = s.split(",")
            x = int(x)
            y = int(y)
            min_x = min(x, min_x)
            min_y = min(y, min_y)
            max_x = max(x, max_x)
            max_y = max(y, max_y)

        return min_x, min_y, max_x, max_y

    def parse_text(self, root):
        for children in root.getchildren():
            return children.text

    def __getitem__(self, index):
        image_path, region, text = self.labels[index]
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print("Warning : can't open :" + image_path)
            return self.__getitem__(randint(0, self.__len__() - 1))
        image = image.crop(region)
        width, height = image.size
        image = image.resize((width * self.height // height, self.height), Image.ANTIALIAS)
        image = image_pillow_to_numpy(image)
        return torch.from_numpy(image), text

    def __len__(self):
        return len(self.labels)
