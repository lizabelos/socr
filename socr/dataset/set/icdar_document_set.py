import os
from random import randint, random

import torch
import numpy as np
from os.path import isfile, join

from PIL import Image
from lxml import etree
from torch.utils.data.dataset import Dataset

from socr.dataset.generator.document_generator import DocumentGenerator
from socr.utils.image import image_pillow_to_numpy


class ICDARDocumentSet(Dataset):

    def __init__(self, helper, path, loss=None):
        self.loss = loss
        self.labels = []
        self.document_generator = DocumentGenerator(helper)
        self.recursive_list(path)

    def recursive_list(self, path):
        for file_name in os.listdir(path):
            if isfile(join(path, file_name)):
                if file_name.endswith(".xml"):
                    self.parse_xml(path, file_name)
            else:
                self.recursive_list(join(path, file_name))

    def parse_xml(self, path, file_name):
        tree = etree.parse(join(path, file_name))
        root = tree.getroot()

        for children in root.getchildren():
            if children.tag.title().endswith("Page"):
                self.parse_page(path, children)

    def parse_page(self, path, root):
        root_dict = {}
        for name, value in root.attrib.items():
            root_dict[name] = value

        image_filename = root_dict["imageFilename"]
        # image_fullpath = join(path, "../" + image_filename)
        image_fullpath = join(path, image_filename)

        regions = []
        for children in root.getchildren():
            if children.tag.title().endswith("Textregion"):
                region = self.parse_line(children)
                if region is not None:
                    regions = regions + region

        if len(regions) > 0:
            self.labels.append((image_fullpath, regions))

    def parse_line(self, root):
        results = []
        for children in root.getchildren():
            if children.tag.title().endswith("Textline"):
                result = self.parse_region(children)
                if result is not None:
                    results = results + result
        return results

    def parse_region(self, root):
        region = None
        baseline = None

        for children in root.getchildren():
            if children.tag.title().endswith("Coords"):
                region = self.parse_coords(children)
            if children.tag.title().endswith("Baseline"):
                baseline = self.parse_baseline(children)

        if region is None:
            return None

        results = []

        for i in range(0, len(region) // 4 - 1):
            j = i + 1

            x0 = region[2 * i + 0]
            y0 = region[2 * i + 1]
            x1 = region[len(region) - (2 * i + 2)]
            y1 = region[len(region) - (2 * i + 1)]

            x2 = region[2 * j + 0]
            y2 = region[2 * j + 1]
            x3 = region[len(region) - (2 * j + 2)]
            y3 = region[len(region) - (2 * j + 1)]

            h1 = abs(region[2 * i + 1] - region[len(region) - (2 * i + 1)])
            h2 = abs(region[2 * j + 1] - region[len(region) - (2 * j + 1)])

            results.append((int((x0 + x1) / 2), int((y0 + y1) / 2), int((x2 + x3) / 2), int((y2 + y3) / 2), (h1 + h2) / 2))

        return results

    def parse_coords(self, root):
        root_dict = {}
        for name, value in root.attrib.items():
            root_dict[name] = value

        points_s = root_dict["points"].split(" ")
        points = []

        for s in points_s:
            x, y = s.split(",")
            points.append(int(x))
            points.append(int(y))

        return points

    def parse_baseline(self, root):
        root_dict = {}
        for name, value in root.attrib.items():
            root_dict[name] = value

        points_s = root_dict["points"].split(" ")
        points = []

        for s in points_s:
            x, y = s.split(",")
            points.append(int(x))
            points.append(int(y))

        return points

    def __getitem__(self, index):
        image_path, regions = self.labels[index % len(self.labels)]
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print("Warning : can't open :" + image_path)
            return self.__getitem__(randint(0, self.__len__() - 1))

        width, height = image.size
        # new_height = randint(height // 8, height // 2)
        new_height = randint(600, 800)
        image = image.resize((width * new_height // height, new_height), Image.ANTIALIAS)

        new_regions = []
        for region in regions:
            x0, y0, x1, y1, h = region
            x0 = x0 * new_height // height
            y0 = y0 * new_height // height
            x1 = x1 * new_height // height
            y1 = y1 * new_height // height
            h = h * new_height // height
            new_regions.append([x0, y0, x1, y1, h])

        label = self.loss.document_to_ytrue([width * new_height // height, new_height], new_regions)

        image = image_pillow_to_numpy(image)


        # Make the crop
        crop_width = 256
        crop_height = 256
        crop_x = randint(0, image.shape[2] - crop_width)
        crop_y = randint(0, image.shape[1] - crop_height)

        label = label[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        image = np.stack([image[i][crop_y:crop_y + crop_height, crop_x:crop_x + crop_width] for i in range(0, 3)], axis=0)



        return torch.from_numpy(image), torch.from_numpy(label)

    def __len__(self):
        return len(self.labels) * 1024
