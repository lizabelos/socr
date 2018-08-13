from datetime import datetime
import math
import os
from random import randint, random, uniform

import torch
import numpy as np
from os.path import isfile, join

from PIL import Image
from lxml import etree
from scipy.ndimage import rotate
from torch.utils.data.dataset import Dataset


class ICDARDocumentSet(Dataset):

    def __init__(self, helper, path, loss=None, transform=True):
        self.loss = loss
        self.helper = helper
        self.labels = []
        self.recursive_list(path)
        self.transform = transform

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
        if not os.path.isfile(image_fullpath):
            image_fullpath = join(path, "../" + image_filename)

        if not os.path.isfile(image_fullpath):
            return

        regions = []
        for children in root.getchildren():
            if children.tag.title().endswith("Textregion"):
                region = self.parse_line(children)
                if region is not None:
                    regions = regions + region
                else:
                    return None

        if len(regions) == 0:
            print("Warning : 0 len regions for " + image_filename)
            regions = []

        self.labels.append((image_fullpath, regions))

    def parse_line(self, root):
        results = []
        for children in root.getchildren():
            if children.tag.title().endswith("Textline"):
                result = self.parse_region(children)
                if result is not None:
                    results = results + result
                else:
                    return None
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
            print("Warning : not region fonud")
            return None

        if baseline is None:
            print("Warning : not baseline fonud")
            return None

        max_height = 0

        results = []

        for i in range(0, len(region) // 4 - 1):
            j = i + 1

            h1 = abs(region[2 * i + 1] - region[len(region) - (2 * i + 1)])
            h2 = abs(region[2 * j + 1] - region[len(region) - (2 * j + 1)])

            max_height = max((h1 + h2) / 2, max_height)

        for i in range(0, len(baseline) // 2 - 1):
            results.append((baseline[i * 2], baseline[i * 2 + 1], baseline[i * 2 + 2], baseline[i * 2 + 3], max_height))

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

        if os.path.isfile(image_path + ".resized.jpg"):
            image = Image.open(image_path + ".resized.jpg").convert('RGB')
            image_size = np.load(image_path + ".size.np.npy")
            width = image_size[0]
            height = image_size[1]
            new_width, new_height = image.size
        else:
            image = Image.open(image_path).convert('RGB')
            width, height = image.size
            new_width = math.sqrt(6 * (10 ** 5) * width / height)
            # new_width = new_width * uniform(0.8, 1.2)
            new_width = int(new_width)
            new_height = height * new_width // width
            image = image.resize((new_width, new_height), Image.ANTIALIAS)
            image.save(image_path + ".resized.jpg")
            np.save(image_path + ".size.np.npy", np.array([width, height]))



        new_regions = []
        for region in regions:
            x0, y0, x1, y1, h = region
            x0 = x0 * new_height // height
            y0 = y0 * new_height // height
            x1 = x1 * new_height // height
            y1 = y1 * new_height // height
            h = h * new_height // height
            new_regions.append([x0, y0, x1, y1, h])

        if len(new_regions) == 0:
            label = np.zeros((new_height, new_width, 2), dtype=np.float32)
        else:
            label = self.loss.document_to_ytrue(np.array([new_width, new_height], dtype='int32'),
                                            np.array(new_regions, dtype='int32'))

        image = np.array(image, dtype='float') / 255.0

        # angle = randint(-45, 45)
        # label = rotate(label, angle, order=0)
        # image = rotate(image, angle, order=0)

        # image = np.swapaxes(image, 0, 2)
        # image = np.swapaxes(image, 1, 2)

        # label = np.swapaxes(label, 0, 2)
        # label = np.swapaxes(label, 1, 2)

        # if self.transform:
        #     image = self.helper.augment(image, distort=False)

            # c = gauss_distort([image[0], image[1], image[2], label[0], label[1]])
            # image = np.stack((c[0], c[1], c[2]), axis=0)
            # label = np.stack((c[3], c[4]), axis=0)


        return torch.from_numpy(image), torch.from_numpy(label)

    def __len__(self):
        return len(self.labels) * 8
