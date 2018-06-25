import os
from os.path import isfile, join

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from socr.utils.image import image_pillow_to_numpy


class ImageWithTextDatabase(Dataset):

    def __init__(self, helper, path, height):
        self.document_helper = helper
        self.height = height
        self.images = []
        self.labels = ""
        self.recursive_list(path)

    def recursive_list(self, path):
        for file_name in os.listdir(path):
            if isfile(join(path, file_name)):
                if file_name.endswith(".jpg") or file_name.endswith(".png"):
                    image_path = join(path, file_name)
                    label_path = os.path.splitext(os.path.splitext(image_path)[0])[0] + '.gt.txt'
                    with open(label_path, "r") as content_file:
                        label = content_file.read()
                    self.images.append((image_path, label))
                    self.add_label(label)
            else:
                self.recursive_list(join(path, file_name))

    def add_label(self, text):
        for c in text:
            if self.labels.find(c) == -1:
                self.labels = self.labels + c

    def get_labels(self):
        return self.labels

    def __getitem__(self, index):
        image_path, text = self.images[index]

        # Load the image
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        image = image.resize((width * self.height // height, self.height), Image.ANTIALIAS)
        image = image_pillow_to_numpy(image)

        return torch.from_numpy(image), text

    def __len__(self):
        return len(self.images)
