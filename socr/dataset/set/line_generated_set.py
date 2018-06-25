import torch
from PIL import Image
import numpy as np
from torch.utils.data.dataset import Dataset

from socr.dataset.generator.line_generator import LineGenerator
from socr.utils.image.degrade import gauss_degrade, gauss_distort
from socr.utils.image import image_pillow_to_numpy


class LineGeneratedSet(Dataset):

    def __init__(self, helper, labels="", width=None, height=32, transform=True):
        self.width = width
        self.height = height
        self.document_generator = LineGenerator(helper, labels)
        self.transform = transform

    def __getitem__(self, index):
        image_pillow, label = self.generate_image_with_label(index)
        image = image_pillow_to_numpy(image_pillow)

        c = [None, None, None]
        for i in range(0, 3):
            c[i] = gauss_distort([gauss_degrade(image[i])])[0]

        image = np.stack(image, axis=0)

        return torch.from_numpy(image), label

    def __len__(self):
        return self.document_generator.helper.get_number_font() * 5

    def generate_image_with_label(self, index):
        index = index % self.document_generator.helper.get_number_font()

        image, document = self.document_generator.generate(index)
        width, height = image.size
        if self.width is not None:
            image = image.resize((self.width, self.height), Image.ANTIALIAS)
        else:
            image = image.resize((width * self.height // height, self.height), Image.ANTIALIAS)

        if self.transform:
            image = self.document_generator.helper.augment(image)

        return image, document

