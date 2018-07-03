import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data.dataset import Dataset

from socr.dataset.generator.document_generator import DocumentGenerator
from socr.utils.image import image_pillow_to_numpy


class DocumentGeneratedSet(Dataset):

    def __init__(self, helper, loss=None, width=256, height=256):
        self.loss = loss
        self.width = width
        self.height = height
        self.document_generator = DocumentGenerator(helper)

    def __getitem__(self, index):
        image, label = self.generate_image_with_label(index)
        image = image_pillow_to_numpy(image)

        if self.loss is not None:
            return torch.from_numpy(image), torch.from_numpy(self.loss.document_to_ytrue([image.shape[2], image.shape[1]], list(label)))
        else:
            return torch.from_numpy(image), None

    def __len__(self):
        return self.document_generator.helper.get_number_font()

    def generate_image_with_label(self, index):
        image, document = self.document_generator.generate(index)
        width, height = image.size
        if self.width is not None:
            image = image.resize((self.width, self.height), Image.ANTIALIAS)
        else:
            image = image.resize((width * self.height // height, self.height), Image.ANTIALIAS)
        return image, document
