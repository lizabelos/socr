import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data.dataset import Dataset

from socr.dataset.generator.document_generator import DocumentGenerator
from socr.dataset.generator.oriented_document_generator import OrientedDocumentGenerator
from socr.utils.image import image_pillow_to_numpy


class OrientedDocumentGeneratedSet(Dataset):

    def __init__(self, helper, loss, transform=True):
        self.loss = loss
        self.document_generator = OrientedDocumentGenerator(helper)

    def __getitem__(self, index):
        image, label = self.generate_image_with_label(index)
        image = image_pillow_to_numpy(image)
        return torch.from_numpy(image), torch.from_numpy(self.loss.document_to_ytrue([image.shape[2], image.shape[1]], label))

    def __len__(self):
        return self.document_generator.helper.get_number_font()

    def generate_image_with_label(self, index):
        image, document = self.document_generator.generate(index)
        return image, document
