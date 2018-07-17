import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from scribbler.generator import DocumentGenerator


class OrientedDocumentGeneratedSet(Dataset):

    def __init__(self, helper, loss, transform=True):
        self.loss = loss
        self.document_generator = DocumentGenerator()

    def __getitem__(self, index):
        image, label = self.generate_image_with_label(index)

        image = np.array(image, dtype='float') / 255.0

        return torch.from_numpy(image), torch.from_numpy(self.loss.document_to_ytrue([image.shape[2], image.shape[1]], label))

    def __len__(self):
        return self.document_generator.count()

    def generate_image_with_label(self, index):
        image, document = self.document_generator.get(index)
        return image, document
