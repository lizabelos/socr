import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from socr.dataset.generator.line_generator import LineGenerator
from socr.utils.image import image_pillow_to_numpy


class LineGeneratedSet(Dataset):

    def __init__(self, helper, labels="", width=None, height=32, transform=True, loss=None):
        self.width = width
        self.height = height
        self.document_generator = LineGenerator(helper, labels)
        self.transform = transform
        self.loss = loss

    def __getitem__(self, index):
        image_pillow, label = self.generate_image_with_label(index)
        image = image_pillow_to_numpy(image_pillow)

        if self.transform:
            image = self.document_generator.helper.augment(image)

        return torch.from_numpy(image), self.loss.preprocess_label(label), label

    def __len__(self):
        return self.document_generator.helper.get_number_font() * 5

    def generate_image_with_label(self, index):
        index = index % self.document_generator.helper.get_number_font()

        image, document, text = self.document_generator.generate(index)
        width, height = image.size
        if self.width is not None:
            image = image.resize((self.width, self.height), Image.ANTIALIAS)
        else:
            image = image.resize((width * self.height // height, self.height), Image.ANTIALIAS)

        return image, (document, text)

    def get_corpus(self):
        return ". ".join(self.document_generator.helper.preloaded_texts)
