import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from socr.dataset.generator.character_generator import CharacterGenerator
from socr.utils.image import image_pillow_to_numpy


class CharacterGeneratorSet(Dataset):

    def __init__(self, helper, labels, width=32, height=32):
        self.width = width
        self.height = height
        self.document_generator = CharacterGenerator(helper, labels, width, height)

    def __getitem__(self, index):
        image, label = self.generate_image_with_label(index)
        image = image_pillow_to_numpy(image)

        return torch.from_numpy(image), label

    def __len__(self):
        return self.document_generator.helper.get_number_font()

    def generate_image_with_label(self, index):
        image, document = self.document_generator.generate(index)
        image = image.resize((self.width, self.height), Image.ANTIALIAS)
        return image, document

