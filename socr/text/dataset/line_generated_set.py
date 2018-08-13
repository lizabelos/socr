from random import randint


import torch
from PIL import ImageDraw, Image
from torch.utils.data.dataset import Dataset

from socr.utils.image import image_pillow_to_numpy


class LineGenerator:

    def __init__(self, helper, labels):
        self.helper = helper
        self.lst = labels

    def generate(self, index):
        text_start = randint(0, 5)
        text_end = randint(0, 50)
        text_top = randint(0, 5)
        text_bottom = randint(0, 5)

        text = self.helper.get_random_text()

        font_height = randint(8, 30)
        font = self.helper.get_font(index, font_height)

        total_width = 0
        lefts = []
        widths = []
        max_height = 0

        for i in range(0, len(text)):
            while True:
                try:
                    font_width, font_height = font.getsize(text[i])
                    break
                except OSError:
                    # print("Warning : execution context too long ! Continuing...")
                    font = self.helper.get_font(0, font_height)

            lefts.append(text_start + total_width)
            widths.append(font_width)
            total_width = total_width + font_width
            max_height = max(max_height, font_height)

        image_width = text_start + text_end + total_width
        image_height = text_top + text_bottom + font_height
        image = Image.new('RGBA', (image_width, image_height))
        image_draw = ImageDraw.Draw(image)

        for i in range(0, len(text)):
            image_draw.text((lefts[i], text_top), text[i], font=font, fill=(randint(0, 128), randint(0, 128), randint(0, 128)))
            widths[i] = widths[i] / image_width
            lefts[i] = lefts[i] / image_width

        image = image.rotate(randint(-3, 3), expand=True, resample=Image.BICUBIC)

        image_width, image_height = image.size
        background = self.helper.get_random_background(image_width, image_height)
        background.paste(image, (0,0), image)

        return background, "".join(text)

    def get_labels(self):
        return self.lst


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
