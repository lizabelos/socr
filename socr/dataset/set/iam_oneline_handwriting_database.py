import os

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from socr.utils.image import image_pillow_to_numpy


class IAMOneLineHandwritingDatabase(Dataset):

    def __init__(self, helper, path, height=48, labels="", transform=True):
        self.height = height

        self.labels = []
        self.transform = transform
        self.images_path = os.path.join(path, "lineImages")
        self.txt_path = os.path.join(path, "ascii")

        for d1 in os.listdir(self.txt_path):
            d1_txt_path = os.path.join(self.txt_path, d1)
            for d2 in os.listdir(d1_txt_path):
                d2_txt_path = os.path.join(d1_txt_path, d2)
                for txt in os.listdir(d2_txt_path):
                    self.parse_txt(d1, d2, os.path.join(d2_txt_path, txt))

        self.document_helper = helper

    def parse_txt(self, d1, d2, txt_path):
        base = os.path.splitext(os.path.basename(txt_path))[0]

        with open(txt_path) as f:
            content = f.readlines()

        state = 0
        i = 0
        for line in content:
            line = line.rstrip()
            if line == "CSR:":
                state = 1
            elif state == 1:
                state = 2
            elif state == 2:
                if i == 0:
                    ids = [d1, d2, base, "%02d" % (i,)]
                    image_path = self.get_image_path(ids)
                    if not os.path.isfile(image_path):
                        i = i + 1

                ids = [d1, d2, base, "%02d" % (i,)]
                image_path = self.get_image_path(ids)
                if not os.path.isfile(image_path):
                    # Just skip invalid data
                    return
                self.labels.append((ids, line))
                i = i + 1

    def get_image_path(self, ids):
        return os.path.join(self.images_path, ids[0] + "/" + ids[1] + "/" + ids[2] + "-" + ids[3] + ".tif")

    def get_path(self, index):
        ids, text = self.labels[index]
        image_path = self.get_image_path(ids)
        return image_path, text

    def get_corpus(self):
        corpus = ""
        for id, text in self.labels:
            if corpus == "":
                corpus = text
            else:
                corpus = corpus + ". " + text
        return corpus

    def __getitem__(self, index):
        ids, text = self.labels[index]
        image_path = self.get_image_path(ids)

        # Load the image
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        if self.height is not None:
            image = image.resize((width * self.height // height, self.height), Image.ANTIALIAS)

        if self.transform:
            image = self.document_helper.paster_into_random_background(image)

        image = image_pillow_to_numpy(image)
        if self.transform:
            image = self.document_helper.augment(image)

        return torch.from_numpy(image), text

    def __len__(self):
        return len(self.labels)
