import os

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from socr.utils.image import image_pillow_to_numpy


class IAMWashington(Dataset):

    def __init__(self, helper, path, height=48, labels="", transform=True, loss=None):
        self.height = height
        self.loss = loss

        self.transform = transform
        self.labels = []
        self.images_path = os.path.join(path, "data/line_images_normalized")
        self.txt_path = os.path.join(path, "ground_truth/transcription.txt")

        with open(self.txt_path) as f:
            content = f.readlines()

        for line in content:
            id, txt = line.split(" ")
            txt = txt.replace("s_","")
            txt = txt.replace("-","")
            txt = txt.replace("|"," ")
            txt = txt.replace("\n","")
            path = os.path.join(self.images_path, id + ".png")
            self.labels.append((path, txt))

        self.document_helper = helper

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
        image_path, text = self.labels[index]

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


        return torch.from_numpy(image), (self.loss.preprocess_label(text, width * self.height // height), text)

    def __len__(self):
        return len(self.labels)
