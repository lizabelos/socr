import math
from os import listdir
from os.path import isfile, join
from random import uniform

from PIL import Image
from torch.utils.data import Dataset

from socr.utils.image import image_pillow_to_numpy


class FileDataset(Dataset):

    def __init__(self):
        self.list = []

    def recursive_list(self, path):
        if isfile(path):
            if not path.endswith(".result.jpg"):
                if path.endswith(".jpg") or path.endswith(".png"):
                    self.list.append(path)
        else:
            for file_name in listdir(path):
                self.recursive_list(join(path, file_name))

    def sort(self):
        self.list.sort()

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        image = Image.open(self.list[index]).convert("RGB")

        width, height = image.size
        new_width = math.sqrt(6 * (10 ** 5) * width / height)
        new_width = int(new_width)
        new_height = height * new_width // width

        if new_width < width:
            resized = image.resize((new_width, new_height), Image.ANTIALIAS)
        else:
            resized = image

        return image_pillow_to_numpy(resized), image_pillow_to_numpy(image), self.list[index]