from os import listdir
from os.path import isfile, join

from PIL import Image
from torch.utils.data import Dataset

from socr.utils.image import image_pillow_to_numpy


class FileDataset(Dataset):

    def __init__(self, maximum_height):
        self.list = []
        self.maximum_height = maximum_height

    def recursive_list(self, path):
        if isfile(path):
            if not path.endswith(".result.jpg") and path.endswith(".jpg"):
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

        if height > self.maximum_height:
            resized = image.resize((width * self.maximum_height // height, self.maximum_height), Image.ANTIALIAS)
        else:
            resized = image

        return image_pillow_to_numpy(resized), image_pillow_to_numpy(image), self.list[index]