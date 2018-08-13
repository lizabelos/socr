import math
from os import listdir
from os.path import isfile, join
from configparser import ConfigParser

import torch
from PIL import Image
from torch.utils.data import Dataset

from socr.utils.image import image_pillow_to_numpy
from socr.utils.logger import print_normal


def parse_datasets_configuration_file(helper, path="datasets.cfg", with_document=False, with_line=False, training=False, testing=False, args=None):

    config = ConfigParser()
    config.read(path)

    datasets = []

    for section in config.sections():
        dict = {}
        options = config.options(section)
        for option in options:
            dict[option] = config.get(section, option)

        if dict["for"] != "Document" and dict["for"] != "Line":
            print("Invalid for : '" + dict["for"] + "'")

        if dict["for"] == "Document" and with_document == False:
            continue

        if dict["for"] == "Line" and with_line == False:
            continue

        if training and "train" in dict:
            print_normal("Loading train database " + str(dict["type"]) + "...")
            dataset = parse_dataset(helper, dict["type"], dict["train"], args)
            if dataset is not None:
                datasets.append(dataset)

        if testing and "test" in dict:
            print_normal("Loading test database " + str(dict["type"]) + "...")
            dataset = parse_dataset(helper, dict["type"], dict["test"], args)
            if dataset is not None:
                datasets.append(dataset)

    if len(datasets) == 0:
        return None

    return torch.utils.data.ConcatDataset(datasets)

def parse_dataset(helper, type, path, args=None):

    if args is None:
        args = {}

    if type == "DocumentGenerator":
        if path == "yes":
            from socr.line.dataset.scribbler_document_set import ScribblerDocumentSet
            return ScribblerDocumentSet(helper, **args)
        else:
            return None

    if type == "LineGenerator":
        if path == "yes":
            from socr.text.dataset.line_generated_set import LineGeneratedSet
            return LineGeneratedSet(helper, **args)
        else:
            return None

    if type == "ICDAR-Baseline":
        from socr.line.dataset.icdar_document_set import ICDARDocumentSet
        return ICDARDocumentSet(helper, path, **args)

    if type == "IAM":
        from socr.text.dataset.iam_handwriting_line_database import IAMHandwritingLineDatabase
        return IAMHandwritingLineDatabase(helper, path, **args)

    if type == "IAM-Word":
        from socr.text.dataset.iam_handwriting_word_database import IAMHandwritingWordDatabase
        return IAMHandwritingWordDatabase(helper, path, **args)

    if type == "IAM-One-Line":
        from socr.text.dataset.iam_oneline_handwriting_database import IAMOneLineHandwritingDatabase
        return IAMOneLineHandwritingDatabase(helper, path, **args)

    if type == "IAM-Washington":
        from socr.text.dataset.iam_washington import IAMWashington
        return IAMWashington(helper, path, **args)

    print("Warning : unknown database type : '" + type + "'")

    return None



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