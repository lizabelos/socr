import os
from lxml import etree

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from socr.utils.image import image_pillow_to_numpy


class IAMHandwritingLineDatabase(Dataset):

    def __init__(self, helper, path, height=32, labels="", transform=True, loss=None):
        self.height = height
        self.loss = loss

        self.labels = []
        self.transform = transform
        self.images_path = os.path.join(path, "lines")
        self.xmls_path = os.path.join(path, "xml")

        for xml_name in os.listdir(self.xmls_path):
            self.parse_xml(os.path.join(self.xmls_path, xml_name))

        self.document_helper = helper

    def parse_xml(self, xml_path):
        tree = etree.parse(xml_path)
        self.parse_xml_tree(xml_path, tree.getroot())

    def parse_xml_tree(self, xml_path, root):

        for children in root.getchildren():
            if children.tag.title() == "Line":
                root_dict = {}
                for name, value in children.attrib.items():
                    root_dict[name] = value

                line = self.parse_xml_tree_line(children)
                self.labels.append((root_dict["id"], line))
            else:
                self.parse_xml_tree(xml_path, children)

    def parse_xml_tree_line(self, root):
        text_lines = []

        for children in root.getchildren():
            if children.tag.title() == "Word":
                text_lines.append(self.parse_xml_tree_word(children))

        return " ".join(text_lines)

    def parse_xml_tree_word(self, root):
        root_dict = {}
        for name, value in root.attrib.items():
            root_dict[name] = value

        return root_dict["text"]

    def get_corpus(self):
        corpus = ""
        for id, text in self.labels:
            if corpus == "":
                corpus = text
            else:
                corpus = corpus + ". " + text
        return corpus

    def __getitem__(self, index):
        id, text = self.labels[index]
        ids = id.split("-")
        image_path = os.path.join(self.images_path, ids[0] + "/" + ids[0] + "-" + ids[1] + "/" + ids[0] + "-" + ids[1] + "-" + ids[2] + ".png")

        # Load the image
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        image = image.resize((width * self.height // height, self.height), Image.ANTIALIAS)

        if self.transform:
            image = self.document_helper.paster_into_random_background(image)

        image = image_pillow_to_numpy(image)
        return torch.from_numpy(image), (self.loss.preprocess_label(text, width * self.height // height), text)

    def __len__(self):
        return len(self.labels)
