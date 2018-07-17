import os

from PIL import Image, ImageOps
from lxml import etree

from scribbler.dataset import Dataset


class IAMHandwritingLineDataset(Dataset):
    def __init__(self, path):
        self.labels = []
        self.images_path = os.path.join(path, "lines")
        self.xmls_path = os.path.join(path, "xml")

        for xml_name in os.listdir(self.xmls_path):
            self.parse_xml(os.path.join(self.xmls_path, xml_name))

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

    def count(self):
        return len(self.labels)

    def get(self, index):
        id, text = self.labels[index]
        ids = id.split("-")
        image_path = os.path.join(self.images_path, ids[0] + "/" + ids[0] + "-" + ids[1] + "/" + ids[0] + "-" + ids[1] + "-" + ids[2] + ".png")

        # Load the image
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        return image, text