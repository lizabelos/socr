from random import randint

from scribbler.document.document import Document
from scribbler.document.parser import parse_document
from scribbler.resources.resources_helper import list_resources
from scribbler.dataset import Dataset
from scribbler.document.document_handwriting_line import DocumentHandwrittingLine
from scribbler.document.document_text_line import DocumentTextLine
from scribbler.utils.image.image import image_pillow_to_numpy


class LineGenerator(Dataset):

    def __init__(self, height=64):

        self.hand_text_line = DocumentHandwrittingLine(height=height)
        self.hand_text_documnt = Document(height=height)
        self.hand_text_documnt.append_document(self.hand_text_line)

        self.printed_text_line = DocumentTextLine(height=height)
        self.printed_text_document = Document(height=height)
        self.printed_text_document.append_document(self.printed_text_line)

    def count(self):
        return self.printed_text_line.count_ressource() + self.hand_text_line.count_ressource()

    def get(self, index):
        if index >= self.printed_text_line.count_ressource():
            index = index - self.printed_text_line.count_ressource()
            text_line = self.hand_text_line
            text_document = self.hand_text_documnt
        else:
            text_line = self.printed_text_line
            text_document = self.printed_text_document

        text_document.generate_random(index)
        return text_document.to_image(), text_line.get_text()


class DocumentGenerator(Dataset):

    def __init__(self):
        self.documents = [parse_document(path) for path in list_resources("structures")]

    def count(self):
        return 1024

    def get(self, index):
        document = self.documents[randint(0, len(self.documents) - 1)]
        document.generate_random()
        image = document.to_image()

        # image = image_pillow_to_numpy(image)
        return image, document.get_baselines()
