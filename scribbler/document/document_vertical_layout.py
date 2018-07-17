from PIL import Image

from scribbler.document import AbstractDocument


class DocumentVerticalLayout(AbstractDocument):

    def __init__(self, width=None, height=None, vertical_padding=0, horizontal_padding=0, parent=None):
        super().__init__(width, height, parent)
        self.documents = []
        self.vertical_padding = vertical_padding
        self.horizontal_padding = horizontal_padding

    def append_document(self, document, position=(0,0)):
        if not isinstance(document, AbstractDocument):
            raise TypeError("this is not a document")

        self.documents.append((document, position))

    def to_image(self):
        image = Image.new('RGBA', self.get_size())

        width, height = self.get_size()

        children_width, children_height = width, height // len(self.documents)
        current_y = self.vertical_padding

        for document, position in self.documents:
            document_image = document.to_image()
            image.paste(document_image, (self.horizontal_padding, current_y), document_image)
            current_y = current_y + children_height + self.vertical_padding

        return image

    def get_children_size(self):
        width, height = self.get_size()
        return width - (self.horizontal_padding * 2), \
               height // len(self.documents) - (self.vertical_padding * (len(self.documents) + 1))

    def generate_random(self, index=-1):
        for document, position in self.documents:
            document.generate_random(index)

    def get_baselines(self):
        width, height = self.get_size()

        children_width, children_height = width, height // len(self.documents)
        current_y = self.vertical_padding

        baselines = []
        for document, position in self.documents:
            document_baselines = document.get_baselines()
            for document_baseline in document_baselines:
                document_baseline[0] += self.horizontal_padding
                document_baseline[1] += current_y
                document_baseline[2] += self.horizontal_padding
                document_baseline[3] += current_y

            current_y = current_y + children_height + self.vertical_padding
            baselines = baselines + document_baselines
        return baselines
