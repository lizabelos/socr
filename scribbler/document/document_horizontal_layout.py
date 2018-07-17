from PIL import Image

from scribbler.document import AbstractDocument


class DocumentHorizontalLayout(AbstractDocument):

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

        children_width, children_height = width // len(self.documents), height
        current_x = self.horizontal_padding

        for document, position in self.documents:
            document_image = document.to_image()
            image.paste(document_image, (current_x, self.vertical_padding), document_image)
            current_x = current_x + children_width + self.horizontal_padding

        return image

    def get_children_size(self):
        width, height = self.get_size()
        return width // len(self.documents) - (self.horizontal_padding * (len(self.documents) + 1)),\
               height - (self.vertical_padding * 2)

    def generate_random(self, index=-1):
        for document, position in self.documents:
            document.generate_random(index)

    def get_baselines(self):
        width, height = self.get_size()

        children_width, children_height = width // len(self.documents), height
        current_x = self.horizontal_padding

        baselines = []
        for document, position in self.documents:
            document_baselines = document.get_baselines()
            for document_baseline in document_baselines:
                document_baseline[0] += current_x
                document_baseline[1] += self.vertical_padding
                document_baseline[2] += current_x
                document_baseline[3] += self.vertical_padding

            current_x = current_x + children_width + self.horizontal_padding
            baselines = baselines + document_baselines
        return baselines
