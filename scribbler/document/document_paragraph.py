from PIL import Image

from scribbler.document import AbstractDocument, DocumentHandwrittingLine, DocumentTextLine


class DocumentParagraph(AbstractDocument):

    def __init__(self, width=None, height=None, parent=None, font_size=None, first_line_indent=0, type="default"):
        super().__init__(width, height, parent)
        self.font_size = font_size
        self.documents = []
        self.fist_line_indent = first_line_indent
        if type == "handwriting":
            self.type = DocumentHandwrittingLine
        else:
            self.type = DocumentTextLine

    def append_document(self, width, height, position, text=""):
        if len(self.documents) == 0:
            document = self.type(width, height, text=text)
        else:
            top, _ = self.documents[len(self.documents)-1]
            document = self.type(width, height, text=text, top=top)

        x, y = position
        assert x >= 0
        assert y >= 0

        self.documents.append((document, position))

    def to_image(self):
        image = Image.new('RGBA', self.get_size())

        for document, position in self.documents:
            document_image = document.to_image()
            image.paste(document_image, position, document_image)

        return image

    def generate_random(self, index=-1):
        if len(self.documents) == 0 and self.font_size is not None:
            width, height = self.parent.get_children_size()
            for y in range(0, height - self.font_size, self.font_size):
                if y == 0:
                    self.append_document(width, self.font_size, (0, y))
                else:
                    self.append_document(width - self.fist_line_indent, self.font_size, (self.fist_line_indent, y))

        for document, position in self.documents:
            document.generate_random(index)

    def get_baselines(self):
        baselines = []
        for document, position in self.documents:
            document_baselines = document.get_baselines()
            for document_baseline in document_baselines:
                document_baseline[0] += position[0]
                document_baseline[1] += position[1]
                document_baseline[2] += position[0]
                document_baseline[3] += position[1]

            baselines = baselines + document_baselines
        return baselines
