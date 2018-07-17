from PIL import Image, ImageFont, ImageDraw

from scribbler.document import AbstractDocument
from scribbler.resources.resources_helper import peak_random_resource, peak_random_preloaded_text, peak_resource, \
    count_resource


class DocumentTextLine(AbstractDocument):

    def __init__(self, width=None, height=None, parent=None, top=None, text=""):
        super().__init__(width, height, parent)

        if not isinstance(text, str):
            raise TypeError("text is not a str")
        if top is not None and not isinstance(top, DocumentTextLine):
            raise TypeError("top is not a DocumentTextLine")

        self.text = text
        self.excess = ""
        self.top = top
        self.font = self.get_random_font(self.height)

    def get_random_font(self, font_height, index=-1):
        if index == -1:
            font_name = peak_random_resource("fonts")
            return ImageFont.truetype(font_name, int(font_height / 1.5))
        else:
            font_name = peak_resource("fonts", index=index)
            return ImageFont.truetype(font_name, int(font_height / 1.5))

    def to_image(self):
        width, height = self.get_size()
        if width is None:
            width, _ = self.font.getsize(self.text)

        image = Image.new('RGBA', (width, height))

        image_draw = ImageDraw.Draw(image)
        image_draw.text((0, 0), self.text, font=self.font, fill=(0, 0, 0, 255))

        return image

    def get_text(self):
        return self.text

    def get_text_excess(self):
        return self.excess

    def get_font(self):
        return self.font

    def count_ressource(self):
        return count_resource("fonts")

    def generate_random(self, index=-1):
        width, height = self.get_size()

        if self.top is None:
            self.text = peak_random_preloaded_text("texts")
            self.font = self.get_random_font(height, index)
        else:
            self.text = self.top.get_text_excess()
            self.font = self.top.get_font()

        self.excess = ""

        if width is not None:
            text_width, _ = self.font.getsize(self.text)
            while text_width < width:
                self.text = self.text + " " + peak_random_preloaded_text("texts")
                text_width, _ = self.font.getsize(self.text)

            while text_width > width:
                index = self.text.rfind(" ")
                self.excess = self.text[index + 1:] + " " + self.excess
                self.text = self.text[:index]
                text_width, _ = self.font.getsize(self.text)

    def get_maximum_width(self):
        w, h = self.font.getsize(self.text)
        return w

    def get_baselines(self):
        text_width, text_height = self.font.getsize(self.text)
        return [[0, text_height // 2, text_width, text_height // 2, text_height]]