from random import randint

from PIL import Image, ImageOps, ImageEnhance

from scribbler.document import AbstractDocument
from scribbler.resources.resources_helper import get_iam_handwriting_line_dataset_instance


class DocumentHandwrittingLine(AbstractDocument):

    def __init__(self, width=None, height=None, parent=None, top=None, center=True, text=""):
        super().__init__(width, height, parent)

        if not isinstance(text, str):
            raise TypeError("text is not a str")
        if top is not None and not isinstance(top, DocumentHandwrittingLine):
            raise TypeError("top is not a DocumentTextLine")

        self.text = text
        self.excess = ""
        self.top = top
        self.dataset = get_iam_handwriting_line_dataset_instance()
        self.indice = 0
        self.center = center
        self.hwimage, self.text = self.dataset.get(self.indice)

        enhancer = ImageEnhance.Contrast(self.hwimage)
        self.hwimage = enhancer.enhance(2.0)


    def to_image(self):
        width, height = self.get_size()
        if width is None:
            width = self.get_maximum_width()
        image = Image.new('RGBA', (width, height))

        x = 0
        if self.center:
            txt_width, _ = self.hwimage.size
            x = (width - txt_width) // 2

        image.paste(self.hwimage, (x, 0), ImageOps.invert(self.hwimage.convert("L")))
        return image

    def get_text(self):
        return self.text

    def get_text_excess(self):
        return self.excess

    def get_font(self):
        return self.font

    def count_ressource(self):
        return self.dataset.count()

    def generate_random(self, index=-1):
        if index != -1:
            self.hwimage, self.text = self.dataset.get(index)
            hwwidth, hwheight = self.hwimage.size
            width, height = self.get_size()
            self.hwimage = self.hwimage.resize((hwwidth * height // hwheight, height), Image.ANTIALIAS).convert("RGB")
            enhancer = ImageEnhance.Contrast(self.hwimage)
            self.hwimage = enhancer.enhance(2.0)
            return

        while True:
            self.indice = randint(0, self.dataset.count())
            self.hwimage, self.text = self.dataset.get(self.indice)
            hwwidth, hwheight = self.hwimage.size
            width, height = self.get_size()

            if hwwidth * height // hwheight > width:
                continue

            self.hwimage = self.hwimage.resize((hwwidth * height // hwheight, height), Image.ANTIALIAS).convert("RGB")
            enhancer = ImageEnhance.Contrast(self.hwimage)
            self.hwimage = enhancer.enhance(2.0)
            return

    def get_maximum_width(self):
        hwwidth, hwheight = self.hwimage.size
        return hwwidth

    def get_baselines(self):
        width, height = self.get_size()

        x = 0
        if self.center:
            txt_width, _ = self.hwimage.size
            x = (width - txt_width) // 2

        return [[x, self.height // 2, x + self.hwimage.size[0], self.height // 2, self.height]]
