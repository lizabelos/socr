from random import randint

from PIL import ImageDraw, Image

from socr.dataset.generator.generator import Generator


class LineGenerator(Generator):

    def __init__(self, helper, labels):
        self.helper = helper
        self.lst = labels

    def generate(self, index):
        text_start = randint(0, 5)
        text_end = randint(0, 50)
        text_top = randint(0, 5)
        text_bottom = randint(0, 5)

        text = self.helper.get_random_text()

        font_height = randint(8, 30)
        font = self.helper.get_font(index, font_height)

        total_width = 0
        lefts = []
        widths = []
        max_height = 0

        for i in range(0, len(text)):
            while True:
                try:
                    font_width, font_height = font.getsize(text[i])
                    break
                except OSError:
                    print("Warning : execution context too long ! Continuing...")
                    font = self.helper.get_font(0, font_height)

            lefts.append(text_start + total_width)
            widths.append(font_width)
            total_width = total_width + font_width
            max_height = max(max_height, font_height)

        image_width = text_start + text_end + total_width
        image_height = text_top + text_bottom + font_height
        image = Image.new('RGBA', (image_width, image_height))
        image_draw = ImageDraw.Draw(image)

        for i in range(0, len(text)):
            image_draw.text((lefts[i], text_top), text[i], font=font, fill=(randint(0, 128), randint(0, 128), randint(0, 128)))
            widths[i] = widths[i] / image_width
            lefts[i] = lefts[i] / image_width

        image = image.rotate(randint(-3, 3), expand=True, resample=Image.BICUBIC)

        image_width, image_height = image.size
        background = self.helper.get_random_background(image_width, image_height)
        background.paste(image, (0,0), image)

        return background, "".join(text)

    def get_labels(self):
        return self.lst