from random import randint

from PIL import ImageDraw

from socr.dataset.generator.generator import Generator


class DocumentGenerator(Generator):

    def __init__(self, helper):
        self.helper = helper

    def generate(self, index):
        width = randint(200, 400)
        height = randint(400, 600)
        y = 0

        image = self.helper.get_random_background(width, height)
        image_draw = ImageDraw.Draw(image)

        regions = []

        while y < height:
            y = y + randint(0, 10)

            text = self.helper.get_random_text()
            text_start = randint(0, width // 2)
            font_height = randint(10, 30)
            font_color = (randint(0, 128), randint(0, 128), randint(0, 128))

            if y + font_height > height:
                break

            while True:
                try:
                    font = self.helper.get_font(index, font_height)
                    font_width, font_height = font.getsize(text)
                    break
                except OSError:
                    #print("Warning : execution context too long ! Continuing...")
                    index = 0

            while font_width + text_start > width:
                text = text[0:randint(len(text) // 2, len(text) - 1)]
                font_width = font.getsize(text)[0]

            image_draw.text((text_start, y), text, font=font, fill=font_color)

            top_left  = ((text_start             ) / width, (y              ) / height)
            top_right = ((text_start + font_width) / width, (y              ) / height)
            bot_rigth = ((text_start + font_width) / width, (y + font_height) / height)
            bot_left =  ((text_start             ) / width, (y + font_height) / height)

            top_left = self.wrap_01(top_left)
            top_right = self.wrap_01(top_right)
            bot_rigth = self.wrap_01(bot_rigth)
            bot_left = self.wrap_01(bot_left)

            regions.append([top_left, top_right, bot_rigth, bot_left])

            y = y + font_height

        self.helper.add_random_phantom_patterns(image)
        return image, regions

    def wrap_01(self, pos):
        x, y = pos
        x = min(1, max(0, x))
        y = min(1, max(0, y))
        return x, y
