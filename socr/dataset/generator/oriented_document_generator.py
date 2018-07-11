import math
from math import cos, sin
from random import randint, uniform

from PIL import ImageDraw, Image

from socr.dataset.generator.generator import Generator


class OrientedDocumentGenerator(Generator):

    def __init__(self, helper):
        self.helper = helper

    def rotate_point(self, point, center, angle):
        x, y = point
        cx, cy = center

        x = x - cx
        y = y - cy

        cos_angle = cos(angle * 0.0174533)
        sin_angle = sin(angle * 0.0174533)

        new_x = x * cos_angle - y * sin_angle
        new_y = y * cos_angle + x * sin_angle

        return new_x + cx, new_y + cy

    def generate_text_line(self, text, font, font_color):
        width, height = font.getsize(text)

        image = Image.new('RGBA', (width, height))
        image_draw = ImageDraw.Draw(image)
        image_draw.text((0, 0), text, font=font, fill=font_color)

        imageBox = image.getbbox()
        image = image.crop(imageBox)

        width, height = image.size

        rotation = randint(-10, 10)
        image = image.rotate(rotation, expand=True, resample=Image.BICUBIC)

        x0, y0 = 0, height / 2
        x1, y1 = width, height / 2

        x0, y0 = self.rotate_point((x0, y0), (width // 2, height // 2), -rotation)
        x1, y1 = self.rotate_point((x1, y1), (width // 2, height // 2), -rotation)

        new_width, new_height = image.size

        x0 = x0 + (new_width - width) / 2
        x1 = x1 + (new_width - width) / 2
        y0 = y0 + (new_height - height) / 2
        y1 = y1 + (new_height - height) / 2

        return image, [x0, y0, x1, y1, height]

    def generate(self, index):
        width = randint(200, 400)
        height = randint(500, 700)

        new_width = math.sqrt(6 * (10 ** 5) * width / height)
        new_width = new_width * uniform(0.8, 1.2)
        new_width = int(new_width)
        new_height = height * new_width // width

        width = new_width
        height = new_height

        y = 0

        image = self.helper.get_random_background(width, height)

        base_lines = []

        while y < height:
            y = y + randint(0, 10)

            text = self.helper.get_random_text()
            text_start = randint(0, width // 2)
            font_height = randint(10, 30)
            font_color = (randint(0, 128), randint(0, 128), randint(0, 128))

            while True:
                try:
                    font = self.helper.get_font(index, font_height)
                    font_width, font_height = font.getsize(text)
                    break
                except OSError:
                    #print("Warning : execution context too long ! Continuing...")
                    index = 0

            text_image, base_line = self.generate_text_line(text, font, font_color)
            image.paste(text_image, (text_start, y), text_image)

            x0, y0, x1, y1, line_height = base_line
            x0 = int(x0 + text_start)
            x1 = int(x1 + text_start)
            y0 = int(y0 + y)
            y1 = int(y1 + y)
            base_lines.append([x0, y0, x1, y1, line_height])

            y = y + line_height

        self.helper.add_random_phantom_patterns(image)
        return image, base_lines

