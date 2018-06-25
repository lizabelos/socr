import os
import shutil

import torch

import lxml
import sys
from os import listdir
from os.path import isfile, join

from PIL import Image, ImageDraw, ImageFont
from lxml import etree
from torch.utils.data import Dataset

from socr.utils.image import image_numpy_to_pillow, image_pillow_to_numpy
from .line_localizator import LineLocalizator
from .text_recognizer import TextRecognizer


class FileDataset(Dataset):

    def __init__(self):
        self.list = []

    def recursive_list(self, path):
        if isfile(path):
            if not path.endswith(".result.jpg") and path.endswith(".jpg"):
                self.list.append(path)
        else:
            for file_name in listdir(path):
                self.recursive_list(join(path, file_name))

    def sort(self):
        self.list.sort()

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        image = Image.open(self.list[index]).convert("RGB")
        width, height = image.size

        if height > 512:
            resized = image.resize((width * 512 // height, 512), Image.ANTIALIAS)

        return image_pillow_to_numpy(resized), image_pillow_to_numpy(image), self.list[index]


class Recognizer:
    def __init__(self, line_localizator_backend=LineLocalizator, text_recognizer_backend=TextRecognizer):
        print("Loading Line Localizator...")
        self.line_localizator = line_localizator_backend()
        self.line_localizator.eval()

        print("Loading Text Recognizer...")
        self.text_recognizer = text_recognizer_backend()
        self.text_recognizer.eval()

    def recognize_lines(self, image, resized):
        lines, positions = self.line_localizator.extract(image, resized)
        return positions, lines

    def recognize_texts(self, lines):
        texts = self.text_recognizer.recognize(lines)
        return texts

    def output_image_bloc(self, image, lines):
        image = image_numpy_to_pillow(image.cpu().numpy()[0])
        image = image.convert("L").convert("RGB")
        image_drawer = ImageDraw.Draw(image)
        for i in range(0, len(lines)):
            positions = list(lines[i])
            for i in range(0, len(positions) // 2 - 1):
                image_drawer.line((positions[i * 2], positions[i * 2 + 1], positions[i * 2 + 2], positions[i * 2 + 3]), fill=(128, 0, 0))

        return image

    def output_image_text(self, image, lines, positions, texts):
        image = Image.new("RGB", image.size)
        image_drawer = ImageDraw.Draw(image)
        font_color = (255, 0, 0)
        for i in range(0, len(positions)):
            if texts[i] != "":
                x0, y0 = positions[i][0], positions[i][1]

                print((x0, y0))

                font = ImageFont.truetype("resources/displayFonts/arial.ttf", lines[i].size[1])
                # lines[i].draw_box_to_image(image, image_drawer)
                image_drawer.text((x0, y0), texts[i], font=font, fill=font_color)

        return image

    def output_baseline(self, lines):
        result = ""

        for positions in lines:
            first_time = True
            positions = list(positions)
            for i in range(0, len(positions) // 2):
                if not first_time:
                    result = result + ";"
                result = result + str(int(positions[i * 2])) + "," + str(int(positions[i * 2 + 1]))
                first_time = False
            result = result + "\n"

        return result

    def output_result(self, output_file, lines, texts):
        root = etree.Element("socr")
        for i in range(0, len(lines)):
            width, height = lines[i].get_size()
            x, y = lines[i].get_position()

            child = etree.SubElement(root, "line")

            etree.SubElement(child, "x").text = str(x)
            etree.SubElement(child, "y").text = str(y)
            etree.SubElement(child, "width").text = str(width)
            etree.SubElement(child, "height").text = str(height)

        final_str = etree.tostring(root, pretty_print=True)
        with open(texts, "w") as file:
            file.write(final_str)


def main():
    if not os.path.exists("results"):
        os.makedirs("results")

    recognizer = Recognizer()

    data_set = FileDataset()
    for path in sys.argv[1:]:
        data_set.recursive_list(path)
    data_set.sort()


    loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False, num_workers=4)
    count = 0

    for i, data in enumerate(loader, 0):
        resized, image, path = data

        percent = i * 100 // data_set.__len__()
        print(str(percent) + "%... Processing " + path[0])

        try:
            positions, lines = recognizer.recognize_lines(image, resized)
            # texts = recognizer.recognize_texts(image)

            print("Creating output image bloc to" + path[0] + ".bloc.result.jpg")
            output_bloc_image = recognizer.output_image_bloc(image, positions)
            output_bloc_image.save(path[0] + ".bloc.result.jpg", "JPEG")

            xml_path = os.path.dirname(path[0]) + "/page/" + os.path.splitext(os.path.basename(path[0]))[0] + ".xml"
            if os.path.exists(xml_path):

                shutil.copy2(xml_path, "results/" + str(count) + ".xml")

                with open("results/" + str(count) + ".txt", "w") as text_file:
                    text_file.write(recognizer.output_baseline(positions))

            else:
                print("Can't find : '" + xml_path + "'")

        except Exception as e:
            print(e)

        count = count + 1

    # for i in range(0, len(file_list)):
    #
    #     path = file_list[i]
    #     percent = i * 100 // len(file_list)
    #     print(str(percent) + "%... Processing '" + path + "'")
    #
    #     try:
    #         image = Image.open(path)
    #         width, height = image.size
    #
    #         if height > 512:
    #             print("Downsampling the image...")
    #             image = image.resize((width * 512 // height, 512), Image.ANTIALIAS)
    #
    #         texts, positions, lines = recognizer.recognize(image)
    #
    #         output_bloc_image = recognizer.output_image_bloc(image, positions, texts)
    #         output_bloc_image.save(path + ".bloc.result.jpg", "JPEG")
    #
    #         output_line_image = recognizer.output_image_text(image, lines, positions, texts)
    #         output_line_image.save(path + ".line.result.jpg", "JPEG")
    #
    #         with open(path + ".txt", "w") as text_file:
    #             text_file.write(recognizer.output_baseline(positions))
    #
    #     except Exception as e:
    #         print(e)


if __name__ == '__main__':
    main()
