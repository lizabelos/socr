import os
import shutil

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

from socr.dataset.set.file_dataset import FileDataset
from .line_localizator import LineLocalizator
from .text_recognizer import TextRecognizer


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

    def output_image_text(self, image, lines, positions, texts):
        image = Image.new("RGB", (image.size()[3], image.size()[2]))
        image_drawer = ImageDraw.Draw(image)
        font_color = (255, 0, 0)
        for i in range(0, len(positions)):
            if texts[i] != "":
                x0, y0 = positions[i][0], positions[i][1]

                for j in range(0, len(positions[i]) // 2):
                    if positions[i][j * 2] < x0:
                        x0 = positions[i][j * 2]
                    if positions[i][j * 2 + 1] < y0:
                        y0 = positions[i][j * 2 + 1]

                font = ImageFont.truetype("resources/displayFonts/arial.ttf", 24)
                # lines[i].draw_box_to_image(image, image_drawer)
                image_drawer.text((x0, y0), texts[i], font=font, fill=font_color)

        return image


def main(sysarg):
    if not os.path.exists("results"):
        os.makedirs("results")

    recognizer = Recognizer()

    data_set = FileDataset()
    for path in sysarg:
        data_set.recursive_list(path)
    data_set.sort()

    print(str(data_set.__len__()) + " files")

    loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False, num_workers=4)
    count = 0

    for i, data in enumerate(loader, 0):
        resized, image, path = data

        percent = i * 100 // data_set.__len__()
        print(str(percent) + "%... Processing " + path[0])

        positions, lines = recognizer.recognize_lines(image, resized)
        texts = recognizer.recognize_texts(lines)

        print("Creating output image bloc to" + path[0] + ".bloc.result.jpg")
        output_bloc_image = recognizer.line_localizator.output_image_bloc(image, positions, lwidth=1)
        output_bloc_image.save("results/" + str(count) + ".bloc.jpg", "JPEG")

        output_line_image = recognizer.output_image_text(image, lines, positions, texts)
        output_line_image.save("results/" + str(count) + ".line.jpg", "JPEG")

        xml_path = os.path.join(os.path.dirname(path[0]), os.path.splitext(os.path.basename(path[0]))[0] + ".xml")
        if os.path.exists(xml_path):

            shutil.copy2(xml_path, "results/" + str(count) + ".xml")

            with open("results/" + str(count) + ".txt", "w") as text_file:
                text_file.write(recognizer.line_localizator.output_baseline(positions))

        else:
            print("Can't find : '" + xml_path + "'")



        count = count + 1