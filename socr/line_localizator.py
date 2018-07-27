import argparse
import math
import os
import shutil
import subprocess
import sys
from configparser import ConfigParser
from random import randint

import torch
import numpy as np
from PIL import Image, ImageDraw

from socr.dataset import parse_datasets_configuration_file, DocumentGeneratorHelper
from socr.dataset.set.icdar_document_eval_set import ICDARDocumentEvalSet
from socr.models import get_model_by_name
from socr.utils.image.connected_components import save_connected_components
from socr.utils.logging.logger import print_warning, print_normal
from socr.utils.setup.build import load_default_datasets_cfg_if_not_exist
from socr.utils.setup.download import download_resources
from socr.utils.trainer.trainer import Trainer
from socr.utils.image import show_numpy_image, image_pillow_to_numpy, image_numpy_to_pillow, mIoU, \
    image_numpy_to_pillow_bw


class LineLocalizator:
    """
    This is the main class of the line localizator.
    """

    def __init__(self, model_name="dhSegment", lr=0.0001, name=None, is_cuda=True):
        """
        Creae a line localizator with the given models name

        :param model_name: The models name to use
        :param lr: The intial learning rate to use, if the training has not started yet
        """
        self.settings = ConfigParser()
        self.settings.read("settings.cfg")

        # Load the models, the loss, the optimizer and create a trainer from these three. The Trainer class will
        # automatically restore the weight if it exist.
        self.model = get_model_by_name(model_name)(self.settings.get("line", "loss_type"),
                                                   float(self.settings.get("line", "hysteresis_minimum")),
                                                   float(self.settings.get("line", "hysteresis_maximum")),
                                                   int(self.settings.get("line", "thicknesses")),
                                                   float(self.settings.get("line", "height_importance")),
                                                   float(self.settings.get("line", "exponential_decay")))
        self.loss = self.model.create_loss()
        if is_cuda:
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()
        else:
            print_warning("Using the CPU")
            self.model = self.model.cpu()
            self.loss = self.loss.cpu()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=float(self.settings.get("line", "weight_decay")))
        self.trainer = Trainer(self.model, self.loss, self.optimizer, name)

        # Parse and load all the test datasets specified into datasets.cfg
        load_default_datasets_cfg_if_not_exist()
        self.database_helper = DocumentGeneratorHelper()
        self.test_database = parse_datasets_configuration_file(self.database_helper, with_document=True, training=False,
                                                               testing=True,
                                                               args={"loss": self.loss, "transform": False})
        print_normal("Test database length : " + str(self.test_database.__len__()))

    def train(self, batch_size, overlr=None):
        """
        Train the network with the given batch size

        :param overlr: Override the current learning rate
        :param batch_size: The batch size
        """
        if overlr is not None:
            self.set_lr(overlr)

        train_database = parse_datasets_configuration_file(self.database_helper, with_document=True, training=True,
                                                           testing=False, args={"loss": self.loss, "transform": True})
        print_normal("Train database length : " + str(train_database.__len__()))
        self.trainer.train(train_database, batch_size=batch_size, callback=self.callback)

    def set_lr(self, lr):
        """
        Ovverride the current learning rate

        :param lr: The new learning rate
        """
        print_normal("Overwriting the lr to " + str(lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def eval(self):
        """
        This function need to be called before evaluting the network
        """
        self.model.eval()

    def test(self):
        """
        Test the network and return the average loss and the test data-set
        :return: The average error and the test data-set
        """
        error = 0
        test_len = min(32, len(self.test_database) - 1)

        for i in range(0, test_len):
            image, label = self.test_database.__getitem__(i)

            result = self.model(torch.autograd.Variable(image.unsqueeze(0).float().cuda()))
            segmentation = self.loss.ytrue_to_segmentation(result)

            error = error + mIoU(result, segmentation)
            sys.stdout.write("Testing..." + str(i * 100 // test_len) + "%\r")

        error = error / test_len
        print_normal("Testing...100%. Error : " + str(error) + "\n")
        return error

    def test_generator(self):
        """
        Show random image from the test data-set, and the ground-truth
        """
        for i in range(0, 4):
            image, label = self.test_database.__getitem__(randint(0, len(self.test_database) - 1))
            self.loss.show_ytrue(image.cpu().numpy(), label.cpu().numpy())

    def generateandexecute(self):
        """
        Show the result of the network in some data from test data-set
        """
        is_cuda = next(self.model.parameters()).is_cuda
        for i in range(0, 4):
            image, label = self.test_database.__getitem__(randint(0, len(self.test_database) - 1))

            if is_cuda:
                result = self.model(torch.autograd.Variable(image.unsqueeze(0).float().cuda()))[0]
            else:
                result = self.model(torch.autograd.Variable(image.unsqueeze(0).float().cpu()))[0]

            self.loss.show_ytrue(image.cpu().numpy(), result.cpu().detach().numpy())
            lines = self.loss.ytrue_to_lines(image.cpu().numpy(), result.cpu().detach().numpy())
            for line, pos in lines:
                show_numpy_image(line, invert_axes=True)

    def execute(self, image_path):
        """
        Show the result of the network on the given image
        :param image_path: The path of the image
        """
        is_cuda = next(self.model.parameters()).is_cuda

        image = Image.open(image_path).convert('RGB')

        width, height = image.size
        new_width = math.sqrt(6 * (10 ** 5) * width / height)
        new_width = int(new_width)
        new_height = height * new_width // width

        resized = image.resize((new_width, new_height), Image.ANTIALIAS)

        image = torch.from_numpy(image_pillow_to_numpy(image))
        resized = torch.from_numpy(image_pillow_to_numpy(resized))

        if is_cuda:
            result = self.model(torch.autograd.Variable(resized.unsqueeze(0).float().cuda()))[0]
        else:
            result = self.model(torch.autograd.Variable(resized.unsqueeze(0).float().cpu()))[0]

        self.loss.show_ytrue(resized.cpu().numpy(), result.cpu().detach().numpy())
        lines = self.loss.ytrue_to_lines(image.cpu().numpy(), result.cpu().detach().numpy())
        for line, pos in lines:
            show_numpy_image(line, invert_axes=True)

    def extract(self, original_image, resized_image, with_images=True):
        """
        Extract all the line from the given image

        :param image: A tensor image
        :return: The extracted line, as pillow image, and their positions
        """
        if type(original_image).__module__ == np.__name__:
            original_image = torch.from_numpy(original_image).unsqueeze(0)

        if type(resized_image).__module__ == np.__name__:
            resized_image = torch.from_numpy(resized_image).unsqueeze(0)

        is_cuda = next(self.model.parameters()).is_cuda

        image = torch.autograd.Variable(resized_image).float()

        if is_cuda:
            image = image.cuda()
        else:
            image = image.cpu()

        image = self.loss.process_labels(image)
        result = self.model(torch.autograd.Variable(image))[0]
        lines, components = self.loss.ytrue_to_lines(original_image.cpu().numpy()[0], result.cpu().detach().numpy(), with_images)

        pillow_lines = [line for line, pos in lines]
        pos = [pos for line, pos in lines]

        return pillow_lines, pos, result, components

    def output_image_bloc(self, image, lines, lwidth=5):
        """
        Draw the lines to the image

        :param image: The image
        :param lines: The lines
        :return: The new image
        """
        image = image_numpy_to_pillow(image.cpu().numpy()[0])
        image = image.convert("L").convert("RGB")
        image_drawer = ImageDraw.Draw(image)
        for i in range(0, len(lines)):
            positions = list(lines[i])
            for i in range(0, len(positions) // 2 - 1):
                image_drawer.line((positions[i * 2], positions[i * 2 + 1], positions[i * 2 + 2], positions[i * 2 + 3]),
                                  fill=(128, 0, 0), width=lwidth)

        return image

    def output_baseline(self, lines):
        """
        Output a writable string of the lines

        :param lines: The lines
        :return: The string
        """
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

    def evaluate(self, path):
        """
        Evaluate the line localizator. Output all the results to the 'results' directory.

        :param path: The path of the images, with or without associated XMLs
        """
        print_normal("Evaluating " + path)

        if not os.path.exists("results"):
            os.makedirs("results")

        data_set = ICDARDocumentEvalSet(path, self.loss)

        loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False, num_workers=1)

        count = 0

        for i, data in enumerate(loader, 0):
            resized, image, path, label = data

            percent = i * 100 // data_set.__len__()
            sys.stdout.write(str(percent) + "%... Processing \r")

            lines, positions, probsmap, components = self.extract(image, resized, with_images=False)

            self.output_image_bloc(image, positions).save("results/" + str(count) + ".jpg", "JPEG")

            save_connected_components(components, "results/" + str(count) + ".components.jpg")

            image_numpy_to_pillow_bw(probsmap[0].cpu().detach().numpy()).save("results/" + str(count) + ".probs.jpg")
            del probsmap

            image_numpy_to_pillow_bw(label[0][0].cpu().detach().numpy()).save("results/" + str(count) + ".probs.gt.jpg")

            xml_path = os.path.join(os.path.dirname(path[0]), os.path.splitext(os.path.basename(path[0]))[0] + ".xml")
            if not os.path.exists(xml_path):
                xml_path = os.path.join(os.path.dirname(path[0]),
                                        "page/" + os.path.splitext(os.path.basename(path[0]))[0] + ".xml")

            if os.path.exists(xml_path):
                shutil.copy2(xml_path, "results/" + str(count) + ".xml")
                with open("results/" + str(count) + ".txt", "w") as text_file:
                    text_file.write(self.output_baseline(positions))
            else:
                print_warning("Can't find : '" + xml_path + "'")

            count = count + 1

    def evaluate_hist(self, path):

        with open("p.txt", "w") as pfile:
            with open("r.txt", "w") as rfile:
                with open("f1.txt", "w") as f1file:

                    for i in range(0, 10 + 1):
                        for j in range(i, 10 + 1):
                            print("For " + "(" + str(i / 10) + "," + str(j / 10) + ")")
                            subprocess.run(['rm', '-R', 'results'])
                            self.loss.decoder.set_hysteresis(i / 10, j / 10)

                            self.evaluate(path)
                            result = self.run_transkribus()
                            lines = result.split("\n")
                            probs = [line.split(",") for line in lines]
                            probs = [[prob.replace(" ", "") for prob in problist] for problist in probs]

                            # new_probs = []
                            total = None

                            for k in range(0, len(probs)):
                                try:
                                    id = probs[k][3].split(".")[0]
                                    if id == "TOTAL":
                                        total = probs[k]
                                except Exception as e:
                                    pass

                            p = float(total[0])
                            r = float(total[1])
                            f1 = float(total[2])

                            pfile.write("(" + str(i / 10) + "," + str(j / 10) + "," + "%.3f" % (p) + ")")
                            rfile.write("(" + str(i / 10) + "," + str(j / 10) + "," + "%.3f" % (r) + ")")
                            f1file.write("(" + str(i / 10) + "," + str(j / 10) + "," + "%.3f" % (f1) + ")")

                    for i in range(3, 6 + 1):
                        for j in range(75, 95):
                            print("For " + "(" + str(i / 10) + "," + str(j / 10) + ")")
                            subprocess.run(['rm', '-R', 'results'])
                            self.loss.decoder.set_hysteresis(i / 10, j / 100)

                            self.evaluate(path)
                            result = self.run_transkribus()
                            lines = result.split("\n")
                            probs = [line.split(",") for line in lines]
                            probs = [[prob.replace(" ", "") for prob in problist] for problist in probs]

                            # new_probs = []
                            total = None

                            for k in range(0, len(probs)):
                                try:
                                    id = probs[k][3].split(".")[0]
                                    if id == "TOTAL":
                                        total = probs[k]
                                except Exception as e:
                                    pass


                            p = float(total[0])
                            r = float(total[1])
                            f1 = float(total[2])

                            pfile.write("(" + str(i / 10) + "," + str(j / 10) + "," + "%.3f" % (p) + ")")
                            rfile.write("(" + str(i / 10) + "," + str(j / 10) + "," + "%.3f" % (r) + ")")
                            f1file.write("(" + str(i / 10) + "," + str(j / 10) + "," + "%.3f" % (f1) + ")")

    def run_transkribus(self):
        pipe = subprocess.Popen(["sh", "evaluate.sh"], stdout=subprocess.PIPE)
        texts = pipe.communicate()
        texts = ["" if text is None else text.decode() for text in texts]
        text = "\n".join(texts)
        return text

    def callback(self):
        self.eval()
        subprocess.run(['rm', '-R', 'results'])

        config = ConfigParser()
        config.read("datasets.cfg")

        for section in config.sections():
            dict = {}
            options = config.options(section)
            for option in options:
                dict[option] = config.get(section, option)

            if dict["for"] == "Document" and "test" in dict:
                self.evaluate(dict["test"])

        result = self.run_transkribus()
        lines = result.split("\n")
        probs = [line.split(",") for line in lines]
        probs = [[prob.replace(" ", "") for prob in problist] for problist in probs]

        new_probs = []
        total = None

        for i in range(0, len(probs)):
            try:
                id = probs[i][3].split(".")[0]
                if id == "TOTAL":
                    total = probs[i]
            except Exception as e:
                pass

        print_normal("P : " + str(total[0]) + "; F : " + str(total[1]) + "; F1 : " + str(total[2]))

        for i in range(0, len(probs)):
            try:
                new_probs.append([float(probs[i][0]), float(probs[i][1]), float(probs[i][2]), probs[i][3], probs[i][4]])
            except Exception as e:
                pass

        new_probs.sort(key=lambda x: x[2])

        for i in range(0, len(new_probs)):
            id = new_probs[i][3].split(".")[0]
            if id != "TOTAL":
                for ext in [".jpg", ".probs.jpg", ".probs.gt.jpg", ".components.jpg", ".txt", ".xml"]:
                    os.rename("results/" + id + ext, 'results/%.4f%s' % (new_probs[i][2], ext))
            else:
                print(new_probs[i])

        return total[2]


def main(sysarg):
    parser = argparse.ArgumentParser(description="socr")
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--model', type=str, default="dhSegment", help="Model name")
    parser.add_argument('--execute', type=str, default=None)
    parser.add_argument('--evaluate', type=str, default=None)
    parser.add_argument('--evaluatehist', type=str, default=None)
    parser.add_argument('--generateandexecute', action='store_const', const=True, default=False)
    parser.add_argument('--test', action='store_const', const=True, default=False)
    parser.add_argument('--testgenerator', action='store_const', const=True, default=False)
    parser.add_argument('--disablecuda', action='store_const', const=True, default=False)
    parser.add_argument('--overlr', action='store_const', const=True, default=False)
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--name', type=str, default=None)
    args = parser.parse_args(sysarg)

    download_resources()
    line_recognizer = LineLocalizator(args.model, args.lr, args.name, not args.disablecuda)

    if args.testgenerator:
        line_recognizer.test_generator()
    elif args.generateandexecute:
        line_recognizer.eval()
        line_recognizer.generateandexecute()
    elif args.execute is not None:
        line_recognizer.eval()
        line_recognizer.execute(args.execute)
    elif args.evaluate is not None:
        line_recognizer.eval()
        line_recognizer.evaluate(args.evaluate)
    elif args.evaluatehist is not None:
        line_recognizer.eval()
        line_recognizer.evaluate_hist(args.evaluatehist)
    elif args.test:
        line_recognizer.eval()
        line_recognizer.callback()
    else:
        if args.overlr:
            line_recognizer.train(args.bs, args.lr)
        else:
            line_recognizer.train(args.bs)
