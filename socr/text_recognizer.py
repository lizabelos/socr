import argparse
import string
import sys
from random import randint

import numpy as np
import torch
from PIL import Image

from socr.utils.setup.download import download_resources
from socr.utils.rating.word_error_rate import levenshtein
from socr.dataset import parse_datasets_configuration_file, LineGeneratedSet
from socr.dataset.generator.character_generator import CharacterGenerator
from socr.dataset.generator.document_generator_helper import DocumentGeneratorHelper
from socr.models import get_model_by_name, get_optimizer_by_name
from socr.utils.trainer.trainer import Trainer
from socr.utils.image import image_pillow_to_numpy, show_pytorch_image


class TextRecognizer:

    def __init__(self, model_name="DilatationGruNetwork", optimizer_name="Adam", lr=0.001, name=None, is_cuda=True):
        with open("resources/characters.txt", "r") as content_file:
            self.labels = content_file.read() + " "

        self.document_helper = DocumentGeneratorHelper()

        self.model = get_model_by_name(model_name)(self.labels)
        self.loss = self.model.create_loss()
        if is_cuda:
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()

        self.optimizer = get_optimizer_by_name(optimizer_name)(self.model.parameters(), lr=lr)
        self.trainer = Trainer(self.model, self.loss, self.optimizer, name)

        self.database_helper = DocumentGeneratorHelper()
        self.test_database = parse_datasets_configuration_file(self.database_helper, with_line=True, training=False, testing=True, args={"height": self.model.get_input_image_height(), "labels": self.labels, "transform":True})

        print("Test database length : " + str(self.test_database.__len__()))

    def train(self, overlr=None):
        if overlr is not None:
            self.set_lr(overlr)

        train_database = parse_datasets_configuration_file(self.database_helper, with_line=True, training=True, testing=False,
                                                           args={"height": self.model.get_input_image_height(),
                                                                 "labels": self.labels}
                                                           )
        self.trainer.train(train_database, callback=lambda: self.trainer_callback())

    def trainer_callback(self):
        self.model.eval()
        return self.test(limit=32)

    def eval(self):
        self.model.eval()

    def test(self, limit=32):
        loader = torch.utils.data.DataLoader(self.test_database, batch_size=1, shuffle=False, num_workers=4)

        test_len = len(self.test_database)
        if limit is not None:
            test_len = min(limit, test_len)

        wer_s, wer_i, wer_d, wer_n = 0, 0, 0, 0
        cer_s, cer_i, cer_d, cer_n = 0, 0, 0, 0

        sen_err = 0
        count = 0


        for i, data in enumerate(loader, 0):
            image, label = self.test_database.__getitem__(i)

            result = self.model(torch.autograd.Variable(image.unsqueeze(0).float().cuda()))
            text = self.loss.ytrue_to_lines(result)

            # update CER statistics
            _, (s, i, d) = levenshtein(label, text)
            cer_s += s
            cer_i += i
            cer_d += d
            cer_n += len(label)
            # update WER statistics
            _, (s, i, d) = levenshtein(label.split(), text.split())
            wer_s += s
            wer_i += i
            wer_d += d
            wer_n += len(label.split())
            # update SER statistics
            if s + i + d > 0:
                sen_err += 1

            count = count + 1

            sys.stdout.write("Testing..." + str(count * 100 // test_len) + "%\r")

            if count == test_len - 1:
                break

        cer = (100.0 * (cer_s + cer_i + cer_d)) / cer_n
        wer = (100.0 * (wer_s + wer_i + wer_d)) / wer_n
        ser = (100.0 * sen_err) / count

        sys.stdout.write("Testing...100%. WER : " + str(wer) + "; CER : " + str(cer) + "; SER : " + str(ser) + "\n")
        return wer

    def generateandexecute(self, onlyhand=False):
        while True:
            if not onlyhand:
                image, label = self.test_database.__getitem__(randint(0, len(self.test_database) - 1))
            else:
                image, label = self.test_iam.__getitem__(randint(0, len(self.test_iam) - 1))

            result = self.model(torch.autograd.Variable(image.unsqueeze(0).float().cuda()))
            text = self.loss.ytrue_to_lines(result)

            show_pytorch_image(image)

    def recognize(self, images):

        height = self.model.get_input_image_height()
        texts = []

        for image in images:
            image_width, image_height = image.size
            image = image.resize((image_width * height // image_height, height), Image.ANTIALIAS)
            image = image_pillow_to_numpy(image)

            result = self.model(torch.autograd.Variable(torch.from_numpy(np.expand_dims(image, axis=0))).float().cuda())
            text = self.loss.ytrue_to_lines(result)

            texts.append(text)

        return texts

    def recognize_files(self, files):
        self.eval()
        result = []
        for file in files:
            image = Image.open(file)
            result.append(self.recognize(image)[0])
        return result

    def set_lr(self, lr):
        print("Overwriting the lr to " + str(lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def test_line_generator():
    with open("../resources/characters.txt", "r") as content_file:
        labels = content_file.read() + " "

    document_helper = DocumentGeneratorHelper()

    all = LineGeneratedSet(document_helper, labels, None, 48)

    # iam1 = IAMHandwritingLineDatabase(document_helper, "/space_sde/tbelos/dataset/iam-line/train", 64)
    # iam2 = IAMOneLineHandwritingDatabase(document_helper, "/space_sde/tbelos/dataset/iam-one-line/train", 64)
    # iam = MergedSet(iam1, iam2)

    # all = IAMWashington(document_helper, "/space_sde/tbelos/dataset/iam-washington", 64)
    # all = ICDARLineSet(document_helper, "/space_sde/tbelos/dataset/icdar/2017/Train-A", 64)

    for i in range(0, 4):
        image, line = all.__getitem__(randint(0, all.__len__() - 1))

        print("LINE : " + str(line))
        show_pytorch_image(image)


def test_character_generator(labels=string.digits + string.ascii_letters + ".!?, "):
    character_generator = CharacterGenerator(labels)
    image, character = character_generator.generate()
    image.show()
    print(character)


def main():
    parser = argparse.ArgumentParser(description="SOCR Text Recognizer")
    parser.add_argument('--model', type=str, default="DilatationGruNetwork", help="Model name")
    parser.add_argument('--optimizer', type=str, default="Adam", help="SGD, RMSProp, Adam")
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--overlr', action='store_const', const=True, default=False)
    parser.add_argument('--disablecuda', action='store_const', const=True, default=False)
    parser.add_argument('--generateandexecute', action='store_const', const=True, default=False)
    parser.add_argument('--onlyhand', action='store_const', const=True, default=False)
    parser.add_argument('--testlinegenerator', action='store_const', const=True, default=False,
                        help='Test the line generator')
    parser.add_argument('--testchargenerator', action='store_const', const=True, default=False,
                        help='Test the char generator')
    parser.add_argument('--test', action='store_const', const=True, default=False,
                        help='Test the char generator')
    args = parser.parse_args()

    if args.testlinegenerator:
        download_resources()
        test_line_generator()
    elif args.testchargenerator:
        download_resources()
        test_character_generator()
    elif args.generateandexecute:
        download_resources()
        line_ctc = TextRecognizer(args.model, args.optimizer, args.lr, args.name, not args.disablecuda)
        line_ctc.eval()
        line_ctc.generateandexecute(args.onlyhand)
    elif args.test:
        download_resources()
        line_ctc = TextRecognizer(args.model, args.optimizer, args.lr, args.name, not args.disablecuda)
        line_ctc.eval()
        line_ctc.test()
    else:
        download_resources()
        line_ctc = TextRecognizer(args.model, args.optimizer, args.lr, args.name, not args.disablecuda)
        if args.overlr:
            line_ctc.train(args.lr)
        else:
            line_ctc.train()


if __name__ == '__main__':
    main()
