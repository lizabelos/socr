import argparse
import sys
from random import randint

import torch
from PIL import Image

from socr.dataset import parse_datasets_configuration_file, DocumentGeneratorHelper
from socr.models import get_model_by_name
from socr.utils.setup.download import download_resources
from socr.utils.trainer.trainer import Trainer
from socr.utils.image import show_numpy_image, image_pillow_to_numpy, image_numpy_to_pillow, mIoU


class LineLocalizator:
    """
    This is the main class of the line localizator.
    """

    def __init__(self, model_name="XHeightResnetModel", lr=0.0001, name=None, is_cuda=True):
        """
        Creae a line localizator with the given models name

        :param model_name: The models name to use
        :param lr: The intial learning rate to use, if the training has not started yet
        """

        # Load the models, the loss, the optimizer and create a trainer from these three. The Trainer class will
        # automatically restore the weight if it exist.
        self.model = get_model_by_name(model_name)()
        self.loss = self.model.create_loss()
        if is_cuda:
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.trainer = Trainer(self.model, self.loss, self.optimizer, name)

        # Parse and load all the test datasets specified into datasets.cfg
        self.database_helper = DocumentGeneratorHelper()
        self.test_database = parse_datasets_configuration_file(self.database_helper, with_document=True, training=True, testing=True, args={"loss":self.loss})
        print("Test database length : " + str(self.test_database.__len__()))

    def train(self, batch_size):
        """
        Train the network with the given batch size

        :param batch_size: The batch size
        """
        train_database = parse_datasets_configuration_file(self.database_helper, with_document=True, training=True, testing=False, args={"loss": self.loss})
        print("Training database length : " + str(train_database.__len__()))
        self.trainer.train(train_database, batch_size=batch_size)

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
        sys.stdout.write("Testing...100%. Error : " + str(error) + "\n")
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
        for i in range(0, 4):
            image, label = self.test_database.__getitem__(randint(0, len(self.test_database) - 1))
            result = self.model(torch.autograd.Variable(image.unsqueeze(0).float().cuda()))[0]
            self.loss.show_ytrue(image.cpu().numpy(), result.cpu().detach().numpy())
            lines = self.loss.ytrue_to_lines(image.cpu().numpy(), result.cpu().detach().numpy())
            for line, pos in lines:
                show_numpy_image(line, invert_axes=True)

    def execute(self, image_path):
        """
        Show the result of the network on the given image
        :param image_path: The path of the image
        """
        image = Image.open(image_path).convert('RGB')
        image = torch.from_numpy(image_pillow_to_numpy(image))
        result = self.model(torch.autograd.Variable(image.unsqueeze(0).float().cuda()))[0]
        self.loss.show_ytrue(image.cpu().numpy(), result.cpu().detach().numpy())
        lines = self.loss.ytrue_to_lines(image.cpu().numpy(), result.cpu().detach().numpy())
        for line, pos in lines:
            show_numpy_image(line, invert_axes=True)

    def extract(self, original_image, resized_image):
        """
        Extract all the line from the given image

        :param image: A tensor image
        :return: The extracted line, as pillow image, and their positions
        """
        image = torch.autograd.Variable(resized_image).float()
        image = image.cuda()

        image = self.loss.process_labels(image)
        result = self.model(torch.autograd.Variable(image))[0]
        lines = self.loss.ytrue_to_lines(original_image.cpu().numpy()[0], result.cpu().detach().numpy())

        pillow_lines = [line for line, pos in lines]
        pos = [pos for line, pos in lines]
        return pillow_lines, pos


def main():
    parser = argparse.ArgumentParser(description="socr")
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--model', type=str, default="XHeightResnetModel", help="Model name")
    parser.add_argument('--execute', type=str, default=None)
    parser.add_argument('--generateandexecute', action='store_const', const=True, default=False)
    parser.add_argument('--testgenerator', action='store_const', const=True, default=False)
    parser.add_argument('--disablecuda', action='store_const', const=True, default=False)
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--name', type=str, default=None)
    args = parser.parse_args()

    line_recognizer = LineLocalizator(args.model, args.lr, args.name, not args.disablecuda)

    if args.testgenerator:
        download_resources()
        line_recognizer.test_generator()
    elif args.generateandexecute:
        download_resources()
        line_recognizer.eval()
        line_recognizer.generateandexecute()
    elif args.execute is not None:
        line_recognizer.eval()
        line_recognizer.execute(args.execute)
    else:
        download_resources()
        line_recognizer.train(args.bs)


if __name__ == '__main__':
    main()
