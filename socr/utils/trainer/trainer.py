import os
import sys
from datetime import datetime

import numpy as np
import torch

from socr.utils.logging.logger import print_normal, print_warning, print_error, TerminalColors
from socr.utils.maths.moving_average import MovingAverage


class Trainer:
    """
    The purpose of this class is to load and train any given network with any given loss and optimizer.
    """

    def __init__(self, model, loss, optimizer, name=None, clip_gradient=None, checkpoint_userdata=None):
        """
        Create a trainer with the given models, loss and optimizer

        :param model: The models to use. It must inherit from models abstract class. :class:`socr.models.models.Model`
        :param loss: The loss to use. It must inherit from loss abstract class. :class:`socr.models.loss.loss.Loss`
        :param optimizer: The optimizer to use.
        :param name: The name of the checkpoint to load and to save. By default, it is the models name.
        :param clip_gradient: The value to clip is not none, during training.
        :param checkpoint_userdata: Complements data to save with the checkpoint.
        """
        os.makedirs('checkpoints', exist_ok=True)

        if checkpoint_userdata is None:
            checkpoint_userdata = {}
        self.original_model = model
        self.model = torch.nn.DataParallel(model)
        self.loss = loss
        self.optimizer = optimizer
        self.checkpoint_userdata = checkpoint_userdata

        if name is None:
            name = model.get_name()

        self.checkpoint_name = "checkpoints/" + name + ".pth.tar"
        self.csv_name = "checkpoints/" + name + ".csv"
        self.adaptative_optimizer = model.adaptative_learning_rate(self.optimizer)
        self.epoch = 0
        self.clip_gradient = clip_gradient
        self.start_time = None
        self.elapsed = 0.0
        self.error = None

        if os.path.exists(self.checkpoint_name):
            self.restore()
        else:
            print_warning("Can't find '" + self.checkpoint_name + "'")

    def restore(self):
        """
        Restore the checkpoint

        :return: The complement data saved with the checkpoint (given in the constructor parameters).
        """

        print_normal("Restoring the weights...")
        checkpoint = torch.load(self.checkpoint_name)
        self.epoch = checkpoint['epoch']
        self.checkpoint_userdata = checkpoint['userdata']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.adaptative_optimizer.load_state_dict(checkpoint['adaptative_optimizer'])
        self.elapsed = checkpoint['elapsed']
        return self.checkpoint_userdata

    def save(self):
        """
        Save the checkpoint
        """
        checkpoint = {
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'userdata': self.checkpoint_userdata,
            'elapsed': self.elapsed,
            'adaptative_optimizer': self.adaptative_optimizer.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_name)

    def autosave(self):
        """
        Auto-save function which is called at every epochs. It save the checkpoint in to the file autosave.pth.tar.
        """

        checkpoint = {
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'userdata': self.checkpoint_userdata,
            'elapsed': self.elapsed,
            'adaptative_optimizer': self.adaptative_optimizer.state_dict(),
         }
        torch.save(checkpoint, self.checkpoint_name + ".autosave")

    def train(self, data_set, batch_size=1, callback=None):
        """
        Train the network until the loss won't decrease.

        :param data_set: The data-set which will be used to train the network
        :param batch_size: The batch size
        :param callback: A test function to call after every epochs. The returned value from this function will be writed into the CSV as test accuracy value.
        """
        self.moving_average = MovingAverage(max(data_set.__len__() // batch_size, 1024))

        if hasattr(self.original_model, "collate"):
            loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=self.original_model.collate)
        else:
            loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=4)
        try:
            if os.path.exists(self.csv_name):
                append_write = 'a'
            else:
                append_write = 'w'

            with open(self.csv_name, append_write) as csv_file:
                while self.optimizer.state_dict()['param_groups'][0]['lr'] > 1e-7:
                    self.do_one_epoch(loader, batch_size, csv_file)
                    if callback is not None:
                        try:
                            self.error = callback()
                        except Exception as e:
                            print_error("Callback error : " + str(e))

            print_normal("Done training ! Saving...")
            self.save()

        except KeyboardInterrupt:
            while True:
                sys.stdout.write("\n\n\nDo you want to save the weight ? [yes/no]")
                i = input()
                if i == "yes":
                    sys.stdout.write("Saving... \n")
                    self.save()
                    sys.stdout.write("Done! \n")
                    break
                if i == "no":
                    break

    def do_one_epoch(self, data_loader, batch_size, csv_file, is_cuda=True):
        """
        Train for one epoch

        :param data_loader: The data-loader which will be used to train the network
        :param batch_size: The batch size
        :param csv_file: The csv file where to write the results
        """
        self.model.train()


        for i, data in enumerate(data_loader, 0):

            if self.start_time is None:
                self.start_time = datetime.now()

            inputs, labels = data
            self.optimizer.zero_grad()

            variable = torch.autograd.Variable(inputs).float()
            if is_cuda:
                variable = variable.cuda()

            outputs = self.model(variable)
            loss_value = self.loss.forward(outputs, self.loss.process_labels(labels, is_cuda=is_cuda))

            loss_value_cpu = loss_value.data.cpu().numpy()
            if loss_value_cpu < 0:
                sys.stdout.write("\nWarning : negative loss value, " + str(loss_value_cpu) + "\n")
                sys.stdout.write("With label(s) : " + str(self.loss.process_labels(labels)) + "\n")
                continue
            
            if np.isnan(loss_value_cpu):
                sys.stdout.write("\nWarning : nan loss value, " + str(loss_value_cpu) + "\n")
                sys.stdout.write("With label(s) : " + str(self.loss.process_labels(labels)) + "\n")
                continue
            
            if np.isinf(loss_value_cpu):
                sys.stdout.write("\nWarning : inf loss value, " + str(loss_value_cpu) + "\n")
                sys.stdout.write("With label(s) : " + str(self.loss.process_labels(labels)) + "\n")
                continue

            loss_value.backward()

            if self.clip_gradient is not None:
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip_gradient)

            self.optimizer.step()

            loss_value_np = float(loss_value.data.cpu().numpy())
            self.moving_average.addn(loss_value_np)

            if (i * batch_size) % 8 == 0:
                end_time = datetime.now()
                diff = end_time - self.start_time
                self.start_time = end_time
                self.elapsed = self.elapsed + diff.total_seconds()
                sys.stdout.write(TerminalColors.BOLD + '[%d, %5d] ' % (self.epoch + 1, (i * batch_size) + 1) + TerminalColors.ENDC)
                sys.stdout.write('lr: %.8f; loss: %.4f ; curr : %.4f; time : %dmn\r' % (self.optimizer.state_dict()['param_groups'][0]['lr'], self.moving_average.moving_average(), loss_value_np, self.elapsed / 60))

        self.epoch = self.epoch + 1
        self.adaptative_optimizer.step()

        epoch_s = str(self.epoch)
        elapsed_s = str(self.elapsed).replace(".", ",")
        ma_loss_s = str(self.moving_average).replace(".", ",")
        lr_s = str(self.optimizer.state_dict()['param_groups'][0]['lr']).replace(".", ",")
        error_s = "" if self.error is None else str(self.error).replace(".", ",")
        csv_file.write(epoch_s + "\t" + elapsed_s + "\t" + ma_loss_s + "\t" + lr_s + "\t" + error_s + "\n")

        self.autosave()
        sys.stdout.write("\n")
