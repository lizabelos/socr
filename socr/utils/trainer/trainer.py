import os
import sys
from datetime import datetime

import numpy as np
import torch

from socr.nn.modules.cpu_parallel import CPUParallel
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

        is_cuda = next(model.parameters()).is_cuda
        if is_cuda:
            print_normal("Using GPU Data Parallel")
            self.model = torch.nn.DataParallel(model)
        else:
            print_warning("Using CPU")
            self.model = CPUParallel(model)
        self.loss = loss
        self.optimizer = optimizer
        self.checkpoint_userdata = checkpoint_userdata

        if name is None:
            name = model.get_name()

        self.checkpoint_name = "checkpoints/" + name + ".pth.tar"
        self.csv_name_acc = "checkpoints/" + name + ".acc.txt"
        self.csv_name_lr = "checkpoints/" + name + ".lr.txt"
        self.csv_name_loss = "checkpoints/" + name + ".loss.txt"
        self.adaptative_optimizer = model.adaptative_learning_rate(self.optimizer)
        self.epoch = 0
        self.clip_gradient = clip_gradient
        if self.clip_gradient is not None:
            print_normal("Clipping the gradient to " + str(clip_gradient))

        self.start_time = None
        self.elapsed = 0.0
        self.error = None
        self.best_error = None

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
        self.best_error = checkpoint['best_error']
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
            'best_error': self.best_error
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
            'best_error': self.best_error
         }
        torch.save(checkpoint, self.checkpoint_name + ".autosave")

    def train(self, data_set, batch_size=1, callback=None, epoch_limit=None, alternative_loss=None):
        """
        Train the network until the loss won't decrease.

        :param data_set: The data-set which will be used to train the network
        :param batch_size: The batch size
        :param callback: A test function to call after every epochs. The returned value from this function will be writed into the CSV as test accuracy value.
        """
        self.moving_average = MovingAverage(max(data_set.__len__() // batch_size, 1024))
        self.alt_moving_average = MovingAverage(max(data_set.__len__() // batch_size, 1024))

        loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=self.loss.collate)

        try:
            if os.path.exists(self.csv_name_acc):
                append_write = 'a'
            else:
                append_write = 'w'

            with open(self.csv_name_acc, append_write) as csv_acc, open(self.csv_name_loss, append_write) as csv_loss, open(self.csv_name_lr, append_write) as csv_lr:
                while self.optimizer.state_dict()['param_groups'][0]['lr'] > 1e-7:
                    if epoch_limit is not None and self.epoch > epoch_limit:
                        print_normal("Epoch " + str(epoch_limit) + "reached !")
                        break

                    self.do_one_epoch(loader, batch_size, alternative_loss)
                    if callback is not None:
                        self.error = callback()

                        if self.error is not None:
                            if self.best_error is None or self.error < self.best_error:
                                print_normal("Best score ! Saving !")
                                self.best_error = self.error
                                self.save()
                    self.write_to_file(csv_loss, csv_acc, csv_lr)

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

    def do_one_epoch(self, data_loader, batch_size, alternative_loss):
        """
        Train for one epoch

        :param data_loader: The data-loader which will be used to train the network
        :param batch_size: The batch size
        :param csv_file: The csv file where to write the results
        """
        is_cuda = next(self.model.parameters()).is_cuda
        self.model.train()


        for i, data in enumerate(data_loader, 0):

            if self.start_time is None:
                self.start_time = datetime.now()

            inputs, labels = data
            self.optimizer.zero_grad()

            variable = torch.autograd.Variable(inputs).float()
            if is_cuda:
                variable = variable.cuda()
            else:
                variable = variable.cpu()

            outputs = self.model(variable)
            loss_value = self.loss.forward(outputs, self.loss.process_labels(labels, is_cuda=is_cuda))

            loss_value_cpu = loss_value.data.cpu().numpy()
            
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradient)

            self.optimizer.step()

            loss_value_np = float(loss_value.data.cpu().numpy())
            self.moving_average.addn(loss_value_np)

            if (i * batch_size) % 32 == 0:
                if alternative_loss is not None:
                    self.alt_moving_average.addn(alternative_loss(labels, outputs))

            if (i * batch_size) % 8 == 0:
                end_time = datetime.now()
                diff = end_time - self.start_time
                self.start_time = end_time
                self.elapsed = self.elapsed + diff.total_seconds()
                sys.stdout.write(TerminalColors.BOLD + '[%d, %5d] ' % (self.epoch + 1, (i * batch_size) + 1) + TerminalColors.ENDC)
                sys.stdout.write('lr: %.8f; loss: %.4f ; aloss: %.4f ; time : %dmn\r' % (self.optimizer.state_dict()['param_groups'][0]['lr'], self.moving_average.moving_average(), self.alt_moving_average.moving_average(), self.elapsed / 60))

        self.epoch = self.epoch + 1
        # self.adaptative_optimizer.step()

        self.autosave()
        sys.stdout.write("\n")

    def write_to_file(self, csv_loss, csv_acc, csv_lr):
        epoch_s = str(self.epoch)
        ma_loss_s = str(self.moving_average.moving_average())
        error_s = "" if self.error is None else str(self.error)
        lr_s = str(self.optimizer.state_dict()['param_groups'][0]['lr'])

        csv_loss.write("(" + epoch_s + "," + ma_loss_s + ")")
        csv_acc.write("(" + epoch_s + "," + error_s + ")")
        csv_lr.write("(" + epoch_s + "," + lr_s + ")")
