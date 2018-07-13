import math
import sys
import time

import torch

from socr.dataset.set.corpus import Corpus
from socr.models.lm import LM
from socr.utils import Trainer
from socr.utils.maths.moving_average import MovingAverage


class TextGenerator:

    def __init__(self, batch_size=32):
        self.bptt = 35
        self.batch_size = batch_size

        self.corpus = Corpus("./resources/corpus")
        self.model = LM("GRU", len(self.corpus.dictionary), 200, 200, 2).cuda()
        self.loss = torch.nn.CrossEntropyLoss().cuda()

        self.train_data = self.batchify(self.corpus.train, batch_size).cuda()
        self.val_data = self.batchify(self.corpus.valid, batch_size).cuda()
        self.test_data = self.batchify(self.corpus.test, batch_size).cuda()

        self.ma = MovingAverage(64)

    def train(self, lr):
        best_val_loss = None

        try:
            for epoch in range(1, 50):
                epoch_start_time = time.time()
                self.one_epoch(epoch, lr)
                val_loss = self.evaluate(self.val_data)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                 val_loss, math.exp(val_loss)))
                print('-' * 89)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    # with open(args.save, 'wb') as f:
                    #    torch.save(model, f)
                    best_val_loss = val_loss
                else:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    lr /= 4.0
        except KeyboardInterrupt:
            print('-' * 89)
        print('Exiting from training early')

    def one_epoch(self, epoch, lr):
        # Turn on training mode which enables dropout.
        self.model.train()

        start_time = time.time()
        ntokens = len(self.corpus.dictionary)
        hidden = self.model.init_hidden(self.batch_size)
        for batch, i in enumerate(range(0, self.train_data.size(0) - 1, self.bptt)):
            data, targets = self.get_batch(self.train_data, i)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = self.repackage_hidden(hidden)
            self.model.zero_grad()
            output, hidden = self.model(data, hidden)
            loss = self.loss(output.view(-1, ntokens), targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.25)
            for p in self.model.parameters():
                p.data.add_(-lr, p.grad.data)

            self.ma.addn(loss.item())

            if batch % 1 == 0:
                cur_loss = self.ma.moving_average()
                elapsed = time.time() - start_time
                sys.stdout.write('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}\r'.format(
                    epoch, batch, len(self.train_data) // self.bptt, lr, elapsed * 1000, cur_loss, math.exp(cur_loss)))

            start_time = time.time()

    def batchify(self, data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()

        return data

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def get_batch(self, source, i):
        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].view(-1)
        return data, target

    def evaluate(self, data_source):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        total_loss = 0.
        ntokens = len(self.corpus.dictionary)
        hidden = self.model.init_hidden(1)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, self.bptt):
                data, targets = self.get_batch(data_source, i)
                output, hidden = self.model(data, hidden)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * self.criterion(output_flat, targets).item()
                hidden = self.repackage_hidden(hidden)
        return total_loss / len(data_source)


def main(sysarg):
    textGenerator = TextGenerator()
    textGenerator.train(20)
