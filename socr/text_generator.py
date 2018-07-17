import math
import os
import sys
import time

import torch

from socr.dataset.set.corpus import Corpus
from socr.models.lm import LM
from socr.utils.logging.logger import print_warning
from socr.utils.maths.moving_average import MovingAverage


class TextGenerator:

    def __init__(self, batch_size=32):
        self.bptt = 35
        self.batch_size = batch_size

        self.corpus = Corpus("./resources/corpus")
        self.model = LM(len(self.corpus.dictionary)).cuda()
        self.loss = torch.nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.adaptative_optimizer = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.90)

        self.train_data = self.batchify(self.corpus.train, batch_size).cuda()
        self.val_data = self.batchify(self.corpus.valid, 1).cuda()
        self.test_data = self.batchify(self.corpus.test, 1).cuda()

        self.ma = MovingAverage(1024)

        self.checkpoint_name = "checkpoints/LM.pth.tar"
        if os.path.exists(self.checkpoint_name):
            checkpoint = torch.load(self.checkpoint_name)
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            print_warning("Can't find '" + self.checkpoint_name + "'")

    def train(self):
        try:
            for epoch in range(1, 100):
                epoch_start_time = time.time()
                self.one_epoch(epoch)
                val_loss = self.evaluate(self.val_data)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                 val_loss, math.exp(val_loss)))
                print('-' * 89)

                checkpoint = {
                    'state_dict': self.model.state_dict()
                }
                torch.save(checkpoint, self.checkpoint_name)

        except KeyboardInterrupt:
            print('-' * 89)
        print('Exiting from training early')

    def one_epoch(self, epoch):
        # Turn on training mode which enables dropout.
        self.model.train()

        start_time = time.time()
        ntokens = len(self.corpus.dictionary)
        hidden = None
        for batch, i in enumerate(range(0, self.train_data.size(0) - 1, self.bptt)):
            data, targets = self.get_batch(self.train_data, i)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            if hidden is not None:
                hidden = self.repackage_hidden(hidden)
            self.optimizer.zero_grad()
            output, hidden = self.model(data, hidden)
            loss = self.loss(output.view(-1, ntokens), targets)
            loss.backward()

            self.optimizer.step()

            self.ma.addn(loss.item())

            if batch % 1 == 0:
                cur_loss = self.ma.moving_average()
                elapsed = time.time() - start_time
                sys.stdout.write('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}\r'.format(
                    epoch, batch, len(self.train_data) // self.bptt, self.optimizer.state_dict()['param_groups'][0]['lr'], elapsed * 1000, cur_loss, math.exp(cur_loss)))

            start_time = time.time()

        self.adaptative_optimizer.step()

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
        hidden = None
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, self.bptt):
                data, targets = self.get_batch(data_source, i)
                output, hidden = self.model(data, hidden)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * self.loss(output_flat, targets).item()
                hidden = self.repackage_hidden(hidden)
        return total_loss / len(data_source)

    def get_ngram_prob(self, words, last_word):
        self.model.eval()

        words = [w.lower() for w in words]
        words = [self.corpus.dictionary.word_to_id(w) for w in words]
        words = torch.autograd.Variable(torch.LongTensor(words), requires_grad=False).cuda().unsqueeze(0)

        last_word = last_word.lower()
        last_word = self.corpus.dictionary.word_to_id(last_word)

        output, hidden = self.model(words)

        output = torch.nn.functional.softmax(output, dim=2)

        word_weights = output.cpu().exp().detach().numpy()
        result = word_weights[0][len(words) - 1][last_word]

        return result

    def in_dictionnary(self, word):
        pass

    def get_bigram_prob(self, w1, w2):
        self.model.eval()

        w1 = w1.lower()
        w2 = w2.lower()

        id1 = self.corpus.dictionary.word_to_id(w1)
        id2 = self.corpus.dictionary.word_to_id(w2)

        input = torch.randint(len(self.corpus.dictionary), (1, 1), dtype=torch.long).cuda()
        input.fill_(id1)
        # hidden = self.model.init_hidden(1)
        # output, hidden = self.model(input, hidden)
        output, hidden = self.model(input)

        word_weights = output.cpu().detach().numpy()
        result = word_weights[0][0][id2]

        if result < 0:
            result = 0

        return result


def main(sysarg):
    textGenerator = TextGenerator()
    textGenerator.train()
