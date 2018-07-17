import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        word = word.lower()
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def word_to_id(self, word):
        if word not in self.word2idx:
            return 1
        else:
            return self.word2idx[word] + 2

    def id_to_word(self, id):
        if id == 0:
            return ''
        if id == 1:
            return ''
        return self.idx2word[id - 2]

    def __len__(self):
        return len(self.idx2word) + 2


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.words = self.add_to_dict(os.path.join(path, 'words.txt'))
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def add_to_dict(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

    def tokenize(self, path):
        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)

        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word_to_id(word.lower())
                    token += 1

        return ids
