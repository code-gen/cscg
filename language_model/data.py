import os
from io import open
import torch

class Dictionary:
    def __init__(self):
        self.word2idx = {'<unk>': 0}
        self.idx2word = ['<unk>']
        self.wordcnt  = {'<unk>': 1}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.wordcnt[word] = 1
        else:
            self.wordcnt[word] = self.wordcnt[word] + 1
            
        return self.word2idx[word]

    def getid(self, word, thresh=1):
        if (word not in self.word2idx) or (self.wordcnt[word] < thresh):
            return self.word2idx['<unk>']
        
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test  = self.tokenize(os.path.join(path, 'test.txt'))
        
    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                if len(line.strip().split()) == 0:
                    continue
                words = ['<s>'] + line.strip().split() + ['</s>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                if len(line.strip().split()) == 0:
                    continue
                words = ['<s>'] + line.strip().split() + ['</s>']
                for word in words:
                    ids[token] = self.dictionary.getid(word)
                    token += 1

        return ids