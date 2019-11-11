from collections import defaultdict, Counter

import torch
import pandas as pd
import en_core_web_sm

from utils import load_pt_glove, build_embedding_matrix

nlp = en_core_web_sm.load()
    

class Lang:
    reserved_tokens = ['<pad>', '<unk>', '<s>', '</s>']
    
    def __init__(self, name):
        self.name = name
        
        self.word2index = defaultdict(lambda: self.reserved_tokens.index('<unk>'))
        self.index2word = defaultdict(lambda: '<unk>')
        
        for i, w in enumerate(self.reserved_tokens):
            self.word2index[w] = i
            self.index2word[i] = w
                                                  
        self.word2count = Counter()
        self.n_words    = len(self.reserved_tokens)
        self.emb_matrix = None
        
        self.__tokenize = lambda s: [tok.text for tok in nlp.tokenizer(s)]
    
    def __getitem__(self, item):
        if isinstance(item, str):
            return {'index': self.word2index[item],
                    'count': self.word2count[item]}
            
        if isinstance(item, int):
            return {'word': self.index2word[item],
                    'count': self.word2count[self.index2word[item]]}
            
        return None
    
    def __len__(self):
        n1 = len(self.word2index)
        n2 = len(self.index2word)
        n3 = len(self.word2count)
        assert n1 == n2 == (n3 + len(self.reserved_tokens))
        return n1
    
    def add_sentence(self, sentence, tokenize='default'):
        if tokenize == 'default':
            tokenize = self.__tokenize
            
        for word in tokenize(Preprocess.clean_text(sentence)):
            self.add_word(word)
            
    def add_word(self, word: str):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
                    
    def normalize_vocab(self, min_freq):
        items = list(self.word2count.items())
        
        for word, count in items:
            if count >= min_freq:
                continue
                
            i = self.word2index[word]    
            self.word2count.pop(word, None)
            self.word2index.pop(word, None)
            self.index2word.pop(i, None)
            self.n_words -= 1
                  
    def build_emb_matrix(self, emb_file: str):
        # TODO: don't hardcode glove loader
        emb_dict = load_pt_glove(emb_file)
        self.emb_matrix = build_embedding_matrix(emb_dict, 
                                                 w_idx=self.word2index,
                                                 len_voc=self.n_words,
                                                 init='zeros')
          
    def numericalize(self, sentence, pad_mode=None, maxlen=-1) -> ([str], [int]):
        tokens = []
        seq = self.__tokenize(Preprocess.clean_text(sentence))
        
        if pad_mode is not None:
            pad = ['<pad>'] * max(0, (maxlen - len(seq)))
            if len(seq) > maxlen:
                seq = seq[:maxlen]
       
        for word in seq:
            tokens += [word if self.word2count[word] > 0 else '<unk>']
        
        if pad_mode == 'pre':
            tokens = ['<s>', *pad, *tokens, '</s>']
        elif pad_mode == 'post':
            tokens = ['<s>', *tokens, *pad, '</s>']
        else:
            tokens = ['<s>', *tokens, '</s>']
            
        nums = [0] * len(tokens)
        for i in range(len(tokens)):
            nums[i] = self.word2index[tokens[i]]
            
        return tokens, nums

    
class Preprocess:
    @staticmethod
    def clean_text(x, punct=None):
        if punct is None:
            punct = {
                'sep'   : u'\u200b' + '‘…—−–.',
                'keep'  : "&",
                'remove': '?!，`“”’™•°'
            }

        for p in punct['sep']:
            x = x.replace(p, " ")
        for p in punct['keep']:
            x = x.replace(p, f" {p} ")
        for p in punct['remove']:
            x = x.replace(p, "")

        return x