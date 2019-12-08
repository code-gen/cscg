from collections import defaultdict, Counter

import torch
import numpy as np
import pandas as pd
import en_core_web_sm

from loaders import load_pt_glove

nlp = en_core_web_sm.load()
    

class Lang:
    reserved_tokens = ['<pad>', '<unk>', '<s>', '</s>']
    
    def __init__(self, name):
        self.name = name
        
        self.token2index = defaultdict(lambda: self.reserved_tokens.index('<unk>'))
        self.index2token = defaultdict(lambda: '<unk>')
        
        for i, w in enumerate(self.reserved_tokens):
            self.token2index[w] = i
            self.index2token[i] = w
                                                  
        self.token2count = Counter()
        self.n_tokens    = len(self.reserved_tokens)
        self.emb_matrix = None
        self.pad_idx = self.reserved_tokens.index('<pad>')
 

    def __str__(self):
        return f'Lang<{self.name}>'
    
    def __repr__(self):
        return str(self)
    
    def __getitem__(self, item):
        if isinstance(item, str):
            return {'index': self.token2index[item],
                    'count': self.token2count[item]}
            
        if isinstance(item, int):
            return {'token': self.index2token[item],
                    'count': self.token2count[self.index2token[item]]}
            
        return None
    
    def __len__(self):
        n1 = len(self.token2index)
        n2 = len(self.index2token)
        n3 = len(self.token2count)
        assert n1 == n2 == (n3 + len(self.reserved_tokens)) == self.n_tokens
        return self.n_tokens
    
    def add_sentence(self, sentence, tokenize_mode):        
        tokens = Preprocess.tokenize(Preprocess.clean_text(sentence), tokenize_mode)
        
        for tok in tokens:
            self.add_token(tok)
            
    def add_token(self, tok: str):
        if tok not in self.token2index:
            self.token2index[tok] = self.n_tokens
            self.token2count[tok] = 1
            self.index2token[self.n_tokens] = tok
            self.n_tokens += 1
        else:
            self.token2count[tok] += 1
                    
    def normalize_vocab(self, min_freq):
        items = list(self.token2count.items())
        
        for tok, count in items:
            if count >= min_freq:
                continue
                
            i = self.token2index[tok]    
            self.token2count.pop(tok, None)
            self.token2index.pop(tok, None)
            self.index2token.pop(i, None)
            self.n_tokens -= 1
            
            
    def __emb_from_token_idx(self, emb_dict, init='zeros'):
        assert init in ['zeros', 'normal']
    
        all_embs = np.stack(list(emb_dict.values()))
        embed_size = all_embs.shape[1]
        n_tokens = min(self.n_tokens, len(self.token2index))

        if init == 'normal':
            emb_mean, emb_std = all_embs.mean(), all_embs.std()
            emb_matrix = np.random.normal(emb_mean, emb_std, (n_tokens, embed_size))
        if init == 'zeros':
            emb_matrix = np.zeros((n_tokens, embed_size))

        for tok, idx in self.token2index.items():
            if idx >= self.n_tokens: 
                continue

            emb_vector = emb_dict.get(tok, None)
            if emb_vector is not None:
                emb_matrix[idx] = emb_vector
        
        return emb_matrix
        
                  
    def build_emb_matrix(self, emb_file: str, init_mode='zeros'):
        emb_dict = load_pt_glove(emb_file) # TODO: don't hardcode glove loader
        self.emb_matrix = self.__emb_from_token_idx(emb_dict, init='zeros')
        
          
    def to_numeric(self, sentence, tokenize_mode, pad_mode=None, maxlen=-1) -> ([str], [int]):
        tokens = Preprocess.tokenize(Preprocess.clean_text(sentence), tokenize_mode)
        
        if pad_mode is not None:
            pad = ['<pad>'] * max(0, (maxlen - len(tokens)))
            if len(tokens) > maxlen:
                tokens = tokens[:maxlen]
        
        tokens = [tok if self.token2count[tok] > 0 else '<unk>' for tok in tokens]
        
        if pad_mode == 'pre':
            tokens = ['<s>', *pad, *tokens, '</s>']
        elif pad_mode == 'post':
            tokens = ['<s>', *tokens, *pad, '</s>']
        else:
            tokens = ['<s>', *tokens, '</s>']
            
        return [self.token2index[tok] for tok in tokens]
    
    
    def to_tokens(self, nums):
        if len(nums.shape) == 1:
            nums = nums.unsqueeze(0)
        
        n, seq_len = nums.shape
            
        tokens = []
        for i in range(n):
            tokens.append([self.index2token[int(idx.item())] for idx in nums[i]])
            
        return tokens

    
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
    
    @staticmethod
    def tokenize(s, tokenize_mode):
        if tokenize_mode == 'anno':
            return [tok.text for tok in nlp.tokenizer(s)]
        
        if tokenize_mode == 'code':
            return s.split()
        
        raise NotImplementedError(tokenize_mode)