import re
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
        pp = Preprocess(tokenize_mode)
        tokens = pp.tokenize(pp.clean_text(sentence))
        
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
        
          
    def to_numeric(self, sentence, tokenize_mode, min_freq=1, pad_mode=None, max_len=-1):
        pp = Preprocess(tokenize_mode)
        
        tokens = pp.tokenize(pp.clean_text(sentence))
        tokens = [tok if self.token2count[tok] >= min_freq else '<unk>' for tok in tokens]
        
        if pad_mode is not None:
            pad = ['<pad>'] * max(0, (max_len - len(tokens) - 2)) # -2 for <s> and </s>
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
        
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
    def __init__(self, mode):
        assert mode in ['anno', 'code']
        self.mode = mode
    
    def clean_text(self, x):        
        x = re.sub(r'[‘…—−–]', ' ', x)
        x = re.sub(r'[?，`“”’™•°]', '', x)
        
        if self.mode == 'anno':
            x = re.sub(r'[,:;]', '', x)
            x = re.sub(r'([\+\-\*/=(){}%^&])', r' \1 ', x)
            x = re.sub(r'\.+$', r'', x)
        
        if self.mode == 'code':
            # TODO: x = re.sub(r'(\+=|\-=|\*=|/=|==|\*\*|//)', r' \1 ', x)
            x = re.sub(r'([\+\-\*/,:;=(){}%^&])', r' \1 ', x)
        
        x = re.sub(r'[ ]+', ' ', x)
        x = x.strip()
        return x
        
    def tokenize(self, x):        
        if self.mode == 'anno':
            return [tok.text for tok in nlp.tokenizer(x)]
        
        if self.mode == 'code':
            return x.split()