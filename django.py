import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from argparse import Namespace

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

from lang import Lang


class Django(Dataset):
    def __init__(self, config: Namespace):
        super(Django, self).__init__()
        
        self.config = config
        
        self.anno_lang = Lang('anno')
        self.code_lang = Lang('code')
        
        self.__preprocess()
    
    def __str__(self):
        return 'Dataset<Django>'
    
    def __repr__(self):
        return str(self)
    
    def __preprocess(self) -> None:
        anno = [l.strip() for l in open(os.path.join(self.config.root_dir, 'all.anno')).readlines()]
        code = [l.strip() for l in open(os.path.join(self.config.root_dir, 'all.code')).readlines()]
        self.df = pd.DataFrame({'anno': anno, 'code': code})
        
        # construct anno language
        for s in anno:
            self.anno_lang.add_sentence(s, tokenize_mode='anno')
            
        self.anno_lang.normalize_vocab(min_freq=self.config.anno_min_freq)
        self.anno_lang.build_emb_matrix(emb_file=self.config.emb_file)
        
        # construct code language
        for s in code:
            self.code_lang.add_sentence(s, tokenize_mode='code')
      
        self.code_lang.normalize_vocab(min_freq=self.config.code_min_freq)
        
        # build examples
        self.anno, self.code = [], []
        
        for s in anno:
            nums = self.anno_lang.to_numeric(s, pad_mode='post', tokenize_mode='anno', maxlen=self.config.anno_seq_maxlen)
            self.anno += [torch.tensor(nums)]
        
        for s in code:
            nums = self.code_lang.to_numeric(s, pad_mode='post', tokenize_mode='code', maxlen=self.config.code_seq_maxlen)
            self.code += [torch.tensor(nums)]
             
    def __getitem__(self, idx):
        assert 0 <= idx < len(self)
        return self.anno[idx], self.code[idx]
            
    def __len__(self):
        assert len(self.anno) == len(self.code) == self.df.shape[0]
        return len(self.anno)
           
    def raw(self, idx):
        return {k: self.df.iloc[idx][k] for k in self.df.columns}
    
    def train_test_valid_split(self, test_p: float, valid_p: float, to_dir=None):
        """
        Generate train/test/valid splits and optionally dump to a directory.
        Useful for language models.
        
        :param test_p : percentage of all data for test
        :param valid_p: percentage of all data for train
        """
        x = self.df['anno'].values
        y = self.df['code'].values
        
        sz = 1 - test_p - valid_p
        x_train, x_test_valid, y_train, y_test_valid = train_test_split(x, y, train_size=sz)
        
        sz = test_p / (test_p + valid_p)
        x_test, x_valid, y_test, y_valid = train_test_split(x_test_valid, y_test_valid, train_size=sz)
        
        assert sum(map(len, [x_train, x_test, x_valid])) == len(x)
        assert sum(map(len, [y_train, y_test, y_valid])) == len(y)
        
        splits = {
            'train': (x_train, y_train),
            'test' : (x_test, y_test),
            'valid': (x_valid, y_valid)
        }
        
        if to_dir is None:
            return splits
        
        # dump to file
        os.makedirs(os.path.join(to_dir, 'anno'))
        os.makedirs(os.path.join(to_dir, 'code'))
        
        for k in splits:
            for i, t in enumerate(['anno', 'code']):
                with open(os.path.join(to_dir, f'{t}/{k}.txt'), 'wt') as fp:
                    for ex in splits[k][i]:
                        fp.write(f'{ex}\n')