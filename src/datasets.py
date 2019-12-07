import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from argparse import Namespace

import torch
from torch.utils.data import Dataset

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
    
    def path_of(self, file) -> str:
        __path = os.path.join(self.config.root_dir, file)
        assert os.path.exists(__path), f'{file} is not part of the dataset!'
        return __path
    
    def __preprocess(self) -> None:
        anno = [l.strip() for l in open(self.path_of('all.anno')).readlines()]
        code = [l.strip() for l in open(self.path_of('all.code')).readlines()]
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