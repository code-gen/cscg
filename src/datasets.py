import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from argparse import Namespace

from torch.utils.data import Dataset

from lang import Lang


class Django(Dataset):
    def __init__(self, config: Namespace):
        super(Django, self).__init__()
        
        self.config = config
        
        self.anno_lang = Lang('anno')
        self.code_lang = Lang('code')
        
        self.__preprocess()
        
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
            self.anno_lang.add_sentence(s, tokenize='default')
            
        self.anno_lang.normalize_vocab(min_freq=self.config.anno_min_freq)
        self.anno_lang.build_emb_matrix(emb_file=self.config.emb_file)
        
        # construct code language
        for s in code:
            self.code_lang.add_sentence(s, tokenize=lambda s: s.split())
      
        self.code_lang.normalize_vocab(min_freq=self.config.code_min_freq)
        
        # build examples
        self.X, self.Y = [], []
        for s in anno:
            tokens, nums = self.anno_lang.numericalize(s, pad_mode='post', maxlen=self.config.anno_seq_maxlen)
            self.X += [nums]
        
        for s in code:
            tokens, nums = self.code_lang.numericalize(s, pad_mode='post', maxlen=self.config.code_seq_maxlen)
            self.Y += [nums]
             
    def __getitem__(self, idx):
        assert 0 <= idx < len(self)
        return self.X[idx], self.Y[idx]
            
    def __len__(self):
        assert len(self.X) == len(self.Y) == self.df.shape[0]
        return len(self.X)
           
    def raw(self, idx):
        return {k: self.df.iloc[idx][k] for k in self.df.columns}