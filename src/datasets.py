import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from argparse import Namespace
from collections import Counter

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from torch.utils.data import Dataset

from utils import build_embedding_matrix, load_pt_glove

import en_core_web_sm
nlp = en_core_web_sm.load()


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


def build_vocab(df: pd.DataFrame, tokenize, min_freq=1) -> Counter:
    sentences = df.apply(tokenize).values
    tmp, vocab = Counter(), Counter()

    for sentence in tqdm(sentences):
        for word in sentence:
            tmp[word] += 1
            
    for word, count in tmp.items():
        if count < min_freq:
            continue
        vocab[word] = count

    return vocab    
    
    
class Django(Dataset):
    def __init__(self, config: Namespace):
        super(Django, self).__init__()
        
        self.config = config
        self.__preprocess()
        
    def path_of(self, file):
        __path = os.path.join(self.config.root_dir, file)
        assert os.path.exists(__path), f'{file} is not part of the dataset!'
        return __path
    
    def __preprocess(self) -> None:
        anno = [l.strip() for l in open(self.path_of('all.anno')).readlines()]
        code = [l.strip() for l in open(self.path_of('all.code')).readlines()]
        self.raw_df = pd.DataFrame(list(zip(anno, code)), columns=['anno', 'code'])
        
        print('> clean text')
        self.raw_df['anno'] = self.raw_df['anno'].apply(clean_text)

        print('> construct vocab')
        anno_vocab = build_vocab(self.raw_df['anno'], 
                                 tokenize=lambda sentence: [tok.text for tok in nlp.tokenizer(sentence)],
                                 min_freq=self.config.anno_min_freq)
        
        code_vocab = build_vocab(self.raw_df['code'], 
                                 tokenize=lambda s: s.split(),
                                 min_freq=self.config.code_min_freq)
        
        print('> tokenize')
        self.anno_tok = Tokenizer(num_words=len(anno_vocab), 
                                  oov_token='<unk>', 
                                  filters='',
                                  lower=False)
        self.anno_tok.fit_on_texts(self.raw_df['anno'].values.tolist())

        self.code_tok = Tokenizer(num_words=len(code_vocab), 
                                  oov_token='<unk>', 
                                  filters='',
                                  lower=False)
        self.code_tok.fit_on_texts(self.raw_df['code'].values.tolist())

        print('> pad')
        anno_tts = self.anno_tok.texts_to_sequences(self.raw_df['anno'].values)
        self.X = pad_sequences(anno_tts, maxlen=self.config.anno_seq_maxlen, padding='post')
        
        code_tts = self.code_tok.texts_to_sequences(self.raw_df['code'].values)
        self.Y = pad_sequences(code_tts, maxlen=self.config.code_seq_maxlen, padding='post')

        print('> build emb matrix')
        emb_dict = load_pt_glove(self.config.emb_file)
        self.emb_matrix = build_embedding_matrix(emb_dict, w_idx=self.anno_tok.word_index, len_voc=len(anno_vocab))
        
        print('> DONE')
        
    def __getitem__(self, idx):
        assert 0 <= idx < len(self)
        return self.X[idx], self.Y[idx]
        
    def __len__(self):
        assert len(self.X) == len(self.Y) == self.raw_df.shape[0]
        return len(self.X)
        
    def raw_example(self, idx):
        return {k: self.raw_df.iloc[idx][k] for k in self.raw_df.columns}