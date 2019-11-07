import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from argparse import Namespace

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import clean_text, build_vocab, build_embedding_matrix, load_pt_glove


class Dataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
    
    def path_of(self, file):
        __path = os.path.join(self.root_dir, file)
        assert os.path.exists(__path), f'{file} is not part of the dataset!'
        return __path
    
    
class Django(Dataset):
    def __init__(self, root_dir, config):
        super(Django, self).__init__(root_dir)
        
        self.__preprocess(config)
    
    def __preprocess(self, config: Namespace) -> None:
        anno = [l.strip() for l in open(self.path_of('all.anno')).readlines()]
        code = [l.strip() for l in open(self.path_of('all.code')).readlines()]
        df = pd.DataFrame(list(zip(anno, code)), columns=['anno', 'code'])

        print('> clean text')
        df['anno'] = df['anno'].progress_apply(clean_text)

        print('> construct vocab')
        self.anno_vocab = build_vocab(df['anno'], tokenize=lambda s: list(map(lambda w: w.strip(), s.split())))
        self.code_vocab = build_vocab(df['code'], tokenize=lambda s: list(map(lambda w: w.strip(), s.split())))

        # split train / test
        n = int(config.p_split * len(df))
        df_train, df_test = df[:n], df[n:]

        x_train, y_train = df_train['anno'].values, df_train['code'].values
        x_test, y_test   = df_test['anno'].values, df_test['code'].values

        print('> tokenize')
        anno_tok = Tokenizer(num_words=len(self.anno_vocab))
        anno_tok.fit_on_texts(list(x_train) + list(x_test))

        code_tok = Tokenizer(num_words=len(self.code_vocab))
        code_tok.fit_on_texts(list(y_train) + list(y_test))

        print('> pad')
        self.x_train = pad_sequences(anno_tok.texts_to_sequences(x_train), maxlen=config.anno_seq_maxlen)
        self.x_test  = pad_sequences(anno_tok.texts_to_sequences(x_test), maxlen=config.anno_seq_maxlen)

        self.y_train = pad_sequences(code_tok.texts_to_sequences(y_train), maxlen=config.code_seq_maxlen)
        self.y_test  = pad_sequences(code_tok.texts_to_sequences(y_test), maxlen=config.code_seq_maxlen)

        print('> build emb matrix')
        emb_dict = load_pt_glove(config.emb_file)
        self.emb_matrix = build_embedding_matrix(emb_dict, w_idx=anno_tok.word_index, len_voc=len(self.anno_vocab))
        
        print('> DONE')