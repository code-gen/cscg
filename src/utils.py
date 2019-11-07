import os
import numpy as np
import pandas as pd
import pickle
from tqdm.auto import tqdm
from collections import Counter


def clean_text(x, punct=None):
    if punct is None:
        punct = {
            'sep'   : u'\u200b' + "/-'´′‘…—−–.",
            'keep'  : "&",
            'remove': '?!.,，"#$%\'()*+-/:;<=>@[\\]^_`{|}~“”’™•°'
        }
        
    # x = x.lower()

    for p in punct['sep']:
        x = x.replace(p, " ")
    for p in punct['keep']:
        x = x.replace(p, f" {p} ")
    for p in punct['remove']:
        x = x.replace(p, "")

    return x


def build_vocab(df: pd.DataFrame, tokenize) -> Counter:
    sentences = df.progress_apply(tokenize).values
    vocab = Counter()

    for sentence in tqdm(sentences):
        for word in sentence:
            vocab[word] += 1

    return vocab


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_pt_glove(emb_file: str) -> dict:
    if any([os.path.isfile(emb_file + ".pickle"), emb_file.endswith(".pickle")]):
        print("> load glove from pickle")
        emb_file = emb_file + ".pickle" if not emb_file.endswith(".pickle") else emb_file
        emb = pickle.load(open(emb_file, "rb"))
    else:
        print("> load glove from txt, dumping to pickle")
        emb = dict(get_coefs(*o.split(" ")) for o in open(emb_file, encoding='latin'))
        pickle.dump(emb, open(emb_file + ".pickle", "wb"))

    return emb


def build_embedding_matrix(emb_dict, w_idx, len_voc):
    all_embs = np.stack(list(emb_dict.values()))
    emb_words_list = list(emb_dict.keys())
    embed_size = all_embs.shape[1]

    n_words = min(len_voc, len(w_idx))
    # emb_mean, emb_std = all_embs.mean(), all_embs.std()
    # emb_matrix = np.random.normal(emb_mean, emb_std, (n_words, embed_size))
    emb_matrix = np.zeros((n_words, embed_size))

    for word, wi in tqdm(w_idx.items(), total=len(w_idx.items())):
        if wi >= len_voc: continue

        emb_vector = emb_dict.get(word, None)
        if emb_vector is not None:
            emb_matrix[wi] = emb_vector

    return emb_matrix