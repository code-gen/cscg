import os
import pickle
import numpy as np
from tqdm.auto import tqdm


def from_home(x):
    return os.path.join(os.environ['HOME'], x)


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


def build_embedding_matrix(emb_dict, w_idx, len_voc, init='zeros'):
    assert init in ['zeros', 'normal']
    
    all_embs = np.stack(list(emb_dict.values()))
    emb_words_list = list(emb_dict.keys())
    embed_size = all_embs.shape[1]

    n_words = min(len_voc, len(w_idx))
    
    if init == 'normal':
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        emb_matrix = np.random.normal(emb_mean, emb_std, (n_words, embed_size))
    if init == 'zeros':
        emb_matrix = np.zeros((n_words, embed_size))

    for word, wi in w_idx.items():
        if wi >= len_voc: 
            continue

        emb_vector = emb_dict.get(word, None)
        if emb_vector is not None:
            emb_matrix[wi] = emb_vector

    return emb_matrix