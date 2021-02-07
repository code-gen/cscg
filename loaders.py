import pickle
from pathlib import Path


def load_pt_glove(emb_file: Path) -> dict:
    def _get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    with open(emb_file, 'rb') as fp:
        emb_dict = pickle.load(fp)

    return emb_dict
