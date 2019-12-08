import os
import pickle


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_pt_glove(emb_file: str) -> dict:
    if any([os.path.isfile(emb_file + ".pickle"), emb_file.endswith(".pickle")]):
        emb_file = emb_file + ".pickle" if not emb_file.endswith(".pickle") else emb_file
        emb_dict = pickle.load(open(emb_file, "rb"))
    else:
        emb_dict = dict(get_coefs(*o.split(" ")) for o in open(emb_file, encoding='latin'))
        pickle.dump(emb_dict, open(emb_file + ".pickle", "wb"))

    return emb_dict