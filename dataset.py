import os
from argparse import Namespace

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from lang import Lang
from language_model.lm_prob import LMProb


class StandardDataset(Dataset):
    def __init__(self, config: Namespace, shuffle_at_init=False, seed=None):
        super(StandardDataset, self).__init__()

        self.config = config

        self.anno_lang = Lang("anno")
        self.code_lang = Lang("code")

        self.__preprocess(shuffle_at_init, seed)

    def __str__(self):
        return f"Dataset<{os.path.basename(self.config.root_dir)}>"

    def __repr__(self):
        return str(self)

    def __preprocess(self, shuffle, seed) -> None:
        anno = np.array(
            [
                l.strip()
                for l in open(
                    os.path.join(self.config.root_dir, "all.anno")
                ).readlines()
            ]
        )
        code = np.array(
            [
                l.strip()
                for l in open(
                    os.path.join(self.config.root_dir, "all.code")
                ).readlines()
            ]
        )
        assert anno.shape == code.shape

        if shuffle:
            np.random.seed(seed)
            ridx = np.random.permutation(len(anno))
            anno = anno[ridx]
            code = code[ridx]

        self.df = pd.DataFrame({"anno": anno, "code": code})

        # construct anno language
        for s in anno:
            self.anno_lang.add_sentence(s, tokenize_mode="anno")

        self.anno_lang.build_emb_matrix(emb_file=self.config.emb_file)

        # construct code language
        for s in code:
            self.code_lang.add_sentence(s, tokenize_mode="code")

        # build examples
        self.anno, self.code = [], []

        for s in anno:
            nums = self.anno_lang.to_numeric(
                s,
                tokenize_mode="anno",
                min_freq=self.config.anno_min_freq,
                pad_mode="post",
                max_len=self.config.anno_seq_maxlen,
            )
            self.anno += [torch.tensor(nums)]

        for s in code:
            nums = self.code_lang.to_numeric(
                s,
                tokenize_mode="code",
                min_freq=self.config.code_min_freq,
                pad_mode="post",
                max_len=self.config.code_seq_maxlen,
            )
            self.code += [torch.tensor(nums)]

        # construct uniform tensor
        self.anno = torch.stack(self.anno)
        self.code = torch.stack(self.code)

    def __getitem__(self, idx):
        # if lm probabilites have been computed
        if hasattr(self, "lm_probs"):
            return (
                self.anno[idx],
                self.code[idx],
                self.lm_probs["anno"][idx],
                self.lm_probs["code"][idx],
            )
        else:
            return self.anno[idx], self.code[idx]

    def __len__(self):
        assert len(self.anno) == len(self.code) == self.df.shape[0]
        return len(self.anno)

    def raw(self, idx):
        return {k: self.df.iloc[idx][k] for k in self.df.columns}

    def shuffle(self):
        r = np.random.permutation(len(self))
        self.anno = self.anno[r]
        self.code = self.code[r]
        if hasattr(self, "lm_probs"):
            self.lm_probs["anno"] = self.lm_probs["anno"][r]
            self.lm_probs["code"] = self.lm_probs["code"][r]

    def compute_lm_probs(self, lm_paths):
        """
        Compute LM probabilities for each unpadded, numericalized anno/code example.
        """

        self.lm_probs = {"anno": [], "code": []}

        pad_idx = {
            "anno": self.anno_lang.token2index["<pad>"],
            "code": self.code_lang.token2index["<pad>"],
        }

        for kind in self.lm_probs:
            lm = LMProb(lm_paths[kind])
            p = pad_idx[kind]

            for vec in tqdm(getattr(self, kind), total=len(self), desc=f"P({kind})"):
                self.lm_probs[kind] += [lm.get_prob(vec[vec != pad_idx[kind]])]

            self.lm_probs[kind] = torch.stack(self.lm_probs[kind])

        return self.lm_probs

    def train_test_valid_split(self, test_p: float, valid_p: float, seed=None):
        """
        Generate train/test/valid splits.

        :param test_p : percentage of all data for test
        :param valid_p: percentage of all data for train
        """
        x, y = self.anno, self.code

        sz = 1 - test_p - valid_p
        x_train, x_test_valid, y_train, y_test_valid = train_test_split(
            x, y, train_size=sz, random_state=seed
        )

        sz = test_p / (test_p + valid_p)
        x_test, x_valid, y_test, y_valid = train_test_split(
            x_test_valid, y_test_valid, train_size=sz, random_state=seed
        )

        assert sum(map(len, [x_train, x_test, x_valid])) == len(x)
        assert sum(map(len, [y_train, y_test, y_valid])) == len(y)

        splits = {
            "anno": {"train": x_train, "test": x_test, "valid": x_valid},
            "code": {"train": y_train, "test": y_test, "valid": y_valid},
        }

        return splits
