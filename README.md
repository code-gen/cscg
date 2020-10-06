# Code Generation as a Dual Task of Code Summarization

[![render](https://img.shields.io/badge/render-nbviewer-orange)](https://nbviewer.jupyter.org/github/code-gen/cscg/tree/master/)

Ad-hoc implementation of the CS/CG model proposed by [Wei et al.](https://arxiv.org/abs/1910.05923)

## Getting started

- Each dataset must be defined as a sub-class of `torch.utils.data.Dataset`, with methods for
  - preprocessing and vocab builder (text -> vocab look-up indices)
  - `__getitem__` which must return a training example
  - `__len__`
  - generating train/test/valid splits
  - computing language model probabilites (i.e. `P(x)`, where `x`: anno/code tensor)


### Computing LM probabilities

- Get train/test/valid splits for a dataset.
- Construct a configuration for the LM.
- For each kind (anno/code), train a LM and dump the model as `lm-{dataset_name}-{kind}.pt` (e.g. `lm-django-anno.pt`).
- Finally, using these models, compute `P(x)` for each `x` (anno/code tensor).


## Reference
```
@article{wei2019code,
  title={Code Generation as a Dual Task of Code Summarization},
  author={Wei, Bolin and Li, Ge and Xia, Xin and Fu, Zhiyi and Jin, Zhi},
  journal={arXiv preprint arXiv:1910.05923},
  year={2019}
}
```
