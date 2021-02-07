import codeop
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch as T
import torch.nn.functional as F
import torch.optim as O
from bagoftools.namespace import Namespace
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import StandardDataset
from language_model.lm_train import train_language_model

sys.path.insert(0, 'language_model/')


def language_model():
    _tp, _vp = 0.1, 0.2
    splits = dataset.train_test_valid_split(test_p=_tp, valid_p=_vp, seed=42)

    for kind in splits:
        for t in splits[kind]:
            vs = splits[kind][t]
            vs = T.cat(vs)
            vs = vs[vs != 0]
            splits[kind][t] = vs

    print(f'train {(1-_tp-_vp)*len(dataset):.2f} | test {_tp*len(dataset)} | dev {_vp*len(dataset)}')


    # ## 2.2. Train language model
    # for both anno and code.

    CFG.language_model = Namespace()
    CFG.language_model.__dict__ = {
        'dataset': os.path.basename(DATASET_DIR),
        # type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)
        'model': 'LSTM',
        'n_head': None,   # number of heads in the enc/dec of the Transformers
        'emb_size': 32,     # size of the word embeddings
        'n_hid': 64,     # number of hidden units per layer
        'n_layers': 1,      # number of layers
        'lr': 0.25,    # initial learning rate
        'clip': 0.25,   # gradient clipping
        'dropout_p': 0.05,    # dropout applied to layers
        'tied': False,  # whether to tie the word embeddings and softmax weights
        'log_interval': 100,
        'epochs': 500,  # upper epoch limit
        'batch_size': 128,
        'seed': None  # for reproducibility
    }


    lm_cfg = CFG.language_model

    for kind in ['anno', 'code']:
        print(f'Training LM for {kind}\n')

        lm_cfg.kind = kind
        lm_cfg.bptt = CFG.dataset_cfg.__dict__[f'{kind}_seq_maxlen']  # seq len
        # path to save the final model
        lm_cfg.save_path = f'./data/lm/lm-{lm_cfg.kind}-{lm_cfg.dataset}-epochs{lm_cfg.epochs}.pt'

        train_language_model(lm_cfg,
                             num_tokens=len(getattr(dataset, f'{kind}_lang')),
                             train_nums=splits[kind]['train'],
                             test_nums=splits[kind]['test'],
                             valid_nums=splits[kind]['valid'])

        print('*' * 120, '\n')


def compute_lm_probs():
    lm_cfg = CFG.language_model
    lm_paths = {
        k: f'./data/lm/lm-{k}-{lm_cfg.dataset}-epochs{lm_cfg.epochs}.pt' for k in ['anno', 'code']}

    for f in lm_paths.values():
        assert os.path.exists(f), f'Language Model: file <{f}> does not exist!'

    _ = dataset.compute_lm_probs(lm_paths)


def train():
    cg_model = Model(CFG, model_type='cg')
    cg_model.opt = O.Adam(lr=0.001, params=filter(
        lambda p: p.requires_grad, cg_model.parameters()))

    cs_model = Model(CFG, model_type='cs')
    cs_model.opt = O.Adam(lr=0.001, params=filter(
        lambda p: p.requires_grad, cs_model.parameters()))

# TODO: very hacky
    n = int(CFG.train_split * len(dataset))
    train_dataset = deepcopy(dataset)
    train_dataset.anno = dataset.anno[:n]
    train_dataset.code = dataset.code[:n]
    train_dataset.df = dataset.df.iloc[:n]
    train_dataset.lm_probs['anno'] = dataset.lm_probs['anno'][:n]
    train_dataset.lm_probs['code'] = dataset.lm_probs['code'][:n]
# ---

    kwargs = {'num_workers': 4, 'pin_memory': True} if CFG.cuda else {}
    train_loader = DataLoader(
        train_dataset, batch_size=CFG.batch_size, shuffle=True, **kwargs)
    print(
        f'DataLoader: {len(train_loader)} batches of size {CFG.batch_size} (total: {len(train_dataset)})')

    __cg_l = 0
    __cs_l = 0
    __att_l = 0
    __dual_l = 0
    __rep_every = 50
    __tb_every = __rep_every // 4

    CFG.to_file(os.path.join(exp_dir, 'config.json'))

    # train loop
    ts = 0
    for epoch_idx in range(1, CFG.num_epochs+1):

        for batch_idx, (anno, code, anno_lm_p, code_lm_p) in enumerate(train_loader, start=1):
            anno_len, code_len = anno.shape[1], code.shape[1]

            if CFG.cuda:
                anno, code, anno_lm_p, code_lm_p = map(
                    lambda t: t.cuda(), [anno, code, anno_lm_p, code_lm_p])

            # binary mask indicating the presence of padding token
            anno_mask = T.tensor(
                anno != dataset.anno_lang.token2index['<pad>']).byte()
            code_mask = T.tensor(
                code != dataset.code_lang.token2index['<pad>']).byte()

            # forward pass
            code_pred, code_att_mat = cg_model(src=anno, tgt=code)
            anno_pred, anno_att_mat = cs_model(src=code, tgt=anno)

            # loss computation
            l_cg_ce, l_cs_ce = 0, 0

            # CG cross-entropy loss
            for t in range(code_len):
                probs = code_pred[:, t, :].gather(
                    1, code[:, t].view(-1, 1)).squeeze(1)
                l_cg_ce += -T.log(probs) * code_mask[:, t] / code_len

            # CS cross-entropy loss
            for t in range(anno_len):
                probs = anno_pred[:, t, :].gather(
                    1, anno[:, t].view(-1, 1)).squeeze(1)
                l_cs_ce += -T.log(probs) * anno_mask[:, t] / anno_len

            # dual loss: P(x,y) = P(x).P(y|x) = P(y).P(x|y)
            l_dual = (code_lm_p - l_cs_ce - anno_lm_p + l_cg_ce) ** 2

            # attention loss: JSD
            l_att = JSD(anno_att_mat, code_att_mat.transpose(2, 1)) + \
                JSD(anno_att_mat.transpose(2, 1), code_att_mat)

            # final loss
            p, a = 0, 0
            l_cg = T.mean(l_cg_ce + p * 0.5 * l_dual + a * 0.9 * l_att)
            l_cs = T.mean(l_cs_ce + p * 0.5 * l_dual + a * 0.9 * l_att)

            # optimize CG
            cg_model.opt.zero_grad()
            l_cg.backward(retain_graph=True)
            cg_model.opt.step()

            # optimize CS
            cs_model.opt.zero_grad()
            l_cs.backward()
            cs_model.opt.step()

            # tensorboard
            if batch_idx % __tb_every == 0:
                for name, param in cg_model.named_parameters():
                    tb_writer.add_histogram(f'CG-{name}', param, ts)
                for name, param in cs_model.named_parameters():
                    tb_writer.add_histogram(f'CS-{name}', param, ts)
                tb_writer.add_scalar('train/CG_loss', l_cg.item(), ts)
                tb_writer.add_scalar('train/CS_loss', l_cs.item(), ts)
                tb_writer.add_scalar('train/ATT_loss', l_att.mean().item(), ts)
                tb_writer.add_scalar('train/DUAL_loss', l_dual.mean().item(), ts)
                ts += 1

            # reporting
            __cg_l += l_cg.item() / __rep_every
            __cs_l += l_cs.item() / __rep_every
            __att_l += l_att.mean().item() / __rep_every
            __dual_l += l_dual.mean().item() / __rep_every

            if batch_idx % __rep_every == 0:
                status = [f'Epoch {epoch_idx:>5d}/{CFG.num_epochs:>3d}', f'Batch {batch_idx:>5d}/{len(train_loader):5d}',
                          f'CG {__cg_l:7.5f}', f'CS {__cs_l:7.5f}', f'ATT {__att_l:7.5f}', f'DUAL {__dual_l:7.5f}']
                print(' | '.join(status))
                __cg_l, __cs_l, __att_l, __dual_l = 0, 0, 0, 0
        # --- epoch end


    torch.save(cg_model.state_dict(), os.path.join(exp_dir, 'cg_model.pt'))
    torch.save(cs_model.state_dict(), os.path.join(exp_dir, 'cs_model.pt'))

    tb_writer.close()


if __name__ == "__main__":
     # # 1. Setup


    ROOT_DIR = Path.home() / 'workspace/ml-data/msc-research'

    # DJANGO_DIR = ROOT_DIR / 'raw-datasets/testing' # simple django
    DJANGO_DIR = ROOT_DIR / 'raw-datasets/django'
    CONALA_DIR = ROOT_DIR / 'raw-datasets/conala-corpus'

    DATASET_DIR = DJANGO_DIR
    EMB_DIR = ROOT_DIR / 'embeddings'

    print(f'Dataset: {DATASET_DIR.stem}')

    # read dataset

    anno = [len(l.strip().split())
            for l in open(DATASET_DIR / 'all.anno').readlines()]
    code = [len(l.strip().split())
            for l in open(DATASET_DIR / 'all.code').readlines()]
    assert len(anno) == len(code)

    d = pd.DataFrame([{'anno': a, 'code': c} for (a, c) in zip(anno, code)])
    d.describe()


    # ## 1.2. Construct config
    CFG = Namespace()  # main config

    # sub-config for dataset
    CFG.dataset_cfg = Namespace()
    CFG.dataset_cfg.__dict__ = {
        'root_dir': DATASET_DIR,
        'anno_min_freq': 10,
        'code_min_freq': 10,
        'anno_seq_maxlen': 24,
        'code_seq_maxlen': 20,
        'emb_file': EMB_DIR / 'glove.6B.200d-ft-9-1.txt.pickle',
    }

    dataset = StandardDataset(config=CFG.dataset_cfg,
                              shuffle_at_init=True, seed=42)

    # sub-config for NL intents
    CFG.anno = Namespace()
    CFG.anno.__dict__ = {
        'lstm_hidden_size': 64,
        'lstm_dropout_p': 0.2,
        'att_dropout_p': 0.1,
        'lang': dataset.anno_lang,
        'load_pretrained_emb': True,
        'emb_size': 200,
    }

    # sub-config for source code
    CFG.code = Namespace()
    CFG.code.__dict__ = {
        'lstm_hidden_size': 64,
        'lstm_dropout_p': 0.2,
        'att_dropout_p': 0.1,
        'lang': dataset.code_lang,
        'load_pretrained_emb': False,
        'emb_size': 32,
    }

    CFG.__dict__.update({
        'exp_name': f'{DATASET_DIR.stem}-p{0}-a{1}',
        'cuda': True,
        'batch_size': 128,
        'num_epochs': 50,
        'train_split': 0.7,
    })


    exp_dir = f'./experiments/{CFG.exp_name}'
    log_dir = os.path.join(exp_dir, 'tb_logs')
    os.makedirs(exp_dir, exist_ok=False)
    tb_writer = SummaryWriter(log_dir=log_dir)
