import torch as T
import torch.nn as nn
from bagoftools.namespace import Namespace


def get_embeddings(config: Namespace) -> nn.Embedding:
    emb = nn.Embedding(len(config.lang), config.emb_size,
                       padding_idx=config.lang.pad_idx)

    if config.load_pretrained_emb:
        assert config.lang.emb_matrix is not None
        emb.weight = nn.Parameter(
            T.tensor(config.lang.emb_matrix, dtype=T.float32))
        emb.weight.requires_grad = False

    return emb


class Model(nn.Module):
    def __init__(self, config: Namespace, model_type):
        """
        :param model_type: cs / cg
        cs: code -> anno
        cg: anno -> code
        """
        super(Model, self).__init__()

        assert model_type in ['cs', 'cg']
        self.model_type = model_type

        src_cfg = config.anno if model_type == 'cg' else config.code
        tgt_cfg = config.code if model_type == 'cg' else config.anno

        # 1. ENCODER
        self.src_embedding = get_embeddings(src_cfg)
        self.encoder = nn.LSTM(input_size=src_cfg.emb_size,
                               hidden_size=src_cfg.lstm_hidden_size,
                               dropout=src_cfg.lstm_dropout_p,
                               bidirectional=True,
                               batch_first=True)

        self.decoder_cell_init_linear = nn.Linear(in_features=2*src_cfg.lstm_hidden_size,
                                                  out_features=tgt_cfg.lstm_hidden_size)

        # 2. ATTENTION
        # project source encoding to decoder rnn's h space (W from Luong score general)
        self.att_src_W = nn.Linear(in_features=2*src_cfg.lstm_hidden_size,
                                   out_features=tgt_cfg.lstm_hidden_size,
                                   bias=False)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the attentional vector in (W from Luong eq. 5)
        self.att_vec_W = nn.Linear(in_features=2*src_cfg.lstm_hidden_size + tgt_cfg.lstm_hidden_size,
                                   out_features=tgt_cfg.lstm_hidden_size,
                                   bias=False)

        # 3. DECODER
        self.tgt_embedding = get_embeddings(tgt_cfg)
        self.decoder = nn.LSTMCell(input_size=tgt_cfg.emb_size + tgt_cfg.lstm_hidden_size,
                                   hidden_size=tgt_cfg.lstm_hidden_size)

        # prob layer over target language
        self.readout = nn.Linear(in_features=tgt_cfg.lstm_hidden_size,
                                 out_features=len(tgt_cfg.lang),
                                 bias=False)

        self.dropout = nn.Dropout(tgt_cfg.att_dropout_p)

        # 4. COPY MECHANISM
        self.copy_gate = ...  # TODO

        # save configs
        self.src_cfg = src_cfg
        self.tgt_cfg = tgt_cfg

    def forward(self, src, tgt):
        """
        src: bs, max_src_len
        tgt: bs, max_tgt_len
        """
        enc_out, (h0_dec, c0_dec) = self.encode(src)
        scores, att_mats = self.decode(enc_out, h0_dec, c0_dec, tgt)

        return scores, att_mats

    def encode(self, src):
        """
        src : bs x max_src_len (emb look-up indices)
        out : bs x max_src_len x 2*hid_size
        h/c0: bs x tgt_hid_size
        """
        emb = self.src_embedding(src)
        out, (hn, cn) = self.encoder(emb)  # hidden is zero by default

        # construct initial state for the decoder
        c0_dec = self.decoder_cell_init_linear(T.cat([cn[0], cn[1]], dim=1))
        h0_dec = c0_dec.tanh()

        return out, (h0_dec, c0_dec)

    def decode(self, src_enc, h0_dec, c0_dec, tgt):
        """
        src_enc: bs, max_src_len, 2*hid_size (== encoder output)
        h/c0   : bs, tgt_hid_size
        tgt    : bs, max_tgt_len (emb look-up indices)
        """
        batch_size, tgt_len = tgt.shape
        scores, att_mats = [], []

        hidden = (h0_dec, c0_dec)

        emb = self.tgt_embedding(tgt)  # bs, max_tgt_len, tgt_emb_size

        att_vec = T.zeros(
            batch_size, self.tgt_cfg.lstm_hidden_size, requires_grad=False)
        if CFG.cuda:
            att_vec = att_vec.cuda()

        # Luong W*hs: same for each timestep of the decoder
        src_enc_att = self.att_src_W(src_enc)  # bs, max_src_len, tgt_hid_size

        for t in range(tgt_len):
            emb_t = emb[:, t, :]
            x = T.cat([emb_t, att_vec], dim=-1)
            h_t, c_t = self.decoder(x, hidden)

            ctx_t, att_mat = self.luong_attention(h_t, src_enc, src_enc_att)

            # Luong eq. (5)
            att_t = self.att_vec_W(T.cat([h_t, ctx_t], dim=1))
            att_t = att_t.tanh()
            att_t = self.dropout(att_t)

            # Luong eq. (6)
            score_t = self.readout(att_t)
            score_t = F.softmax(score_t, dim=-1)

            scores += [score_t]
            att_mats += [att_mat]

            # for next state t+1
            att_vec = att_t
            hidden = (h_t, c_t)

        # bs, max_tgt_len, tgt_vocab_size
        scores = T.stack(scores).permute((1, 0, 2))

        # each element: bs, max_src_len, max_tgt_len
        att_mats = T.cat(att_mats, dim=1)

        return scores, att_mats

    def luong_attention(self, h_t, src_enc, src_enc_att, mask=None):
        """
        h_t               : bs, hid_size
        src_enc (hs)      : bs, max_src_len, 2*src_hid_size
        src_enc_att (W*hs): bs, max_src_len, tgt_hid_size
        mask              : bs, max_src_len

        ctx_vec    : bs, 2*src_hid_size
        att_weight : bs, max_src_len
        att_mat    : bs, 1, max_src_len
        """

        # bs x src_max_len
        score = T.bmm(src_enc_att, h_t.unsqueeze(2)).squeeze(2)

        if mask:
            score.data.masked_fill_(mask, -np.inf)

        att_mat = score.unsqueeze(1)
        att_weights = F.softmax(score, dim=-1)

        # sum per timestep
        ctx_vec = T.sum(att_weights.unsqueeze(2) * src_enc, dim=1)

        return ctx_vec, att_mat

    def beam_search(self, src, width=3):
        """
        Choose most probable sequence, considering top `width` candidates.
        """

        hyp = []

        batch_size, src_len = src.shape
        enc_out, (h0_dec, c0_dec) = self.encode(src)

        scores, att_mats = [], []

        hidden = (h0_dec, c0_dec)

        att_vec = T.zeros(
            batch_size, self.tgt_cfg.lstm_hidden_size, requires_grad=False).cuda()

        # Luong W*hs: same for each timestep of the decoder
        src_enc_att = self.att_src_W(src_enc)  # bs, max_src_len, tgt_hid_size

        for t in range(tgt_len):
            emb_t = self.tgt_embedding(hyp[-1])
            x = T.cat([emb_t, att_vec], dim=-1)
            h_t, c_t = self.decoder(x, hidden)

            ctx_t, att_mat = self.luong_attention(h_t, src_enc, src_enc_att)

            att_t = F.tanh(self.att_vec_W(T.cat([h_t, ctx_t], dim=1)))
            # att_t = self.dropout(att_t)

            score_t = F.softmax(self.readout(att_t), dim=-1)

            scores += [score_t]
            att_mats += [att_mat]

            # for next state t+1
            att_vec = att_t
            hidden = (h_t, c_t)

        # bs, max_tgt_len, tgt_vocab_size
        scores = T.stack(scores).permute((1, 0, 2))

        # each element: bs, max_src_len, max_tgt_len
        att_mats = T.cat(att_mats, dim=1)

        return hyp


def JSD(a, b, mask=None):
    eps = 1e-8

    assert a.shape == b.shape
    _, n, _ = a.shape

    xa = F.softmax(a, dim=2) + eps
    xb = F.softmax(b, dim=2) + eps

    # common, averaged dist
    avg = 0.5 * (xa + xb)

    # kl
    xa = T.sum(xa * T.log(xa / avg), dim=2)
    xb = T.sum(xb * T.log(xb / avg), dim=2)

    # js
    xa = T.sum(xa, dim=1) / n
    xb = T.sum(xb, dim=1) / n

    return 0.5 * (xa + xb)
