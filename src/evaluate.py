


def evaluate()
    cg_model = Model(CFG, model_type='cg')
    cs_model = Model(CFG, model_type='cs')

# exp_dir = f'./experiments/{CFG.exp_name}'
    exp_dir = f'./experiments/{os.path.basename(DATASET_DIR)}-p{0}-a{1}-minfreq2'

    cg_model.load_state_dict(torch.load(os.path.join(exp_dir, 'cg_model.pt')))
    cs_model.load_state_dict(torch.load(os.path.join(exp_dir, 'cs_model.pt')))

    exp_dir


# ## 5.1. Metrics


    def is_valid_code(line):
        "valid <=> (complete ^ valid) v (incomplete ^ valid_prefix)"
        try:
            codeop.compile_command(line)
        except SyntaxError:
            return False

        return True


    def to_tok(xs, mode):
        z = (xs)[0].cpu()
        z = z[(z != 0) & (z != 1) & (z != 2) & (z != 3)]
        if mode == 'code':
            return dataset.code_lang.to_tokens(z)[0]
        if mode == 'anno':
            return dataset.anno_lang.to_tokens(z)[0]


# TODO: very hacky
    n = int(CFG.train_split * len(dataset))
    test_dataset = deepcopy(dataset)
    test_dataset.anno = dataset.anno[n:]
    test_dataset.code = dataset.code[n:]
    test_dataset.df = dataset.df.iloc[n:]
    test_dataset.lm_probs['anno'] = dataset.lm_probs['anno'][n:]
    test_dataset.lm_probs['code'] = dataset.lm_probs['code'][n:]
# ---

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    assert len(test_loader) == len(dataset) - n


    ms = ['ind_match', 'exact_match', 'coverage']
    metrics = {
        'anno': {k: 0 for k in ms},
        'code': {k: 0 for k in ms}
    }
    metrics['code']['pov'] = 0

    anno_toks, code_toks = [], []

    with T.no_grad():
        cg_model.eval()
        cs_model.eval()

        for batch_idx, (anno, code, _, _) in tqdm(enumerate(test_loader, start=1), total=len(test_loader)):
            if CFG.cuda:
                anno, code = anno.cuda(), code.cuda()

            # binary mask indicating the presence of padding token
#         anno_mask = T.tensor(anno != dataset.anno_lang.token2index['<pad>']).byte()
#         code_mask = T.tensor(code != dataset.code_lang.token2index['<pad>']).byte()

            anno_mask = T.tensor((anno != 0) * (anno != 1)).byte()
            code_mask = T.tensor((code != 0) * (code != 1)).byte()

            # forward pass
            code_pred, code_att_mat = cg_model(src=anno, tgt=code)
            anno_pred, anno_att_mat = cs_model(src=code, tgt=anno)

            # TODO: ideally, this should be beam-search
            code_pred = code_pred.argmax(dim=2)
            anno_pred = anno_pred.argmax(dim=2)

            code_score = (
                ((code_pred == code) * code_mask).float().sum() / code_mask.sum()).cpu()
            anno_score = (
                ((anno_pred == anno) * anno_mask).float().sum() / anno_mask.sum()).cpu()

            # 1)
            metrics['code']['ind_match'] += code_score / len(test_loader)
            metrics['anno']['ind_match'] += anno_score / len(test_loader)

            # 2)
            if np.isclose(code_score, 1):
                metrics['code']['exact_match'] += 1 / len(test_loader)
            if np.isclose(anno_score, 1):
                metrics['anno']['exact_match'] += 1 / len(test_loader)

            # 3)
            sy = set([x.item()
                      for x in (code * code_mask)[0].cpu().data if x.item() != 0])
            sy_ = set([x.item() for x in (code_pred * code_mask)
                       [0].cpu().data if x.item() != 0])
            if len(set.difference(sy_, sy)) == 0:
                metrics['code']['coverage'] += 1 / len(test_loader)
            else:
                if np.isclose(code_score, 1):
                    print(set.difference(sy_, sy))

            sy = set([x.item()
                      for x in (anno * anno_mask)[0].cpu().data if x.item() != 0])
            sy_ = set([x.item() for x in (anno_pred * anno_mask)
                       [0].cpu().data if x.item() != 0])
            if len(set.difference(sy_, sy)) == 0:
                metrics['anno']['coverage'] += 1 / len(test_loader)

            # 4)
            if is_valid_code(' '.join(to_tok(code_pred * code_mask, 'code'))):
                metrics['code']['pov'] += 1 / len(test_loader)

            # save tokens
            code_toks += [(round(code_score.item(), 5),
                           to_tok(code_pred * code_mask, 'code'),
                           to_tok(code * code_mask, 'code'),
                           code_pred[0].cpu(),
                           code[0].cpu())]

            anno_toks += [(round(anno_score.item(), 5),
                           to_tok(anno_pred * anno_mask, 'anno'),
                           to_tok(anno * anno_mask, 'anno'),
                           anno_pred[0].cpu(),
                           anno[0].cpu())]

    code_toks = sorted(code_toks, key=lambda x: x[0])
    anno_toks = sorted(anno_toks, key=lambda x: x[0])

    with open(os.path.join(exp_dir, 'eval_code.txt'), 'wt') as fp:
        for i, (s, pt, tt, p, t) in enumerate(code_toks, start=1):
            fp.write(f'{i}\n')
            fp.write(f'{s}\n')
            fp.write(f'pred: {" ".join(pt)}\n')
            fp.write(f'true: {" ".join(tt)}\n')
            fp.write(f'pred_raw: {p}\n')
            fp.write(f'true_raw: {t}\n')
            fp.write(f'{"-"*80}\n')

    with open(os.path.join(exp_dir, 'eval_anno.txt'), 'wt') as fp:
        for i, (s, pt, tt, p, t) in enumerate(anno_toks, start=1):
            fp.write(f'{i}\n')
            fp.write(f'{s}\n')
            fp.write(f'pred: {" ".join(pt)}\n')
            fp.write(f'true: {" ".join(tt)}\n')
            fp.write(f'pred_raw: {p}\n')
            fp.write(f'true_raw: {t}\n')
            fp.write(f'{"-"*80}\n')

# results
    print(exp_dir.split('/')[-1])
    print(len(test_loader))
    for k in ms:
        print(f"{metrics['anno'][k]:7.5f}/{metrics['code'][k]:7.5f}", end=' ')
    print(round(metrics['code']['pov'], 5))
