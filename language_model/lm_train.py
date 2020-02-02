import os
import time
import pickle

import numpy as np
import torch
import torch.nn as nn

import models


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

    
def batchify(data, bsz, device):
    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.

    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(bptt, source, i):
    # get_batch subdivides the source data into chunks of length CFG.bptt.
    # If source is equal to the example output of the batchify function, with
    # a bptt-limit of 2, we'd get the following two Variables for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of data is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the seq_len dimension in the LSTM.

    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(CFG, model, num_tokens, criterion, data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.0
    
    if CFG.model != 'Transformer':
        hidden = model.init_hidden(bsz=1)
    
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, CFG.bptt):
            data, targets = get_batch(CFG.bptt, data_source, i)
            if CFG.model == 'Transformer':
                output = model(data)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, num_tokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    
    return total_loss / len(data_source)


def train_epoch(epoch, CFG, model, num_tokens, train_data, criterion, lr):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    
    if CFG.model != 'Transformer':
        hidden = model.init_hidden(CFG.batch_size)
        
    for batch, i in enumerate(range(0, train_data.size(0) - 1, CFG.bptt)):
        data, targets = get_batch(CFG.bptt, train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if CFG.model == 'Transformer':
            output = model(data)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, num_tokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if epoch % 20 == 0 and batch % CFG.log_interval == 0 and batch > 0:
            cur_loss = total_loss / CFG.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | loss {:8.5f} | ppl {:10.5f}'.format(
                epoch, batch, len(train_data) // CFG.bptt, lr, elapsed * 1000 / CFG.log_interval, cur_loss, np.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def train_language_model(CFG, train_nums, test_nums, valid_nums, num_tokens):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    if CFG.seed is not None:
        torch.manual_seed(CFG.seed)
        print(f'using seed {CFG.seed}')
            
    train_data = batchify(train_nums, CFG.batch_size, device=device)
    val_data   = batchify(valid_nums, bsz=1, device=device)
    test_data  = batchify(test_nums, bsz=1, device=device)
    
    if CFG.model == 'Transformer':
        model = models.TransformerModel(num_tokens, CFG.emb_size, CFG.nhead, CFG.n_hid, CFG.n_layers, CFG.dropout_p).to(device)
    else:
        model = models.RNNModel(CFG.model, num_tokens, CFG.emb_size, CFG.n_hid, CFG.n_layers, CFG.dropout_p, CFG.tied).to(device)

    criterion = nn.CrossEntropyLoss()
    
    lr = CFG.lr
    best_val_loss = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, CFG.epochs+1):
            epoch_start_time = time.time()
            train_epoch(epoch, CFG, model, num_tokens, train_data, criterion, lr)
            val_loss = evaluate(CFG, model, num_tokens, criterion, val_data)

            _saved = False
            
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(CFG.save_path, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
                _saved = True
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr = 0.9 * lr
                
            if epoch % 20 == 0:
                print('-' * 120)
                _s = '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:8.5f} | valid ppl {:10.5f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss, np.exp(val_loss))
                if _saved:
                    _s += ' | * saved best model'
                print(_s)
                print('-' * 120)
            
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
        
    # Load the best saved model.
    with open(CFG.save_path, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if CFG.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(CFG, model, num_tokens, criterion, test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, np.exp(test_loss)))
    print('=' * 89)