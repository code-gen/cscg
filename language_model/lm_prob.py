import torch
import pickle

class LMProb():
    def __init__(self, model_path, dict_path):
        self.dictionary = pickle.load(open(dict_path, 'rb'))
        
        self.model = torch.load(open(model_path, 'rb'), map_location={'cuda:0': 'cpu'})
        self.model = self.model.cpu()
        self.model.eval()

    def get_prob(self, words, verbose=False):
        with torch.no_grad():
            pad_words = ['<s>'] + words + ['</s>']
            idxs = [self.dictionary.getid(w) for w in pad_words]
            inp = torch.tensor([int(idxs[0])]).long().unsqueeze(0)

            if verbose:
                print('words =', pad_words)
                print('idxs  =', idxs)

            hidden = self.model.init_hidden(bsz=1)
            log_probs = []
            
            for i in range(1, len(pad_words)):
                output, hidden = self.model(inp, hidden)
                word_weights = output.squeeze().data.double().exp()
                prob = word_weights[idxs[i]] / word_weights.sum()
                log_probs.append(torch.log(prob))
                inp.data.fill_(int(idxs[i]))

            if verbose:
                for i in range(len(log_probs)):
                    print('{:>24s} => {:4d},\tlogP(w|s)={:8.4f}'.format(pad_words[i+1], idxs[i+1], log_probs[i]))
                print('\n  => sum_prob = {:.4f}'.format(sum(log_probs)))

        return sum(log_probs) / len(log_probs)