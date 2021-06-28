import numpy as np
import torch
import torch.nn.functional as F


class LMProb:
    def __init__(self, model_path):
        self.model = torch.load(open(model_path, "rb"), map_location={"cuda:0": "cpu"})
        self.model = self.model.cpu()
        self.model.eval()

    def get_prob(self, nums, verbose=False):
        with torch.no_grad():
            inp = torch.tensor([int(nums[0])]).long().unsqueeze(0)
            hidden = self.model.init_hidden(bsz=1)
            log_probs = []

            for i in range(1, len(nums)):
                output, hidden = self.model(inp, hidden)

                # word_weights = output.squeeze().data.double().exp()
                # prob = word_weights[nums[i]] / word_weights.sum()
                probs = F.softmax(output.squeeze(), dim=-1)
                prob = probs[nums[i]]

                # append current log prob
                log_probs += [torch.log(prob)]
                inp.data.fill_(int(nums[i]))

            if verbose:
                for i in range(len(log_probs)):
                    print(
                        f"{nums[i+1]:4d}: P(w|s) = {np.exp(log_probs[i]):8.4f} | logP(w|s) = {log_probs[i]:8.4f}"
                    )
                print(f"=> sum_prob = {sum(log_probs):.4f}")

        return sum(log_probs) / len(log_probs)
