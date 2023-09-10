import torch
from collections import defaultdict
from datetime import datetime
from texttable import Texttable
import os
import numpy as np

class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """

    def __init__(self, runs, info=None, max_as_opt=True):
        self.info = info
        self.max_as_opt = max_as_opt
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) >= 4
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            if self.max_as_opt:
                argopt = result[:, 1].argmax().item()
            else:
                argopt = result[:, 1].argmin().item()
            print(f'Run {run + 1:02d}:')
            if self.max_as_opt:
                print(f'Highest Train: {result[:, 0].max():.2f}')
                print(f'Highest Valid: {result[:, 1].max():.2f}')
                for i in range(result.size(1) - 2):
                    print(f'Highest OOD Test: {result[:, i + 2].max():.2f}')
            else:
                print(f'Lowest Train: {result[:, 0].min():.2f}')
                print(f'Lowest Valid: {result[:, 1].min():.2f}')
                for i in range(result.size(1) - 2):
                    print(f'Lowest OOD Test: {result[:, i + 2].min():.2f}')
            print(f'Chosen epoch: {argopt+1}')
            print(f'Final Train: {result[argopt, 0]:.2f}')
            print(f'Final Valid: {result[argopt, 1]:.2f}')
            for i in range(result.size(1)-2):
                print(f'Final OOD Test: {result[argopt, i+2]:.2f}')
            self.test = result[argopt, 2]
        else:
            result = 100 * torch.tensor(self.results)
            best_results = []
            for r in result:
                if self.max_as_opt:
                    train_opt = r[:, 0].max().item()
                    valid_opt = r[:, 1].max().item()
                    test_ood_opt = []
                    for i in range(r.size(1) - 2):
                        test_ood_opt += [r[:, i+2].max().item()]
                    train_final = r[r[:, 1].argmax(), 0].item()
                    valid_final = r[r[:, 1].argmax(), 1].item()
                    test_ood_final = []
                    for i in range(r.size(1) - 2):
                        test_ood_final += [r[r[:, 1].argmax(), i+2].item()]
                else:
                    train_opt = r[:, 0].min().item()
                    valid_opt = r[:, 1].min().item()
                    test_ood_opt = []
                    for i in range(r.size(1) - 2):
                        test_ood_opt += [r[:, i + 2].min().item()]
                    train_final = r[r[:, 1].argmin(), 0].item()
                    valid_final = r[r[:, 1].argmin(), 1].item()
                    test_ood_final = []
                    for i in range(r.size(1) - 2):
                        test_ood_final += [r[r[:, 1].argmin(), i + 2].item()]
                best_result = [train_opt, valid_opt] + test_ood_opt + [train_final, valid_final] + test_ood_final
                best_results.append(best_result)

            best_result = torch.tensor(best_results)
            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Optimal Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Optimal Valid: {r.mean():.2f} ± {r.std():.2f}')
            ood_size = result[0].size(1)-2
            for i in range(ood_size):
                r = best_result[:, i+2]
                print(f'Optimal OOD Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, ood_size+2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, ood_size+3]
            print(f'Final Valid: {r.mean():.2f} ± {r.std():.2f}')
            for i in range(ood_size):
                r = best_result[:, i+4+ood_size]
                print(f'   Final OOD Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 4+ood_size:4+2*ood_size].mean(dim=-1)
            print(f'   Final All OOD Test: {r.mean():.2f} ± {r.std():.2f}')
            return best_result[:, -ood_size-2:]

import os
def save_result(args, results):
    if not os.path.exists(f'results/{args.dataset}'):
        os.makedirs(f'results/{args.dataset}')
    filename = f'results/{args.dataset}/{args.method}.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(
            f"{args.method} " + f"{args.encoder}: " + f"{args.lr}: " + f"{args.hidden_channels}: " + f"{args.num_layers} " + \
            f"{args.beta} " + f"{args.num_heads} " + f"{args.solver} " + f"{args.theta} " + f"{args.K_order} " + "\n")
        m = ""
        for i in range(results.size(1)):
            r = results[:, i]
            m += f"{r.mean():.2f} $\pm$ {r.std():.2f} "
        r = results[:, 2:].mean(dim=-1)
        m += f"{r.mean():.2f} ± {r.std():.2f} "
        m += "\n"
        write_obj.write(m)
