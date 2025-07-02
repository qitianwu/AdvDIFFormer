import torch
from collections import defaultdict
from datetime import datetime
from texttable import Texttable
import os
import numpy as np

class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        # assert len(result) >= 4
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.4f}')
            print(f'Highest Valid: {result[:, 1].max():.4f}')
            for i in range(result.size(1)-3):
                print(f'Highest OOD Test: {result[:, i+3].max():.4f}')
            print(f'Chosen epoch: {argmax}')
            print(f'Final Train: {result[argmax, 0]:.4f}')
            print(f'Final Valid: {result[argmax, 1]:.4f}')
            for i in range(result.size(1)-2):
                print(f'Final OOD Test: {result[argmax, i+2]:.4f}')
            self.test = result[argmax, 2]
        else:
            result = 100 * torch.tensor(self.results)
            best_results = []
            for r in result:
                train_high = r[:, 0].max().item()
                valid_high = r[:, 1].max().item()
                test_ood_high = []
                for i in range(r.size(1) - 2):
                    test_ood_high += [r[:, i+2].max().item()]
                train_final = r[r[:, 1].argmax(), 0].item()
                valid_final = r[r[:, 1].argmax(), 1].item()
                test_ood_final = []
                for i in range(r.size(1) - 2):
                    test_ood_final += [r[r[:, 1].argmax(), i+2].item()]
                best_result = [train_high, valid_high] + test_ood_high + [train_final, valid_final] + test_ood_final
                best_results.append(best_result)

            best_result = torch.tensor(best_results)
            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.4f} ± {r.std():.4f}')
            ood_size = result[0].size(1)-2
            for i in range(ood_size):
                r = best_result[:, i+2]
                print(f'Highest OOD Test: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, ood_size+2]
            print(f'  Final Train: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, ood_size+3]
            print(f'Final Valid: {r.mean():.4f} ± {r.std():.4f}')
            for i in range(ood_size):
                r = best_result[:, i+4+ood_size]
                print(f'   Final OOD Test: {r.mean():.4f} ± {r.std():.4f}')
            return best_result[:, -ood_size-2:]

    def print_statistics_neg(self, run=None):
        if run is not None:
            result = torch.tensor(self.results[run])
            argmax = result[:, 1].argmin().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].min():.4f}')
            print(f'Highest Valid: {result[:, 1].min():.4f}')
            print(f'Highest OOD Test: {result[:, 2].min():.4f}')
            print(f'Chosen epoch: {argmax}')
            print(f'Final Train: {result[argmax, 0]:.4f}')
            print(f'Final Valid: {result[argmax, 1]:.4f}')
            for i in range(result.size(1)-2):
                print(f'Final OOD Test: {result[argmax, i+2]:.4f}')
            self.test = result[argmax, 2]
        else:
            result = torch.tensor(self.results)
            best_results = []
            for r in result:
                train_high = r[:, 0].min().item()
                valid_high = r[:, 1].min().item()
                test_ood_high = []
                for i in range(r.size(1) - 2):
                    test_ood_high += [r[:, i+2].min().item()]
                train_final = r[r[:, 1].argmin(), 0].item()
                valid_final = r[r[:, 1].argmin(), 1].item()
                test_ood_final = []
                for i in range(r.size(1) - 2):
                    test_ood_final += [r[r[:, 1].argmin(), i+2].item()]
                best_result = [train_high, valid_high] + test_ood_high + [train_final, valid_final] + test_ood_final
                best_results.append(best_result)

            best_result = torch.tensor(best_results)
            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.4f} ± {r.std():.4f}')
            ood_size = result[0].size(1)-2
            for i in range(ood_size):
                r = best_result[:, i+2]
                print(f'Highest OOD Test: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, ood_size+2]
            print(f'  Final Train: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, ood_size+3]
            print(f'Final Valid: {r.mean():.4f} ± {r.std():.4f}')
            for i in range(ood_size):
                r = best_result[:, i+4+ood_size]
                print(f'   Final OOD Test: {r.mean():.4f} ± {r.std():.4f}')
            return best_result[:, -ood_size-2:]

import os
def save_result(args, results):
    if not os.path.exists(f'results/{args.task}'):
        os.makedirs(f'results/{args.task}')
    filename = f'results/{args.task}/{args.method}-{args.encoder}.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(
            f"{args.method} " + f"{args.encoder} " + f"solver: {args.solver}" + f"{args.context_type}: " + f"lr: {args.lr}: " + f"dropout:{args.dropout}"\
            f"#layers:{args.num_layers} " + f"hidden:{args.hidden_channels} " + f"heads:{args.num_heads} " + f"tau:{args.tau} "\
            f"lamda:{args.lamda} " + f"r:{args.r} " + f"K: {args.K} " + f"weight: {args.weight_decay} " + f"beta: {args.beta}" + f"theta: {args.theta}" + f"K_order: {args.K_order}" + "\n")
        write_obj.write(
            f"mixup_alpha: {args.mixup_alpha} " + f"mixup_prob: {args.mixup_prob} " + f"smooth:{args.label_smooth_val} " + \
            f"group:{args.groupdro_step_size} " + '\n')
        write_obj.write(
            f"dann:{args.dann_alpha} " + f"eerm_beta:{args.eerm_beta} " + f"lr_a:{args.lr_a} " + \
            f"variant:{args.variant} " + '\n')
        write_obj.write(
            f"irm_lambda:{args.irm_lambda} " + f"irm_penalty_anneal_iter:{args.irm_penalty_anneal_iter} " + '\n')
        for i in range(results.size(1)):
            r = results[:, i]
            write_obj.write(f"{r.mean():.4f} $\pm$ {r.std():.4f} \n")
