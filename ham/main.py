import time
from GCN import GCN
from torch.nn.functional import binary_cross_entropy_with_logits
import numpy as np
from dataset import HAMDs, col_fn
import json
import argparse
import os
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from GAT import GAT
from advdifformer import AdvDIFFormer
from difformer import DIFFormer
from graphgps import GPSModel


def train(model, loader, optimizer, device):
    model, losses = model.train(), []
    for graph in tqdm(loader):
        graph = graph.to(device)
        result = model(graph).squeeze(-1)
        loss = binary_cross_entropy_with_logits(result, graph.label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
    return np.mean(losses)


def evaluate(model, loader, device):
    model, acc, gt = model.eval(), [], []

    for graph in tqdm(loader):
        graph = graph.to(device)
        with torch.no_grad():
            result = model(graph).squeeze(-1)
        result[result >= 0] = 1
        result[result < 0] = 0

        acc.append(result.cpu().numpy())
        gt.append(graph.label.cpu().numpy())

    acc = np.concatenate(acc, axis=0)
    gt = np.concatenate(gt, axis=0)
    return (acc == gt).mean()


def create_dir(args):
    base_log_dir = os.path.join(args.log_dir, args.gnn)
    time_stamp = time.time()
    if not os.path.exists(base_log_dir):
        os.makedirs(base_log_dir)
    log_dir = os.path.join(base_log_dir, f'log-{time_stamp}.json')
    model_dir = os.path.join(base_log_dir, f'model-{time_stamp}.pth')
    return log_dir, model_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--store_model', action='store_true')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--K_order', type=int, default=3)
    args = parser.parse_args()

    log_dir, model_dir = create_dir(args)

    log_info = {
        'args': args.__dict__, 'train': [],
        'valid': [], 'test': [],
    }

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    if args.gnn == 'gcn':
        model = GCN(args.dim, args.num_layer, args.dropout).to(device)
    elif args.gnn == 'gat':
        model = GAT(
            args.dim, args.num_layer, args.dropout, args.heads
        ).to(device)
    elif args.gnn == 'advdifformer':
        model = AdvDIFFormer(
            emb_dim=args.dim, num_layer=args.num_layer,  num_heads=args.heads,
            K_order=args.K_order, alpha=args.alpha, dropout=args.dropout,
            use_bn=True, use_residual=True,
        ).to(device)
    elif args.gnn == 'difformer':
        model = DIFFormer(
            emb_dim=args.dim, num_layer=args.num_layer, num_heads=args.heads,
            alpha=args.alpha, dropout=args.dropout, use_bn=True,
            use_residual=True, use_weight=True,
        ).to(device)
    elif args.gnn == 'graphgps':
        model = GPSModel(
            hidden_channels=args.dim, num_layers=args.num_layer, 
            num_heads=args.heads, dropout=args.dropout, 
            attn_dropout=args.dropout, use_bn=True
        ).to(device)

    Dataset = HAMDs(args.data_dir)
    train_loader = DataLoader(
        Dataset[Dataset.data_split['train_idx']],
        batch_size=args.bs, shuffle=True, collate_fn=col_fn
    )

    valid_loader = DataLoader(
        Dataset[Dataset.data_split['valid_idx']],
        batch_size=args.bs, shuffle=False, collate_fn=col_fn
    )

    test_loader = DataLoader(
        Dataset[Dataset.data_split['test_idx']],
        batch_size=args.bs, shuffle=False, collate_fn=col_fn
    )
    log_info['split'] = Dataset.get_files_split()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_perf, best_ep = None, None
    for ep in range(args.epoch):
        print(f'Training at epoch {ep}')
        train_loss = train(model, train_loader, optimizer, device)
        val_acc = evaluate(model, valid_loader, device)
        test_acc = evaluate(model, test_loader, device)

        log_info['train'].append(train_loss)
        log_info['valid'].append(val_acc)
        log_info['test'].append(test_acc)
        print(f'train: {train_loss}, val: {val_acc}, test: {test_acc}')

        if best_perf is None or val_acc > best_perf:
            best_perf, best_ep = val_acc, ep
            if args.store_model:
                torch.save(model.state_dict(), model_dir)
            log_info['best'] = {
                'ep': ep, 'train': train_loss,
                'valid': val_acc, 'test': test_acc
            }

        with open(log_dir, 'w') as Fout:
            json.dump(log_info, Fout, indent=4)

    print('[best ep]', best_ep)
    print('[train]', log_info['train'][best_ep])
    print('[valid]', log_info['valid'][best_ep])
    print('[test]', log_info['test'][best_ep])
