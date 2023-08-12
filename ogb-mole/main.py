import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from modules.GNNs import GNN
from modules.SAGE import SAGEMol
from modules.ours import DIFFormerMol
from modules.utils import get_device
import torch_geometric
from tqdm import tqdm
import argparse
import os
import numpy as np

# importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import torch_scatter
import random

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def rewiring(edge_index, batch, ratio, edge_attr=None, type='delete'):
    edge_num, batch_size = edge_index.shape[1], batch.max() + 1

    if type == 'replace':
        num_nodes = torch_scatter.scatter(torch.ones_like(batch), batch) # [B]
        shift = torch.cumsum(num_nodes, dim=0) # [B]

        idx = torch.randint(0, edge_num, (int(ratio * edge_num),)) # [e]
        srx_idx_, end_idx_ = edge_index[:, idx]
        batchs = batch[end_idx_]  # [e]
        shifts = shift[batchs] # [e]

        srx_idx_new = torch.randint(0, edge_num, (idx.shape[0], )).to(batch.device) # [e]
        srx_idx_new = torch.remainder(srx_idx_new, shifts) # [e]
        edge_index[0, idx] = srx_idx_new
    else:
        srx_idx, end_idx = edge_index
        mask = torch.ones_like(srx_idx).bool()
        idx = torch.randint(0, edge_num, (int(ratio * edge_num),))  # [e]
        mask[idx] = False
        edge_index = torch.stack([srx_idx[mask], end_idx[mask]], dim=0) # [2, E - e]
        if edge_attr is not None:
            edge_attr = edge_attr[mask]

    if edge_attr is not None:
        return edge_index, edge_attr
    else:
        return edge_index

def sup_loss_calc(y, pred, task_type):
    # ignore nan targets (unlabeled) when computing training loss.
    is_labeled = y == y
    if "classification" in task_type:
        sup_loss = cls_criterion(
            pred.to(torch.float32)[is_labeled],
            y.to(torch.float32)[is_labeled]
        )
    else:
        sup_loss = reg_criterion(
            pred.to(torch.float32)[is_labeled],
            y.to(torch.float32)[is_labeled]
        )
    return sup_loss

def train(model, device, loader, optimizer, task_type, args):
    model.train()

    for step, d in enumerate(tqdm(loader, desc="Iteration")):
        d = d.to(device)
        x, y, batch = d.x, d.y, d.batch
        edge_index, edge_attr = d.edge_index, d.edge_attr

        if x.shape[0] == 1 or batch[-1] == 0:
            pass
        else:
            pred = model(x, edge_index, edge_attr, batch, args.use_block)
            optimizer.zero_grad()
            sup_loss = sup_loss_calc(y, pred, task_type)

            if args.use_reg:
                reg_loss_ = []
                for i in range(args.num_aug_branch):
                    edge_index_i, edge_attr_i = rewiring(edge_index.clone(), batch, args.modify_ratio, edge_attr.clone(), args.rewiring_type)

                    pred_i = model(x, edge_index_i, edge_attr_i, batch, args.use_block)
                    reg_loss_i = sup_loss_calc(y, pred_i, task_type)
                    reg_loss_.append(reg_loss_i)
                var = torch.var(torch.stack(reg_loss_))
                # reg_loss = torch.stack(reg_loss_).max()
                # print(sup_loss, var)
                loss = sup_loss + args.reg_weight * var
            else:
                loss = sup_loss

            loss.backward()
            optimizer.step()


def eval(model, device, loader, evaluator, args):
    model.eval()
    y_true = []
    y_pred = []

    for step, d in enumerate(tqdm(loader, desc="Iteration")):
        d = d.to(device)
        x, y, batch = d.x, d.y, d.batch
        edge_index, edge_attr = d.edge_index, d.edge_attr

        if x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(x, edge_index, edge_attr, batch, args.use_block)

            y_true.append(y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        'GNN baselines on ogbgmol* data with Pytorch Geometrics'
    )
    parser.add_argument(
        '--device', type=int, default=0,
        help='which gpu to use if any (default: 0), negative for cpu'
    )
    parser.add_argument(
        '--method', type=str, default='ours'
    )
    parser.add_argument(
        '--dropout', type=float, default=0.,
        help='dropout ratio (default: 0.5)'
    )
    parser.add_argument(
        '--num_layer', type=int, default=5,
        help='number of GNN message passing layers (default: 5)'
    )
    parser.add_argument(
        '--emb_dim', type=int, default=256,
        help='dimensionality of hidden units in GNNs (default: 256)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='input batch size for training (default: 32)'
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='number of epochs to train (default: 100)'
    )
    parser.add_argument(
        '--num_workers', type=int, default=0,
        help='number of workers (default: 0)'
    )
    parser.add_argument(
        '--dataset', type=str, default="ogbg-molhiv",
        help='dataset name (default: ogbg-molhiv)'
    )
    parser.add_argument(
        '--seed', default=2022, type=int,
        help='the random seed for experiment'
    )

    parser.add_argument(
        '--feature', type=str, default="full",
        help='full feature or simple feature'
    )
    parser.add_argument(
        '--result_dir', type=str, default="./saved_result/",
        help='filename to output result (default: )'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=0.001,
        help="learning rate"
    )
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument(
        '--model_dir', type=str, default="./saved_model/",
        help='filename to save model'
    )

    parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'sage', 'gat'])
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for residual link')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--use_residual', action='store_true', help='use residual link for each GNN layer')
    parser.add_argument('--use_weight', action='store_true', help='use weight for GNN convolution')
    parser.add_argument('--kernel', type=str, default='simple', choices=['simple', 'sigmoid'])
    parser.add_argument('--use_reg', action='store_true', help='use reg loss')
    parser.add_argument('--reg_weight', type=float, default=10.0, help='weight for variance loss')
    parser.add_argument('--num_aug_branch', type=int, default=3, help='num of branch for augmentation')
    parser.add_argument('--modify_ratio', type=float, default=0.1, help='ratio of edge rewiring for each graph')
    parser.add_argument('--use_block', action='store_true', help='compute all-pair attention within each block')
    parser.add_argument('--rewiring_type', type=str, default='delete', choices=['delete', 'replace'])


    args = parser.parse_args()

    torch_geometric.seed.seed_everything(args.seed)
    fix_seed(args.seed)

    device = get_device(args.device)

    # automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name=args.dataset)

    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

    split_idx = dataset.get_idx_split()

    # automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(
        dataset[split_idx["train"]], batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )
    valid_loader = DataLoader(
        dataset[split_idx["valid"]], batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        dataset[split_idx["test"]], batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )

    if args.method == 'ours':
        model = DIFFormerMol(
            num_tasks=dataset.num_tasks, num_layer=args.num_layer, emb_dim=args.emb_dim,
            JK='last', pooling = 'mean', virtual=False, num_heads=args.num_heads, kernel=args.kernel,
            alpha=args.alpha, dropout=args.dropout, use_bn=args.use_bn, use_residual=args.use_residual, use_weight=args.use_weight
        ).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    valid_curve = []
    test_curve = []
    train_curve = []

    i = 1
    model_path = args.model_dir + args.dataset + '_' + args.method + f'_v{i}'
    while os.path.exists(model_path):
        i += 1
        model_path = args.model_dir + args.dataset + '_' + args.method + f'_v{i}'
    result_path = args.result_dir + args.dataset + '_' + args.method + f'_v{i}'
    best_val = 0.

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, dataset.task_type, args)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator, args)
        valid_perf = eval(model, device, valid_loader, evaluator, args)
        test_perf = eval(model, device, test_loader, evaluator, args)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

        if args.save_model:
            if valid_perf[dataset.eval_metric] > best_val:
                best_val = valid_perf[dataset.eval_metric]
                torch.save(model.state_dict(), model_path)

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Train score: {}'.format(train_curve[best_val_epoch]))
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    if not args.result_dir == '':
        with open(result_path, 'a') as f:
            for i in range(len(train_curve)):
                f.write(f'train score {train_curve[i]:.3f} valid score {valid_curve[i]:.3f} test score {test_curve[i]:.3f}\n')

            f.write(f'best valid epoch: train score {train_curve[best_val_epoch]:.3f} valid score {valid_curve[best_val_epoch]:.3f} test score {test_curve[best_val_epoch]:.3f}')

if __name__ == "__main__":
    main()
