import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

def eval_rmse(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)
    rmse = torch.sqrt(mse)
    return rmse

def eval_f1(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    f1 = f1_score(y_true, y_pred, average='macro')
    # macro_f1 = f1_score(y_true, y_pred, average='macro')
    return f1

def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)


def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
        # y_pred = torch.sigmoid(y_pred).detach().cpu().numpy()  # 将logits转换为概率
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)


@torch.no_grad()
def model_inference_sampler(model, loader, device):
    out = torch.empty((0, ))
    label = torch.empty((0, ), dtype=torch.long)
    for d in loader:
        d.x, d.edge_index, d.domain = \
            d.x.to(device), d.edge_index.to(device), d.domain.to(device)
        out_d = model(d).cpu()
        out = torch.cat([out, out_d[d.root_n_id]], dim=0)
        label = torch.cat([label, d.y], dim=0)
    return out, label


@torch.no_grad()
def evaluate_sampler(model, train_loader, valid_loader, test_loader, eval_func, device):
    model.eval()
    train_out, train_label = model_inference_sampler(model, train_loader, device)
    valid_out, valid_label = model_inference_sampler(model, valid_loader, device)
    test_out, test_label = model_inference_sampler(model, test_loader, device)

    train_acc = eval_func(train_label, train_out)
    valid_acc = eval_func(valid_label, valid_out)
    test_acc = eval_func(test_label, test_out)

    return train_acc, valid_acc, test_acc


@torch.no_grad()
def evaluate_multi_graph(model, dataset, eval_func, args):
    model.eval()

    train_idx, valid_idx, test_idx = dataset.train_idx, dataset.valid_idx, dataset.test_idx
    y = dataset.y.cpu()
    if args.method in ('ours', 'ours2'):
        out = model(dataset.x, dataset.edge_index, dataset.batch, block_wise=args.use_block).cpu()
    else:
        out = model(dataset.x, dataset.edge_index).cpu()

    train_acc = eval_func(y[train_idx], out[train_idx])
    valid_acc = eval_func(y[valid_idx], out[valid_idx])
    test_accs = []
    for t in test_idx:
        test_accs.append(eval_func(y[t], out[t]))
    result = [train_acc, valid_acc] + test_accs

    return result

@torch.no_grad()
def evaluate_single_graph(model, dataset_tr, dataset_val, dataset_te, eval_func, args):
    model.eval()

    y = dataset_tr.y.cpu()
    if args.method in ('ours', 'ours2'):
        out = model(dataset_tr.x, dataset_tr.edge_index, dataset_tr.batch, block_wise=args.use_block).cpu()
    else:
        out = model(dataset_tr.x, dataset_tr.edge_index).cpu()
    train_acc = eval_func(y[dataset_tr.train_idx], out[dataset_tr.train_idx])

    y = dataset_val.y.cpu()
    if args.method in ('ours', 'ours2'):
        out = model(dataset_val.x, dataset_val.edge_index, dataset_val.batch, block_wise=args.use_block).cpu()
    else:
        out = model(dataset_val.x, dataset_val.edge_index).cpu()
    valid_acc = eval_func(y[dataset_val.valid_idx], out[dataset_val.valid_idx])

    test_accs = []
    for d in dataset_te:
        y = d.y.cpu()
        if args.method in ('ours', 'ours2'):
            out = model(d.x, d.edge_index, d.batch, block_wise=args.use_block).cpu()
        else:
            out = model(d.x, d.edge_index).cpu()
        test_accs.append(eval_func(y[d.test_idx], out[d.test_idx]))
    result = [train_acc, valid_acc] + test_accs

    return result

@torch.no_grad()
def evaluate_node_task(model, tr_loader, val_loader, te_loader, eval_func, args, device):
    model.eval()

    train_y, train_output = [], []
    for batch in tr_loader:
        batch.to(device)
        y = batch.node_rgs_y
        train_y.append(y.cpu())
        out = model(batch.x, batch.edge_index)
        train_output.append(out[batch.mask].cpu())
    train_y, train_output = torch.cat(train_y, dim=0), torch.cat(train_output, dim=0)
    train_acc = eval_func(train_y.unsqueeze(1), train_output)

    val_y, val_output = [], []
    for batch in val_loader:
        batch.to(device)
        y = batch.node_rgs_y
        val_y.append(y.cpu())
        out = model(batch.x, batch.edge_index)
        val_output.append(out[batch.mask].cpu())
    val_y, val_output = torch.cat(val_y, dim=0), torch.cat(val_output, dim=0)
    valid_acc = eval_func(val_y.unsqueeze(1), val_output)

    ood_acc = []
    for test_loader in te_loader:
        te_y, te_output = [], []
        for batch in test_loader:
            batch.to(device)
            y = batch.y if args.task == 'node-cls' else batch.node_rgs_y
            te_y.append(y.cpu())
            out = model(batch.x, batch.edge_index)
            te_output.append(out[batch.mask].cpu())
        test_y, test_output = torch.cat(te_y, dim=0), torch.cat(te_output, dim=0)
        # test_acc = eval_func(test_y.unsqueeze(1), test_output)
        ood_acc.append(eval_func(test_y.unsqueeze(1), test_output))
    result = [train_acc, valid_acc] + ood_acc

    return result

@torch.no_grad()
def evaluate_edge_task(model, tr_loader, val_loader, te_loader, eval_func, args, device):
    model.eval()

    train_y, train_output = [], []
    for batch in tr_loader:
        batch.to(device)
        y = batch.edge_attr
        train_y.append(y.cpu())
        out = model(batch.x, batch.edge_index)
        train_output.append(out.cpu())
    train_y, train_output = torch.cat(train_y, dim=0), torch.cat(train_output, dim=0)
    train_acc = eval_func(train_y, train_output)

    val_y, val_output = [], []
    for batch in val_loader:
        batch.to(device)
        y = batch.edge_attr
        val_y.append(y.cpu())
        out = model(batch.x, batch.edge_index)
        val_output.append(out.cpu())
    val_y, val_output = torch.cat(val_y, dim=0), torch.cat(val_output, dim=0)
    valid_acc = eval_func(val_y, val_output)

    ood_acc = []
    for test_loader in te_loader:
        te_y, te_output = [], []
        for batch in test_loader:
            batch.to(device)
            y = batch.edge_attr
            te_y.append(y.cpu())
            out = model(batch.x, batch.edge_index)
            te_output.append(out.cpu())
        test_y, test_output = torch.cat(te_y, dim=0), torch.cat(te_output, dim=0)
        # test_acc = eval_func(test_y.unsqueeze(1), test_output)
        ood_acc.append(eval_func(test_y, test_output))
    result = [train_acc, valid_acc] + ood_acc

    
    return result

@torch.no_grad()
def evaluate_link_task(model, tr_loader, val_loader, te_loader, eval_func, args, device):
    model.eval()

    train_y, train_output = [], []
    for batch in tr_loader:
        batch.to(device)
        y = batch.link_pre_y
        train_y.append(y.cpu())
        # print("out of out: ", batch.x, batch.edge_index_sampled)
        out = model(batch.x, batch.edge_index, batch.edge_index_sampled)
        train_output.append(out.cpu())
    train_y, train_output = torch.cat(train_y, dim=0), torch.cat(train_output, dim=0)
    train_acc = eval_func(train_y, train_output)

    val_y, val_output = [], []
    for batch in val_loader:
        batch.to(device)
        y = batch.link_pre_y
        val_y.append(y.cpu())
        out = model(batch.x, batch.edge_index, batch.edge_index_sampled)
        val_output.append(out.cpu())
    val_y, val_output = torch.cat(val_y, dim=0), torch.cat(val_output, dim=0)
    valid_acc = eval_func(val_y, val_output)

    ood_acc = []
    for test_loader in te_loader:
        te_y, te_output = [], []
        for batch in test_loader:
            batch.to(device)
            y = batch.link_pre_y
            te_y.append(y.cpu())
            out = model(batch.x, batch.edge_index, batch.edge_index_sampled)
            te_output.append(out.cpu())
        test_y, test_output = torch.cat(te_y, dim=0), torch.cat(te_output, dim=0)
        # test_acc = eval_func(test_y.unsqueeze(1), test_output)
        ood_acc.append(eval_func(test_y, test_output))
    result = [train_acc, valid_acc] + ood_acc

    
    return result