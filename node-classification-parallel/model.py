import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from encoders import *
import scipy.sparse
import numpy as np
from data_utils import split_into_groups, convert_to_one_hot
from torch.autograd import grad as agrad
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import torch_scatter
from data_utils import normalize
import random

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

class Identity(object):
    """Change gt_label to one_hot encoding and keep img as the same.
    Args:
        num_classes (int): The number of classes.
        prob (float): MixUp probability. It should be in range [0, 1].
            Default to 1.0
    """

    def __init__(self, num_classes, prob=1.0, mode='original'):
        super(Identity, self).__init__()

        assert isinstance(num_classes, int)
        assert isinstance(prob, float) and 0.0 <= prob <= 1.0
        assert mode in ['original', 'multilabel']

        self.num_classes = num_classes
        self.prob = prob
        self.mode = mode

    def one_hot(self, gt_label):
        return F.one_hot(gt_label, num_classes=self.num_classes)

    def __call__(self, img, gt_label):
        if self.mode == 'original':
            return img, self.one_hot(gt_label)
        else:
            return img, gt_label.float()


class BatchMixupLayer(object):
    """Mixup layer for batch mixup."""

    def __init__(self, alpha, num_classes, prob=1.0, mode='original'):
        super(BatchMixupLayer, self).__init__()

        assert isinstance(alpha, float) and alpha > 0
        assert isinstance(num_classes, int)
        assert isinstance(prob, float) and 0.0 <= prob <= 1.0
        assert mode in ['original', 'multilabel']

        self.alpha = alpha
        self.num_classes = num_classes
        self.prob = prob
        self.mode = mode

    def mixup(self, img, gt_label):
        if self.mode in ['original']:
            one_hot_gt_label = F.one_hot(
                gt_label, num_classes=self.num_classes
            )
        else:
            one_hot_gt_label = gt_label.float()
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = img.size(0)
        index = torch.randperm(batch_size)

        mixed_img = lam * img + (1 - lam) * img[index, :]
        mixed_gt_label = lam * one_hot_gt_label + \
            (1 - lam) * one_hot_gt_label[index, :]

        return mixed_img, mixed_gt_label

    def __call__(self, img, gt_label):
        return self.mixup(img, gt_label)


class Augment(object):
    def __init__(self, prob, alpha, num_classes, mode='original'):
        super(Augment, self).__init__()
        self.augments = [BatchMixupLayer(alpha, num_classes, prob, mode)]
        if prob < 1:
            self.augments.append(Identity(num_classes, 1 - prob, mode))
        self.aug_probs = [aug.prob for aug in self.augments]

    def __call__(self, img, gt_label):
        random_state = np.random.RandomState(random.randint(0, 2 ** 31 - 1))
        aug = random_state.choice(self.augments, p=self.aug_probs)
        return aug(img, gt_label)


class GradientReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class LabelSmoothLoss(nn.Module):
    def __init__(
        self, label_smooth_val, num_classes=None,
        mode=None, reduction='mean'
    ):
        super(LabelSmoothLoss, self).__init__()
        assert reduction in ['mean', 'sum', 'none'], \
            'Invalid Reduction Method'
        assert mode in ['classy_vision', 'original', 'multilabel'], \
            "Invalid LabelSmoothLoss Mode"

        self.reduction = reduction
        self.mode = mode
        self.num_classes = num_classes
        self.label_smooth_val = label_smooth_val
        self._eps = label_smooth_val if mode != 'classy_vision'\
            else label_smooth_val / (1 + label_smooth_val)
        if mode == 'multilabel':
            self.smooth_label = self.multilabel_smooth_label
        else:
            self.smooth_label = self.original_smooth_label

    def generate_one_hot_like_label(self, label):
        if label.dim() == 1 or (label.dim() == 2 and label.shape[1] == 1):
            label = convert_to_one_hot(label.view(-1, 1), self.num_classes)
        return label.float()

    def original_smooth_label(self, one_hot_like_label):
        assert self.num_classes > 0
        smooth_label = one_hot_like_label * (1 - self._eps)
        smooth_label += self._eps / self.num_classes
        return smooth_label

    def multilabel_smooth_label(self, one_hot_like_label):
        assert self.num_classes > 0
        smooth_label = torch.full_like(one_hot_like_label, self._eps)
        smooth_label.masked_fill_(one_hot_like_label > 0, 1 - self._eps)
        return smooth_label

    def _ce(self, x, y, reduction='mean'):
        assert x.shape == y.shape
        x = torch.log_softmax(x, dim=-1)
        result = -torch.sum(x * y, dim=-1)
        if reduction == 'mean':
            return result.mean()
        elif reduction == 'sum':
            return result.sum()
        elif reduction == 'none':
            return result
        else:
            raise NotImplementedError


    def forward(self, cls_score, label):
        if self.num_classes is not None:
            assert self.num_classes == cls_score.shape[1], \
                f'num_classes should equal to cls_score.shape[1], ' \
                f'but got num_classes: {self.num_classes} and ' \
                f'cls_score.shape[1]: {cls_score.shape[1]}'
        else:
            self.num_classes = cls_score.shape[1]

        one_hot_like_label = self.generate_one_hot_like_label(label=label)
        smoothed_label = self.smooth_label(one_hot_like_label)
        if self.mode == 'multilabel':
            return F.binary_cross_entropy_with_logits(
                cls_score, smoothed_label, reduction=self.reduction
            )
        else:
            return self._ce(
                cls_score, smoothed_label, reduction=self.reduction
            )

'''
Adapted from SRGNN 
github: https://github.com/GentleZhu/Shift-Robust-GNNs
'''
def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    return (x1-x2).norm(p=2)

def moment_diff(sx1, sx2, k):
    """
    difference between moments
    """
    ss1 = sx1.pow(k).mean(0)
    ss2 = sx2.pow(k).mean(0)
    #ss1 = sx1.mean(0)
    #ss2 = sx2.mean(0)
    return l2diff(ss1,ss2)

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)

def KMM(X,Xtest,_A=None, _sigma=1e1,beta=0.2):

    H = torch.exp(- 1e0 * pairwise_distances(X)) + torch.exp(- 1e-1 * pairwise_distances(X)) + torch.exp(- 1e-3 * pairwise_distances(X))
    f = torch.exp(- 1e0 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(X, Xtest))
    z = torch.exp(- 1e0 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(Xtest, Xtest))
    H /= 3
    f /= 3
    MMD_dist = H.mean() - 2 * f.mean() + z.mean()

    nsamples = X.shape[0]
    f = - X.shape[0] / Xtest.shape[0] * f.matmul(torch.ones((Xtest.shape[0],1)))
    G = - np.eye(nsamples)
    _A = _A[~np.all(_A==0, axis=1)]
    b = _A.sum(1)
    h = - beta * np.ones((nsamples,1))

    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False
    sol=solvers.qp(matrix(H.numpy().astype(np.double)), matrix(f.numpy().astype(np.double)), matrix(G), matrix(h), matrix(_A), matrix(b))
    return np.array(sol['x']), MMD_dist.item()

def cmd(X, X_test, K=5):
    """
    central moment discrepancy (cmd)
    objective function for keras models (theano or tensorflow backend)

    - Zellinger, Werner, et al. "Robust unsupervised domain adaptation for
    neural networks via moment alignment.", TODO
    - Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning.", ICLR, 2017.
    """
    x1 = X
    x2 = X_test
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = l2diff(mx1,mx2)
    scms = [dm]
    for i in range(K-1):
        # moment diff of centralized samples
        scms.append(moment_diff(sx1,sx2,i+2))
        #scms+=moment_diff(sx1,sx2,1)
    return sum(scms)

class Graph_Editer(nn.Module):
    '''
    Adapted from EERM 
    github: https://github.com/qitianwu/GraphOOD-EERM
    '''
    def __init__(self, K, n, device):
        super(Graph_Editer, self).__init__()
        self.B = nn.Parameter(torch.FloatTensor(K, n, n))
        self.device = device

    def reset_parameters(self):
        nn.init.uniform_(self.B)

    def forward(self, edge_index, n, num_sample, k, train_idx):
        Bk = self.B[k]
        maxn = train_idx.max()
        A = torch.zeros(n, n, dtype=torch.int).to(self.device)
        A[edge_index[0],edge_index[1]] = 1
        # A = to_dense_adj(edge_index)[0].to(torch.int)
        # A = to_dense_adj(edge_index)[0].to(torch.int)[train_idx][:,train_idx]
        A_c = torch.ones(n, n, dtype=torch.int).to(self.device) - A
        P = torch.softmax(Bk, dim=0)
        S = torch.multinomial(P, num_samples=num_sample)  # [n, s]
        M = torch.zeros(n, n, dtype=torch.float).to(self.device)
        col_idx = torch.arange(0, n).unsqueeze(1).repeat(1, num_sample)
        M[S, col_idx] = 1.
        C = A + M * (A_c - A)
        edge_index = dense_to_sparse(C)[0]

        log_p = torch.sum(
            torch.sum(Bk[S, col_idx], dim=1) - torch.logsumexp(Bk, dim=0)
        )

        return edge_index, log_p

class Baseline(nn.Module):
    def __init__(self, d, c, dataset, args, device, train_env_num=None, multilabel=False):
        super(Baseline, self).__init__()
        if args.encoder == 'gcn':
            if args.dataset == 'proteins':
                # Pre-compute GCN normalization.
                dataset.edge_index = normalize(
                    dataset.edge_index)
                self.encoder = GCN(in_channels=d,
                                    hidden_channels=args.hidden_channels,
                                    out_channels=args.hidden_channels,
                                    num_layers=args.num_layers,
                                    dropout=args.dropout,
                                    use_bn=args.use_bn)
            else:
                self.encoder = GCN(in_channels=d,
                                    hidden_channels=args.hidden_channels,
                                    out_channels=args.hidden_channels,
                                    num_layers=args.num_layers,
                                    dropout=args.dropout,
                                    use_bn=args.use_bn)
        elif args.encoder == 'mlp':
            self.encoder = MLP(in_channels=d,
                               hidden_channels=args.hidden_channels,
                               out_channels=args.hidden_channels,
                               num_layers=args.num_layers,
                               dropout=args.dropout)
        elif args.encoder == 'sgc':
            # if args.cached:
            self.encoder = SGC(in_channels=d,
                                out_channels=args.hidden_channels,
                                hops=args.hops)
            # else:
            #     self.encoder = SGCMem(in_channels=d,
            #                     out_channels=args.hidden_channels,
            #                     hops=args.hops)
        elif args.encoder == 'gat':
            self.encoder = GAT(d, args.hidden_channels, args.hidden_channels,
                               num_layers=args.num_layers,
                               dropout=args.dropout, use_bn=args.use_bn,
                               heads=args.gat_heads, out_heads=args.out_heads)
        elif args.encoder == 'gprgnn':
            self.encoder = GPRGNN(d, args.hidden_channels, args.hidden_channels,
                                alpha=args.gpr_alpha)
        elif args.encoder == 'appnp':
            self.encoder = APPNP_Net(d, args.hidden_channels, args.hidden_channels,
                                alpha=args.gpr_alpha)
        elif args.encoder == 'mixhop':
            self.encoder = MixHop(d, args.hidden_channels, args.hidden_channels,
                                num_layers=args.num_layers,
                                dropout=args.dropout,
                                hops=args.hops)
        elif args.encoder == 'gcnjk':
            self.encoder = GCNJK(d, args.hidden_channels, args.hidden_channels,
                                num_layers=args.num_layers,
                                dropout=args.dropout,
                                jk_type=args.jk_type)
        elif args.encoder == 'gatjk':
            self.encoder = GATJK(d, args.hidden_channels, args.hidden_channels,
                                num_layers=args.num_layers,
                                dropout=args.dropout,
                                heads=args.gat_heads,
                                jk_type=args.jk_type)
        elif args.encoder == 'h2gcn':
            self.encoder = H2GCN(d, args.hidden_channels, args.hidden_channels,
                                dataset.edge_index,
                                dataset.num_nodes,
                                num_layers=args.num_layers,
                                dropout=args.dropout,
                                num_mlp_layers=args.num_mlp_layers)
        else:
            raise NotImplementedError
        self.predictor = nn.Linear(args.hidden_channels, c)
        self.device = device

        if args.method == 'irm':
            self.irm_lambda = args.irm_lambda
        if args.method == 'coral':
            self.coral_penalty_weight = args.coral_penalty_weight

        if args.method == 'groupdro':
            assert (train_env_num is not None), "Missing train_env_num"
            self.train_env_num = train_env_num
            self.group_weights_step_size = args.groupdro_step_size
            self.group_weights = torch.ones(train_env_num) / train_env_num

        if args.method == 'dann':
            assert (train_env_num is not None), "Missing train_env_num"
            self.aux_head = nn.Linear(args.hidden_channels, train_env_num)
            self.dann_alpha = args.dann_alpha

        if args.method == 'mixup':
            self.augment = Augment(
                args.mixup_prob, args.mixup_alpha, c,
                mode='multilabel' if multilabel else 'original'
            )

        if args.method == 'eerm':
            self.n = dataset.train_idx.shape[0]
            self.num_sample = args.num_sample
            self.gl = Graph_Editer(args.env_K, self.n, device)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.predictor.reset_parameters()
        if hasattr(self, 'aux_head'):
            self.aux_head.reset_parameters()
        if hasattr(self, 'group_weights'):
            self.group_weights = torch.ones(self.train_env_num)\
                / self.train_env_num

    def forward(self, x, edge_index):
        return self.predictor(self.encoder(x, edge_index))

    def srgnn_preprocess(self, d, num, beta):
        test_idx = torch.cat([d.valid_idx, d.test_in_idx], dim=0)
        train_idx = d.train_idx
        perm_test = torch.randperm(test_idx.shape[0])
        perm_train = torch.randperm(train_idx.shape[0])
        test_idx = test_idx[perm_test[:num]]
        train_idx = train_idx[perm_train[:num]]
        A = to_dense_adj(d.edge_index)[0].to(torch.float)
        Z_train = torch.FloatTensor(A[train_idx])
        Z_test = torch.FloatTensor(A[test_idx])

        #embed()
        label_balance_constraints = np.zeros((d.y.max().item()+1, len(train_idx)))
        for i, idx in enumerate(train_idx):
            label_balance_constraints[d.y[idx], i] = 1

        #embed()
        self.kmm_weight, MMD_dist = KMM(Z_train, Z_test, label_balance_constraints, beta=beta)

    def loss_mixup(self, d, criterion, args):
        feats = self.encoder(d.x, d.edge_index)[d.train_idx]
        feats, target = self.augment(feats, d.y[d.train_idx].squeeze(1).long())
        logits = self.predictor(feats)
        sup_loss = criterion(logits, target)
        return sup_loss

    def loss_eerm(self, d, criterion, args):
        train_idx = d.train_idx
        x,y,edge_index = d.x, d.y.squeeze(1).long(), d.edge_index
        Loss, Log_p = [], 0
        for k in range(args.env_K):
            edge_index_k, log_p = self.gl(d.train_edge_reindex, self.n, self.num_sample, k, train_idx)
            feats = self.encoder(x[train_idx], edge_index_k)
            logits = self.predictor(feats)
            sup_loss = self.sup_loss_calc(d.y[train_idx], logits, criterion, args)
            #sup_loss = criterion(logits, y[train_idx])
            Loss.append(sup_loss.view(-1))
            Log_p += log_p
        Loss = torch.cat(Loss, dim=0)
        Var, Mean = torch.var_mean(Loss)
        return Var, Mean, Log_p


    def loss_srgnn(self, d, criterion, args):
        iid_train = torch.cat([d.valid_idx, d.test_in_idx], dim=0)
        train_idx = d.train_idx
        feats = self.encoder(d.x, d.edge_index)
        logits = self.predictor(feats[train_idx])

        sup_loss = self.sup_loss_calc(d.y[train_idx], logits, criterion, args)
        srgnn_output = args.srgnn_alpha * cmd(feats[train_idx], feats[iid_train])

        loss = (torch.Tensor(self.kmm_weight).reshape(-1).to(self.device) * (sup_loss)).mean() + srgnn_output
        return loss

    def loss_compute(self, d, criterion, args):
        if args.method == 'mixup':
            return self.loss_mixup(d, criterion, args)

        if args.method == 'eerm':
            return self.loss_eerm(d, criterion, args)

        if args.method == 'srgnn':
            return self.loss_srgnn(d, criterion, args)

        train_idx = d.train_idx
        feats = self.encoder(d.x, d.edge_index)[train_idx]
        logits = self.predictor(feats)

        if args.method == 'irm':
            scale = nn.Parameter(torch.tensor(1.)).to(logits.device)
            logits = logits * scale

        sup_loss = self.sup_loss_calc(d.y[train_idx], logits, criterion, args)

        if args.method == 'erm':
            loss = sup_loss
        elif args.method == 'irm':
            _, group_indices, _ = split_into_groups(d.env[train_idx])
            main_loss, irm_penaltys = [], []
            for igroup in group_indices:
                group_loss = sup_loss[igroup]
                main_loss.append(group_loss.mean())
                irm_penaltys.append(self.irm_penalty(group_loss, scale))
            loss = torch.vstack(main_loss).mean() + \
                self.irm_lambda * torch.vstack(irm_penaltys).mean()

        elif args.method == 'coral':
            unique_group, group_indices, _ = split_into_groups(d.env[train_idx])
            coral_penalty = []
            n_group_per_batch = unique_group.numel()
            for igroup in range(n_group_per_batch):
                for jgroup in range(igroup + 1, n_group_per_batch):
                    coral_penalty.append(self.coral_penalty(
                        feats[group_indices[igroup]],
                        feats[group_indices[jgroup]]
                    ))
            loss = sup_loss + self.coral_penalty_weight * \
                torch.vstack(coral_penalty).mean()

        elif args.method == 'groupdro':
            group_loss = torch_scatter.scatter(
                sup_loss, d.env[train_idx].long(), dim=0,
                dim_size=self.train_env_num, reduce='mean'
            )
            if len(group_loss.shape) > 1:
                group_loss = group_loss.mean(dim=-1)
            if self.group_weights.device != group_loss.device:
                self.group_weights = self.group_weights.to(group_loss.device)
            self.group_weights = self.group_weights * \
                torch.exp(self.group_weights_step_size * group_loss.data)
            self.group_weights = self.group_weights / self.group_weights.sum()
            loss = group_loss @ self.group_weights

        elif args.method == 'dann':
            _feat = GradientReverseLayerF.apply(feats, self.dann_alpha)
            env_cls_score = self.aux_head(_feat)
            aux_loss = F.cross_entropy(
                env_cls_score, d.env[train_idx].long())
            loss = sup_loss + aux_loss

        elif args.method == 'reg':
            logits = self.forward(d.x, d.edge_index)[d.train_idx]
            y = d.y[d.train_idx]
            sup_loss = self.sup_loss_calc(y, logits, criterion, args)
            reg_loss_ = []
            for i in range(args.num_aug_branch):
                edge_index_i = rewiring(d.edge_index.clone(), d.batch, args.modify_ratio, type=args.rewiring_type)
                logits_i = self.forward(d.x, edge_index_i)[d.train_idx]
                reg_loss_i = self.sup_loss_calc(y, logits_i, criterion, args)
                reg_loss_.append(reg_loss_i)
            var = torch.var(torch.stack(reg_loss_))
            loss = sup_loss + args.reg_weight * var
        elif args.method == 'dropedge':
            edge_index_new = rewiring(d.edge_index.clone(), d.batch, ratio=args.modify_ratio, type='delete')
            logits = self.forward(d.x, edge_index_new)[d.train_idx]
            loss = self.sup_loss_calc(d.y[train_idx], logits, criterion, args)
        else:
            raise NotImplementedError

        return loss

    def irm_penalty(self, losses, scale):
        grad1 = agrad(losses[0::2].mean(), [scale], create_graph=True)[0]
        grad2 = agrad(losses[1::2].mean(), [scale], create_graph=True)[0]
        return torch.sum(grad1 * grad2)

    def coral_penalty(self, x, y):
        if x.dim() > 2:
            x = x.view(-1, x.size(-1))
            y = y.view(-1, y.size(-1))

        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def sup_loss_calc(self, y, pred, criterion, args):
        if args.dataset in ('twitch', 'elliptic', 'ppi', 'proteins'):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, y.max() + 1).squeeze(1)
            else:
                true_label = y
            loss = criterion(pred, true_label.squeeze(1).to(torch.float))
        else:
            out = F.log_softmax(pred, dim=1)
            target = y.squeeze(1)
            loss = criterion(out, target)
        return loss