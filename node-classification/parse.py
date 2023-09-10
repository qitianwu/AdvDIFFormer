from encoders import *
from data_utils import normalize


def parse_method(args, dataset, n, c, d, device):
    if args.method == 'link':
        model = LINK(n, c).to(device)
    elif args.method == 'gcn':
        if args.dataset == 'ogbn-proteins':
            # Pre-compute GCN normalization.
            dataset.graph['edge_index'] = normalize(
                dataset.graph['edge_index'])
            model = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        dropout=args.dropout,
                        use_bn=args.use_bn).to(device)
        else:
            model = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=args.use_bn).to(device)
    elif args.method == 'mlp' or args.method == 'cs':
        model = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                    out_channels=c, num_layers=args.num_layers,
                    dropout=args.dropout).to(device)
    elif args.method == 'sgc':
        if args.cached:
            model = SGC(in_channels=d, out_channels=c,
                        hops=args.hops).to(device)
        else:
            model = SGCMem(in_channels=d, out_channels=c,
                           hops=args.hops).to(device)
    elif args.method == 'gprgnn':
        model = GPRGNN(d, args.hidden_channels, c,
                       alpha=args.gpr_alpha).to(device)
    elif args.method == 'appnp':
        model = APPNP_Net(d, args.hidden_channels, c,
                          alpha=args.gpr_alpha).to(device)
    elif args.method == 'gat':
        model = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                    dropout=args.dropout, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads).to(device)
    elif args.method == 'lp':
        mult_bin = args.dataset == 'ogbn-proteins'
        model = MultiLP(c, args.lp_alpha, args.hops, mult_bin=mult_bin)
    elif args.method == 'mixhop':
        model = MixHop(d, args.hidden_channels, c, num_layers=args.num_layers,
                       dropout=args.dropout, hops=args.hops).to(device)
    elif args.method == 'gcnjk':
        model = GCNJK(d, args.hidden_channels, c, num_layers=args.num_layers,
                      dropout=args.dropout, jk_type=args.jk_type).to(device)
    elif args.method == 'gatjk':
        model = GATJK(d, args.hidden_channels, c, num_layers=args.num_layers,
                      dropout=args.dropout, heads=args.gat_heads,
                      jk_type=args.jk_type).to(device)
    elif args.method == 'h2gcn':
        model = H2GCN(d, args.hidden_channels, c, dataset.graph['edge_index'],
                      dataset.graph['num_nodes'],
                      num_layers=args.num_layers, dropout=args.dropout,
                      num_mlp_layers=args.num_mlp_layers).to(device)
    else:
        raise ValueError('Invalid method')
    return model


def parser_add_main_args(parser):
    # setup and protocol
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--data_dir', type=str,
                        default='../../../OODgraph-GNNSafe/data/')
    parser.add_argument('--syn_type', type=str, choices=['edge', 'block', 'both'], default='edge')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--runs', type=int, default=5,
                        help='number of distinct runs')
    parser.add_argument('--epochs', type=int, default=1500)

    # model network
    parser.add_argument('--method', type=str, default='erm')
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--gat_heads', type=int, default=8,
                        help='attention heads for gat')
    parser.add_argument('--out_heads', type=int, default=1,
                        help='out heads for gat')
    parser.add_argument('--hops', type=int, default=1,
                        help='power of adjacency matrix for certain methods')
    parser.add_argument('--lp_alpha', type=float, default=.1,
                        help='alpha for label prop')
    parser.add_argument('--gpr_alpha', type=float, default=.1,
                        help='alpha for gprgnn')
    parser.add_argument('--jk_type', type=str, default='max', choices=['max', 'lstm', 'cat'],
                        help='jumping knowledge type')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')
    parser.add_argument('--num_mlp_layers', type=int, default=1,
                        help='number of mlp layers in h2gcn')

    #irm
    parser.add_argument('--irm_lambda', default=1,
                        type=float, help='lambda for irm')
    parser.add_argument('--irm_penalty_anneal_iter', default=500,
                        help='the step to reset optimizer in irm')

    parser.add_argument('--dann_alpha', type=float,
                        default=0.2, help='alpha for dann')
    parser.add_argument('--coral_penalty_weight', type=float, default=0.001,
                        help='penalty weight for coral loss')
    parser.add_argument('--groupdro_step_size', type=float, default=0.01,
                        help='step size for groupdro')

    #mixup
    parser.add_argument('--mixup_prob', default=0.4, type=float,
                        help='prob for using batch mixup')
    parser.add_argument('--mixup_alpha', default=0.2, type=float,
                        help='alpha for beta distribution in mixup')
    parser.add_argument('--label_smooth_val', default=0.1, type=float,
                        help='the smooth value for label smooth loss in mixup')

    #srgnn
    parser.add_argument('--srgnn_alpha', type=float,
                        default=1, help='alpha for srgnn')
    parser.add_argument('--kmm_beta', type=float, default=0.2,
                        help='beta for calculating KMM weight.')

    #eerm
    parser.add_argument('--env_K', type=int, default=3,
                        help='num of views for data augmentation')
    parser.add_argument('--T', type=int, default=1,
                        help='steps for graph learner before one step for GNN')
    parser.add_argument('--num_sample', type=int, default=5,
                        help='num of samples for each node with graph edit')
    # parser.add_argument('--beta', type=float, default=2.0,
    #                     help='weight for mean of risks from multiple domains')
    parser.add_argument('--lr_a', type=float, default=0.005,
                        help='learning rate for graph edit model')

    #ours
    parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'sage', 'gat'])
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for residual link')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--use_residual', action='store_true', help='use residual link for each GNN layer')
    parser.add_argument('--use_graph', action='store_true', help='use pos emb')
    parser.add_argument('--use_weight', action='store_true', help='use weight for GNN convolution')
    parser.add_argument('--kernel', type=str, default='simple', choices=['simple', 'sigmoid'])
    parser.add_argument('--use_reg', action='store_true', help='use reg loss')
    parser.add_argument('--reg_weight', type=float, default=10.0, help='weight for variance loss')
    parser.add_argument('--num_aug_branch', type=int, default=3, help='num of branch for augmentation')
    parser.add_argument('--modify_ratio', type=float, default=0.1, help='ratio of edge rewiring for each graph')
    parser.add_argument('--use_block', action='store_true', help='compute all-pair attention within each block')
    parser.add_argument('--rewiring_type', type=str, default='delete', choices=['delete', 'replace'])
    parser.add_argument('--K_order', type=int, default=3, help='order for gcn conv each layer')
    parser.add_argument('--beta', type=float, default=0.5, help='weight for local and global prop')
    parser.add_argument('--theta', type=float, default=1.0, help='weight for identity mapping in Laplacian prop')
    parser.add_argument('--solver', type=str, default='series', choices=['series', 'inverse'])

    # training
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--print_prop', action='store_true',
                        help='print proportions of predicted class')
    parser.add_argument('--save_result', action='store_true', help='save results')

