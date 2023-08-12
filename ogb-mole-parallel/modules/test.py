import random
import torch


def to_block(inputs, n_nodes):
    '''
    input: (N, n_col), n_nodes: (B)
    '''
    feat_list = []
    cnt = 0
    for n in n_nodes:
        feat_list.append(inputs[cnt: cnt + n])
        cnt += n
    blocks = torch.block_diag(*feat_list)

    return blocks  # (N, n_col*B)


def unpack_block(inputs, n_col, n_nodes):
    '''
    input: (N, B*n_col), n_col: int, n_nodes: (B)
    '''
    feat_list = []
    cnt = 0
    start_col = 0
    for n in n_nodes:
        feat_list.append(inputs[cnt:cnt + n, start_col:start_col + n_col])
        cnt += n
        start_col += n_col

    return torch.cat(feat_list, dim=0)  # (N, n_col)


def batch_repeat(inputs, n_col, n_nodes):
    '''
    input: (B*n_col), n_col: int, n_nodes: (B)
    '''
    x_list = []
    cnt = 0
    for n in n_nodes:
        x = inputs[cnt:cnt + n_col].repeat((n, 1))  # (n, n_col)
        x_list.append(x)
        cnt += n_col

    return torch.cat(x_list, dim=0)


def make_batch_mask(n_nodes, device='cpu'):
    max_node = n_nodes.max().item()
    mask = torch.zeros(len(n_nodes), max_node)
    for idx, nx in enumerate(n_nodes):
        mask[idx, :nx] = 1
    return mask.bool().to(device), max_node


def make_batch(n_nodes, device='cpu'):
    x = []
    for idx, ns in enumerate(n_nodes):
        x.extend([idx] * ns)
    return torch.LongTensor(x).to(device)


def to_pad(feat, mask, max_node, batch_size):
    model_dim = feat.shape[-1]
    new_feat = torch.zeros((batch_size, max_node, model_dim)).to(feat)
    new_feat[mask] = feat
    return new_feat


def full_attention_conv(
    qs, ks, vs, kernel, n_nodes, block_wise=False, output_attn=False
):
    if kernel == 'simple':
        if block_wise:
            # normalize input
            qs = qs / torch.norm(qs, p=2, dim=1, keepdim=True)  # (N, D)
            ks = ks / torch.norm(ks, p=2, dim=1, keepdim=True)  # (N, D)

            # common vars
            device = qs.device
            node_mask, max_node = make_batch_mask(n_nodes, device)
            batch_size, batch = len(n_nodes), make_batch(n_nodes, device)

            # numerator

            q_pad = to_pad(qs, node_mask, max_node, batch_size)  # [B, M, D]
            k_pad = to_pad(ks, node_mask, max_node, batch_size)  # [B, M, D]
            v_pad = to_pad(vs, node_mask, max_node, batch_size)  # [B, M, D]
            qk_pad = torch.matmul(q_pad, k_pad.transpose(2, 1))  # [B, M, M]

            v_sum = torch.zeros((batch_size, vs.shape[-1])).to(device)
            v_idx = batch.unsqueeze(-1).repeat(1, vs.shape[-1])
            v_sum.scatter_add_(dim=0, index=v_idx, src=vs)

            numerator = torch.matmul(qk_pad, v_pad)[node_mask] + \
                torch.index_select(v_sum, dim=0, index=batch)

            denominator = qk_pad[node_mask].sum(dim=-1)  # [N]
            denominator += torch.index_select(
                n_nodes.float(), dim=0, index=batch
            )

            attn_output = numerator / denominator.unsqueeze(dim=-1)

        else:
            # normalize input
            qs = qs / torch.norm(qs, p=2, dim=1, keepdim=True)  # (N, D)
            ks = ks / torch.norm(ks, p=2, dim=1, keepdim=True)  # (N, D)
            N = qs.shape[0]

            # numerator
            kvs = torch.einsum("lm,ld->md", ks, vs)
            attention_num = torch.einsum("nm,md->nd", qs, kvs)  # [N, D]
            all_ones = torch.ones([vs.shape[0]]).to(vs.device)
            vs_sum = torch.einsum("l,ld->d", all_ones, vs)  # [D]
            # [N, D]
            attention_num += vs_sum.unsqueeze(0).repeat(vs.shape[0], 1)

            # denominator
            all_ones = torch.ones([ks.shape[0]]).to(ks.device)
            ks_sum = torch.einsum("lm,l->m", ks, all_ones)
            attention_normalizer = torch.einsum("nm,m->n", qs, ks_sum)  # [N]

            # attentive aggregated results
            attention_normalizer = torch.unsqueeze(
                attention_normalizer, len(attention_normalizer.shape)
            )  # [N, 1]
            attention_normalizer += torch.ones_like(attention_normalizer) * N
            attn_output = attention_num / attention_normalizer  # [N, D]

            # compute attention for visualization if needed
            if output_attn:
                attention = torch.einsum("nhm,lhm->nlh", qs, ks) \
                    / attention_normalizer  # [N, L, H]
    else:
        raise NotImplementedError(f'Not Implemented for kernel {kernel}')

    if output_attn:
        attn = None
        return attn_output, attn
    else:
        return attn_output


def full_attention_conv_v1(
    qs, ks, vs, kernel, n_nodes, block_wise=False, output_attn=False
):
    '''
    qs: query tensor [N, D]
    ks: key tensor [N, D]
    vs: value tensor [N, D]
    n_nodes: num of nodes per graph [B]

    return output [N, D]
    '''
    if kernel == 'simple':

        if block_wise:
            # normalize input
            qs = qs / torch.norm(qs, p=2, dim=1, keepdim=True)  # (N, D)
            ks = ks / torch.norm(ks, p=2, dim=1, keepdim=True)  # (N, D)

            # numerator
            q_block = to_block(qs, n_nodes)  # (N, B*D)
            k_block_T = to_block(ks, n_nodes).T  # (B*D, N)
            v_block = to_block(vs, n_nodes)  # (N, B*D)
            kv_block = torch.matmul(k_block_T, v_block)  # (B*M, B*D)
            qkv_block = torch.matmul(q_block, kv_block)  # (N, B*D)
            qkv = unpack_block(qkv_block, qs.shape[1], n_nodes)  # (N, D)

            v_sum = v_block.sum(dim=0)  # (B*D,)
            v_sum = batch_repeat(v_sum, vs.shape[1], n_nodes)  # (N, D)
            numerator = qkv + v_sum  # (N, D)

            # denominator
            one_list = []
            for n in n_nodes:
                one = torch.ones((n, 1))
                one_list.append(one)
            one_block = torch.block_diag(*one_list).to(qs.device)
            k_sum_block = torch.matmul(k_block_T, one_block)  # (B*D, B)
            denom_block = torch.matmul(q_block, k_sum_block)  # (N, B)
            denominator = unpack_block(denom_block, 1, n_nodes)  # (N, 1)
            denominator += batch_repeat(n_nodes, 1, n_nodes)  # (N, 1)

            attn_output = numerator / denominator  # (N, D)

        else:
            # normalize input
            qs = qs / torch.norm(qs, p=2, dim=1, keepdim=True)  # (N, D)
            ks = ks / torch.norm(ks, p=2, dim=1, keepdim=True)  # (N, D)
            N = qs.shape[0]

            # numerator
            kvs = torch.einsum("lm,ld->md", ks, vs)
            attention_num = torch.einsum("nm,md->nd", qs, kvs)  # [N, D]
            all_ones = torch.ones([vs.shape[0]]).to(vs.device)
            vs_sum = torch.einsum("l,ld->d", all_ones, vs)  # [D]
            # [N, D]
            attention_num += vs_sum.unsqueeze(0).repeat(vs.shape[0], 1)

            # denominator
            all_ones = torch.ones([ks.shape[0]]).to(ks.device)
            ks_sum = torch.einsum("lm,l->m", ks, all_ones)
            attention_normalizer = torch.einsum("nm,m->n", qs, ks_sum)  # [N]

            # attentive aggregated results
            attention_normalizer = torch.unsqueeze(
                attention_normalizer, len(attention_normalizer.shape)
            )  # [N, 1]
            attention_normalizer += torch.ones_like(attention_normalizer) * N
            attn_output = attention_num / attention_normalizer  # [N, D]

            # compute attention for visualization if needed
            if output_attn:
                attention = torch.einsum("nhm,lhm->nlh", qs, ks) \
                    / attention_normalizer  # [N, L, H]

    if output_attn:
        attn = None
        return attn_output, attn
    else:
        return attn_output


for x in range(20):
    dim = random.randint(5, 50)
    BS = random.randint(5, 50)
    N_nodes = [random.randint(5, 40) for _ in range(BS)]
    sum_nodes = sum(N_nodes)

    N_nodes = torch.LongTensor(N_nodes)

    x = torch.randn(sum_nodes, dim)
    y = torch.randn(sum_nodes, dim)
    z = torch.randn(sum_nodes, dim)

    res1 = full_attention_conv(x, y, z, 'simple', N_nodes, True)
    res2 = full_attention_conv_v1(x, y, z, 'simple', N_nodes, True)

    if not torch.allclose(res1, res2, atol=1e-5):
        print(res1, '\n', res2)
