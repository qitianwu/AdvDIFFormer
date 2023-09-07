import random
import torch



def to_block(inputs, n_nodes):
    '''
    input: (N, H, n_col), n_nodes: (B)
    '''
    blocks = []
    for h in range(inputs.size(1)):
        feat_list = []
        cnt = 0
        for n in n_nodes:
            feat_list.append(inputs[cnt : cnt + n, h])
            cnt += n
        blocks_h = torch.block_diag(*feat_list) # (N, n_col*B)
        blocks.append(blocks_h)
    blocks = torch.stack(blocks, dim=1) # (N, H, n_col*B)
    return blocks

def unpack_block(inputs, n_col, n_nodes):
    '''
    input: (N, H, B*n_col), n_col: int, n_nodes: (B)
    '''
    unblocks = []
    for h in range(inputs.size(1)):
        feat_list = []
        cnt = 0
        start_col = 0
        for n in n_nodes:
            feat_list.append(inputs[cnt:cnt + n, h, start_col:start_col + n_col])
            cnt += n
            start_col += n_col
        unblocks_h = torch.cat(feat_list, dim=0) # (N, n_col)
        unblocks.append(unblocks_h)
    unblocks = torch.stack(unblocks, dim=1) # (N, H, n_col)
    return unblocks

def batch_repeat(inputs, n_col, n_nodes):
    '''
    input: (H, B*n_col), n_col: int, n_nodes: (B)
    '''
    x_list = []
    cnt = 0
    for n in n_nodes:
        x = inputs[:, cnt:cnt + n_col].repeat(n, 1, 1)  # (n, H, n_col)
        x_list.append(x)
        cnt += n_col
    return torch.cat(x_list, dim=0) # [N, H, n_col]

def full_attention_conv(qs, ks, vs, kernel, n_nodes=None, block_wise=False, output_attn=False):
    '''
    qs: query tensor [N, H, D]
    ks: key tensor [N, H, D]
    vs: value tensor [N, H, D]
    n_nodes: num of nodes per graph [B]

    return output [N, H, D]
    '''
    if kernel == 'simple':
        if block_wise:
            # normalize input
            qs = qs / torch.norm(qs, p=2, dim=2, keepdim=True)  # (N, H, D)
            ks = ks / torch.norm(ks, p=2, dim=2, keepdim=True)  # (N, H, D)

            # numerator
            q_block = to_block(qs, n_nodes)  # (N, H, B*D)
            k_block = to_block(ks, n_nodes)  # (N, H, B*D)
            v_block = to_block(vs, n_nodes)  # (N, H, B*D)
            kvs = torch.einsum("lhm,lhd->hmd", k_block, v_block) # [H, B*D, B*D]
            attention_num = torch.einsum("nhm,hmd->nhd", q_block, kvs)  # (N, H, B*D)
            attention_num = unpack_block(attention_num, qs.shape[2], n_nodes)  # (N, H, D)

            vs_sum = v_block.sum(dim=0)  # (H, B*D)
            vs_sum = batch_repeat(vs_sum, vs.shape[2], n_nodes)  # (N, H, D)
            attention_num += vs_sum  # (N, H, D)

            # denominator
            all_ones = torch.ones([ks.shape[0], qs.shape[1]]).to(ks.device).unsqueeze(2)  # [N, H, 1]
            one_block = to_block(all_ones, n_nodes) # [N, H, B]
            ks_sum = torch.einsum("lhm,lhb->hmb", k_block, one_block) # [H, B*D, B]
            attention_normalizer = torch.einsum("nhm,hmb->nhb", q_block, ks_sum)  # [N, H, B]

            attention_normalizer = unpack_block(attention_normalizer, 1, n_nodes)  # (N, H, 1)
            attention_normalizer += batch_repeat(n_nodes.repeat(qs.shape[1], 1), 1, n_nodes)  # (N, 1)

            attn_output = attention_num / attention_normalizer  # (N, D)
        else:
            # normalize input
            qs = qs / torch.norm(qs, p=2, dim=2, keepdim=True)  # (N, H, D)
            ks = ks / torch.norm(ks, p=2, dim=2, keepdim=True)  # (N, H, D)
            N = qs.shape[0]

            # numerator
            kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
            attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
            all_ones = torch.ones([vs.shape[0]]).to(vs.device)
            vs_sum = torch.einsum("l,lhd->hd", all_ones, vs)  # [H, D]
            attention_num += vs_sum.unsqueeze(0).repeat(vs.shape[0], 1, 1)  # [N, H, D]

            # denominator
            all_ones = torch.ones([ks.shape[0]]).to(ks.device) # [N]
            ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
            attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

            # attentive aggregated results
            attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
            attention_normalizer += torch.ones_like(attention_normalizer) * N
            attn_output = attention_num / attention_normalizer  # [N, H, D]

            # compute attention for visualization if needed
            if output_attn:
                attention = torch.einsum("nhm,lhm->nlh", qs, ks) / attention_normalizer  # [N, L, H]

    if output_attn:
        return attn_output, attention
    else:
        return attn_output


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
    n_heads, model_dim = feat.shape[-2:]
    feat_shape = (batch_size, max_node, n_heads, model_dim)
    new_feat = torch.zeros(feat_shape).to(feat)
    new_feat[mask] = feat
    return new_feat


def full_attention_conv_v1(
    qs, ks, vs, kernel, n_nodes=None, block_wise=False,
    output_attn=False
):
    '''
    qs: query tensor [N, H, D]
    ks: key tensor [N, H, D]
    vs: value tensor [N, H, D]
    n_nodes: num of nodes per graph [B]

    return output [N, H, D]
    '''
    if kernel == 'simple':
        if block_wise:
            # normalize input
            qs = qs / torch.norm(qs, p=2, dim=2, keepdim=True)  # (N, H, D)
            ks = ks / torch.norm(ks, p=2, dim=2, keepdim=True)  # (N, H, D)

            # numerator

            device = qs.device
            node_mask, max_node = make_batch_mask(n_nodes, device)
            batch_size, batch = len(n_nodes), make_batch(n_nodes, device)

            q_pad = to_pad(qs, node_mask, max_node, batch_size)  # [B, M, H, D]
            k_pad = to_pad(ks, node_mask, max_node, batch_size)  # [B, M, H, D]
            v_pad = to_pad(vs, node_mask, max_node, batch_size)  # [B, M, H, D]

            kv_pad = torch.einsum('abcd,abce->adce', k_pad, v_pad) 
            # [B, D, H, D]

            (n_heads, v_dim), k_dim = vs.shape[-2:], ks.shape[-1]
            v_sum = torch.zeros((batch_size, n_heads, v_dim)).to(device)
            v_idx = batch.reshape(-1, 1, 1).repeat(1, n_heads, v_dim)
            v_sum.scatter_add_(dim=0, index=v_idx, src=vs)  # [B, H, D]

            numerator = torch.einsum('abcd,adce->abce', q_pad, kv_pad)
            numerator = numerator[node_mask] + v_sum[batch]


            k_sum = torch.zeros((batch_size, n_heads, k_dim)).to(device)
            k_sum.scatter_add_(dim=0, index=v_idx, src=ks) #[B, H, D]
            denominator = torch.einsum('abcd,acd->abc', q_pad, k_sum)
            denominator = denominator[node_mask] + torch.index_select(
                n_nodes.float(), dim=0, index=batch
            ).unsqueeze(dim=-1)

            attn_output = numerator / denominator.unsqueeze(dim=-1) # [N, H, D]




            # qk_pad = torch.einsum('abcd,aecd->abce', q_pad, k_pad)
            # # [B, M, H, M]

            # n_heads, v_dim = vs.shape[-2:]
            # v_sum = torch.zeros((batch_size, n_heads, v_dim)).to(device)
            # v_idx = batch.reshape(-1, 1, 1).repeat(1, n_heads, v_dim)
            # v_sum.scatter_add_(dim=0, index=v_idx, src=vs)  # [B, H, D]

            # numerator = torch.einsum('abcd,adce->abce', qk_pad, v_pad)
            # numerator = numerator[node_mask] + \
            #     torch.index_select(v_sum, dim=0, index=batch)
            # # [N, H, D]

            # denominator = qk_pad[node_mask].sum(dim=-1)  # [N, H]
            # denominator += torch.index_select(
            #     n_nodes.float(), dim=0, index=batch
            # ).unsqueeze(dim=-1)

            # attn_output = numerator / denominator.unsqueeze(dim=-1) # [N, H, D]

        else:
            # normalize input
            qs = qs / torch.norm(qs, p=2, dim=2, keepdim=True)  # (N, H, D)
            ks = ks / torch.norm(ks, p=2, dim=2, keepdim=True)  # (N, H, D)
            N = qs.shape[0]

            # numerator
            kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
            attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
            all_ones = torch.ones([vs.shape[0]]).to(vs.device)
            vs_sum = torch.einsum("l,lhd->hd", all_ones, vs)  # [H, D]
            # [N, H, D]
            attention_num += vs_sum.unsqueeze(0).repeat(vs.shape[0], 1, 1)

            # denominator
            all_ones = torch.ones([ks.shape[0]]).to(ks.device)  # [N]
            ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
            attention_normalizer = torch.einsum(
                "nhm,hm->nh", qs, ks_sum
            )  # [N, H]

            # attentive aggregated results
            attention_normalizer = torch.unsqueeze(
                attention_normalizer, len(attention_normalizer.shape)
            )  # [N, H, 1]
            attention_normalizer += torch.ones_like(attention_normalizer) * N
            attn_output = attention_num / attention_normalizer  # [N, H, D]

            # compute attention for visualization if needed
            if output_attn:
                attention = torch.einsum("nhm,lhm->nlh", qs, ks) /\
                    attention_normalizer  # [N, L, H]

    if output_attn:
        return attn_output, attention
    else:
        return attn_output



def attn_comp(qs, ks, n_nodes=None, block_wise=False):
    if block_wise:
        device = qs.device
        node_mask, max_node = make_batch_mask(n_nodes, device)
        batch_size, batch = len(n_nodes), make_batch(n_nodes, device)
        q_pad = to_pad(qs, node_mask, max_node, batch_size)  # [B, M, H, D]
        k_pad = to_pad(ks, node_mask, max_node, batch_size)  # [B, M, H, D]
        qk_pad = torch.einsum('abcd,aecd->abce', q_pad, k_pad)
        # [B, M, H, M]
        useful_block = []
        for idx, np in enumerate(n_nodes):
            useful_block.append(qk_pad[idx, :np, :, :np] + 1)
        
        N, heads = qs.shape[:2]
        attention_num = torch.zeros((N, N, heads)).to(device)

        for i in range(heads):
            attention_num[:, :, i] = torch.block_diag(*[
                x[:, i, :] for x in useful_block
            ])

        attention_normalizer = attention_num.sum(dim=1, keepdim=True) # [N, 1, H]

    else:
        qks = torch.einsum("nhd,lhd->nlh", qs, ks)  # [N, N, H]
        attention_num = qks + torch.ones_like(qks)  # (N, N, H)
        attention_normalizer = attention_num.sum(dim=1, keepdim=True)  # [N, 1, H]

    return attention_num,  attention_num / attention_normalizer



def attn_comp_old(qs, ks, n_nodes=None, block_wise=False):

    if block_wise:
        q_block = to_block(qs, n_nodes)  # (N, H, B*D)
        k_block = to_block(ks, n_nodes)  # (N, H, B*D)
        qks = torch.einsum("nhd,lhd->nlh", q_block, k_block)  # [N, N, H]
        ones_block = torch.zeros_like(qks)
        cnt = 0
        for n in n_nodes:
            ones_block[cnt:cnt + n, cnt:cnt + n, :] = 1.0
            cnt += n
        attention_num = qks + ones_block  # (N, N, H)
        attention_normalizer = attention_num.sum(dim=1, keepdim=True) # [N, 1, H]

    else:
        qks = torch.einsum("nhd,lhd->nlh", qs, ks)  # [N, N, H]
        attention_num = qks + torch.ones_like(qks)  # (N, N, H)
        attention_normalizer = attention_num.sum(dim=1, keepdim=True)  # [N, 1, H]

    return attention_num, attention_num / attention_normalizer  # [N, N, H]



for x in range(10):
    dim = random.randint(5, 50)
    BS = random.randint(5, 50)
    N_heads = random.randint(5, 10)
    N_nodes = [random.randint(5, 50) for _ in range(BS)]
    sum_nodes = sum(N_nodes)

    N_nodes = torch.LongTensor(N_nodes)

    x = torch.randn(sum_nodes, N_heads, dim)
    y = torch.randn(sum_nodes, N_heads, dim)
    z = torch.randn(sum_nodes, N_heads, dim)


    res1 = full_attention_conv(x, y, z, 'simple', N_nodes, True)
    res2 = full_attention_conv_v1(x, y, z, 'simple', N_nodes, True)
    res3 = full_attention_conv(x, y, z, 'simple', N_nodes, False)

    if not torch.allclose(res1, res2, atol=1e-5):
        print(res1, '\n', res2)

    if not torch.allclose(res1, res3, atol=1e-5):
        print('[INFO] blok_wise and unblock_wise result different')


    aa, res1 = attn_comp(x, y, N_nodes, True)
    bb, res2 = attn_comp_old(x, y, N_nodes, True)

    if not torch.allclose(res1, res2, atol=1e-5):
        # print(aa, '\n', bb)
        print(torch.allclose(aa, bb, atol=1e-5))
