import torch
import math
import numpy as np
import torch
from torch import nn


class GATCodeur(torch.nn.Module):
    def __init__(self, n_layers, in_features, n_hidden, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout):
        super(GATCodeur, self).__init__()
        self.in_features = in_features
        self.proj1 = nn.Linear(in_features, ff_dim)
        self.GAT = Encoder(ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout)
        self.encoders = Stacked_Encoder(n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout)
        self.init_dim=in_features


    def forward(self, src, adj_mat, temp=0.1, gumbel=False):
        init_emb=self.proj1(src)[None, :]
        gat_output = self.GAT(init_emb, msk=adj_mat)
        proj=self.encoders(gat_output)
        if gumbel :
            logits = torch.nn.functional.gumbel_softmax(proj / temp, dim=-1)  # we can use gumbel softmax here to force more exploration
        else:
            logits = torch.nn.functional.softmax(proj/temp, dim=-1)
        return logits


class Stacked_Encoder(torch.nn.Module):
    def __init__(self, n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout):
        super(Stacked_Encoder, self).__init__()
        self.encoderlayers = torch.nn.ModuleList(
            [Encoder(ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout) for _ in range(n_layers)])
        self.norm = torch.nn.LayerNorm(emb_dim, eps=1e-6)
        self.final_proj = torch.nn.Linear(in_features=emb_dim, out_features=2)

    def forward(self, src, msk=None):
        if len(src.shape) < 3:
            src = src[None, :]
        for i, encoderlayer in enumerate(self.encoderlayers):
            src = encoderlayer(src, msk)  # [bs, ls, ed]
        return self.final_proj(self.norm(src))


class Encoder(torch.nn.Module):
    def __init__(self, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout):
        super(Encoder, self).__init__()
        self.multihead_attn = MultiHead_Attn(n_heads, emb_dim, qk_dim, v_dim, dropout)
        self.feedforward = FeedForward(emb_dim, ff_dim, dropout)
        self.norm_att = torch.nn.LayerNorm(emb_dim, eps=1e-6)
        self.norm_ff = torch.nn.LayerNorm(emb_dim, eps=1e-6)

    def forward(self, src, msk):
        # NORM
        tmp1 = self.norm_att(src)
        # ATTN over source words
        tmp2 = self.multihead_attn(q=tmp1, k=tmp1, v=tmp1, msk=msk)  # [bs, ls, ed] contains dropout
        # ADD
        tmp = tmp2 + src

        # NORM
        tmp1 = self.norm_ff(tmp)
        #tmp1 = tmp
        # FF
        tmp2 = self.feedforward(tmp1)  # [bs, ls, ed] contains dropout
        # ADD
        z = tmp2 + tmp

        return z


class MultiHead_Attn(torch.nn.Module):
    def __init__(self, n_heads, emb_dim, qk_dim, v_dim, dropout):
        super(MultiHead_Attn, self).__init__()
        self.nh = n_heads
        self.ed = emb_dim
        self.qd = qk_dim
        self.kd = qk_dim
        self.vd = v_dim
        self.WQ = torch.nn.Linear(emb_dim, qk_dim * n_heads)
        self.WK = torch.nn.Linear(emb_dim, qk_dim * n_heads)
        self.WV = torch.nn.Linear(emb_dim, v_dim * n_heads)
        self.WO = torch.nn.Linear(v_dim * n_heads, emb_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, k, v, msk=None):
        # q is [bs, lq, ed]
        # k is [bs, lk, ed]
        # v is [bs, lv, ed]
        # msk is [bs, 1, ls] or [bs, lt, lt]

        if msk is not None:
            msk = msk.unsqueeze(1)  # [bs, 1, 1, ls] or [bs, 1, lt, lt]

        # logging.info('msk = {}'.format(msk.shape))
        bs = q.shape[0]
        lq = q.shape[1]
        lk = k.shape[1]
        lv = v.shape[1]

        assert self.ed == q.shape[2] == k.shape[2] == v.shape[2], (self.ed, q.shape)
        assert lk == lv  # when applied in decoder both refer the source-side (lq refers the target-side)
        Q = self.WQ(q).contiguous().view([bs, lq, self.nh, self.qd]).permute(0, 2, 1,
                                                                             3)  # => [bs,lq,nh*qd] => [bs,lq,nh,qd] => [bs,nh,lq,qd]
        K = self.WK(k).contiguous().view([bs, lk, self.nh, self.kd]).permute(0, 2, 1,
                                                                             3)  # => [bs,lk,nh*kd] => [bs,lk,nh,kd] => [bs,nh,lk,kd]
        V = self.WV(v).contiguous().view([bs, lv, self.nh, self.vd]).permute(0, 2, 1,
                                                                             3)  # => [bs,lv,nh*vd] => [bs,lv,nh,vd] => [bs,nh,lv,vd]
        # Scaled dot-product Attn from multiple Q, K, V vectors (bs*nh*l vectors)
        Q = Q / math.sqrt(self.kd)
        s = torch.matmul(Q, K.transpose(2,
                                        3))  # [bs,nh,lq,qd] x [bs,nh,kd,lk] = [bs,nh,lq,lk] # thanks to qd==kd #in decoder lq are target words and lk are source words

        # logging.info('s = {}'.format(s.shape))

        if msk is not None:
            #s = s.masked_fill(msk == 0, float('-inf'))  # score=-Inf to masked tokens
            s = s.masked_fill(msk == 0, -1000)
        w = torch.nn.functional.softmax(s, dim=-1)  # [bs,nh,lq,lk] (these are the attention weights)
        #import pdb; pdb.set_trace()
        #### we can use relu instead of softmax: w = torch.nn.functional.relu(s)
        w = self.dropout(w)  # [bs,nh,lq,lk]

        z = torch.matmul(w, V)  # [bs,nh,lq,lk] x [bs,nh,lv,vd] = [bs,nh,lq,vd] #thanks to lk==lv
        z = z.transpose(1, 2).contiguous().view([bs, lq, self.nh * self.vd])  # => [bs,lq,nh,vd] => [bs,lq,nh*vd]
        z = self.WO(z)  # [bs,lq,ed]
        return self.dropout(z)


class FeedForward(torch.nn.Module):
    def __init__(self, emb_dim, ff_dim, dropout):
        super(FeedForward, self).__init__()
        self.FF_in = torch.nn.Linear(emb_dim, ff_dim)
        self.FF_out = torch.nn.Linear(ff_dim, emb_dim)
        self.dropout = torch.nn.Dropout(dropout)  # we can remove that

    def forward(self, x):
        tmp = self.FF_in(x)
        tmp = torch.nn.functional.relu(tmp)
        tmp = self.dropout(tmp)
        tmp = self.FF_out(tmp)
        tmp = self.dropout(tmp)
        return tmp


