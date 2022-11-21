import torch
import math
import numpy as np
import torch
from torch import nn
from labml_nn.graphs.gat import GraphAttentionLayer
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        print("a")
        e = self._prepare_attentional_mechanism_input(Wh)
        print("b")
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)





class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha=0.3, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class GAT(torch.nn.Module):
    """
    ## Graph Attention Network (GAT)

    This graph attention network has two [graph attention layers](index.html).
    """

    def __init__(self, in_features: int, n_hidden: int, n_heads: int, dropout: float):
        """
        * `in_features` is the number of features per node
        * `n_hidden` is the number of features in the first graph attention layer
        * `n_classes` is the number of classes
        * `n_heads` is the number of heads in the graph attention layers
        * `dropout` is the dropout probability
        """
        super().__init__()

        # First graph attention layer where we concatenate the heads
        self.layer1 = GraphAttentionLayer(in_features, n_hidden, n_heads, is_concat=False, dropout=dropout)
        # Activation function after first graph attention layer
        self.activation = nn.ELU()
        # Final graph attention layer where we average the heads
        self.output = nn.Linear(n_hidden, 128)
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        """
        * `x` is the features vectors of shape `[n_nodes, in_features]`
        * `adj_mat` is the adjacency matrix of the form
         `[n_nodes, n_nodes, n_heads]` or `[n_nodes, n_nodes, 1]`
        """
        # Apply dropout to the input
        x = self.dropout(x)
        # First graph attention layer
        x = self.layer1(x, adj_mat)
        # Activation function
        x = self.activation(x)
        # Dropout
        x = self.dropout(x)
        # Output layer (without activation) for logits
        return self.output(x)


class GATCodeur(torch.nn.Module):
    def __init__(self, n_layers, in_features, n_hidden, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout):
        super(GATCodeur, self).__init__()

        # GAT related :
        #self.GAT = GAT(in_features, n_hidden, n_heads, dropout)
        #self.GAT = SpGraphAttentionLayer(in_features, ff_dim, dropout, alpha=0.2, concat=True)
        self.proj1 = nn.Linear(in_features,ff_dim)
        self.GAT = Encoder(ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout)
        self.encoders = Stacked_Encoder(n_layers, ff_dim, n_heads, emb_dim, qk_dim, v_dim, dropout)
        self.init_dim=in_features

        #torch.nn.init.xavier_uniform(self.proj1.weight)

    def forward(self, src, adj_mat):
        init_emb=self.proj1(src)[None, :]
        gat_output = self.GAT(init_emb, msk=adj_mat)
        proj=self.encoders(gat_output)
        #logits = torch.nn.functional.softmax(proj, dim=-1)
        logits = torch.nn.functional.gumbel_softmax(proj, dim=-1)
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
        tmp1 = self.norm_ff(tmp)   # THIS NORM CAUSES A NAN PROBLEM !!!!!
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
        # logging.info('q = {}'.format(q.shape))
        # logging.info('k = {}'.format(k.shape))
        # logging.info('v = {}'.format(v.shape))

        if msk is not None:
            msk = msk.unsqueeze(1)  # [bs, 1, 1, ls] or [bs, 1, lt, lt]

        # logging.info('msk = {}'.format(msk.shape))
        bs = q.shape[0]
        lq = q.shape[1]  ### sequence length of q vectors (length of target sentences)
        lk = k.shape[1]  ### sequence length of k vectors (may be length of source/target sentences)
        lv = v.shape[1]  ### sequence length of v vectors (may be length of source/target sentences)

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


