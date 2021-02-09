import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class ERL_AKT(nn.Module):
    def __init__(self, n_question, n_pid, n_tid, n_fid, n_sd, n_rd, n_xid, n_yid, d_model, n_blocks,
                 kq_same, dropout, model_type, final_fc_dim=512, n_heads=8, d_ff=2048,  l2=1e-5, separate_qa=False):
        super().__init__()

        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.n_tid = n_tid
        self.n_fid = n_fid
        self.n_sd = n_sd
        self.n_rd = n_rd
        self.n_o1id = n_xid
        self.n_o2id = n_yid
        self.l2 = l2
        self.model_type = model_type
        self.separate_qa = separate_qa
        embed_l = d_model
        concat_embed_l = embed_l

        if self.n_pid > 10000:
            self.difficult_param = nn.Embedding(self.n_pid + 1, 1)
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)
        else:
            self.q_embed_diff = nn.Embedding(self.n_pid+1, embed_l)
            self.qa_embed_diff = nn.Embedding(2, embed_l)
            concat_embed_l += embed_l
        if self.n_tid > 0:
            self.t_embed = nn.Embedding(self.n_tid + 1, embed_l)
            self.ta_embed = nn.Embedding(2, embed_l)
            concat_embed_l += embed_l
        if self.n_fid > 0:
            self.f_embed = nn.Embedding(self.n_fid + 1, embed_l)
            self.fa_embed = nn.Embedding(2, embed_l)
            concat_embed_l += embed_l
        if self.n_sd > 0:
            self.sd_embed = nn.Embedding(self.n_sd + 1, embed_l)
            self.sda_embed = nn.Embedding(2, embed_l)
            concat_embed_l += embed_l
        if self.n_rd > 0:
            self.rd_embed = nn.Embedding(self.n_rd + 1, embed_l)
            self.rda_embed = nn.Embedding(2, embed_l)
            concat_embed_l += embed_l
        if self.n_o1id > 0:
            self.o1_embed = nn.Embedding(self.n_o1id + 1, embed_l)
            self.o1a_embed = nn.Embedding(2, embed_l)
            concat_embed_l += embed_l
        if self.n_o2id > 0:
            self.o2_embed = nn.Embedding(self.n_o2id + 1, embed_l)
            self.o2a_embed = nn.Embedding(2, embed_l)
            concat_embed_l += embed_l

        # n_question+1 ,d_model
        self.q_embed = nn.Embedding(self.n_question+1, embed_l)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
        else:
            self.qa_embed = nn.Embedding(2, embed_l)

        self.x_press = nn.Sequential(nn.Linear(concat_embed_l, d_model), nn.Softplus(), nn.Dropout(self.dropout))
        self.y_press = nn.Sequential(nn.Linear(concat_embed_l, d_model), nn.Softplus(), nn.Dropout(self.dropout))
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / n_heads, d_ff=d_ff,
                                kq_same=self.kq_same, model_type=self.model_type)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)
            if p.size(0) == self.n_tid+1 and self.n_tid > 0:
                torch.nn.init.constant_(p, 0.)
            if p.size(0) == self.n_fid+1 and self.n_fid > 0:
                torch.nn.init.constant_(p, 0.)
            if p.size(0) == self.n_sd+1 and self.n_sd > 0:
                torch.nn.init.constant_(p, 0.)
            if p.size(0) == self.n_rd+1 and self.n_rd > 0:
                torch.nn.init.constant_(p, 0.)
            if p.size(0) == self.n_o1id+1 and self.n_o1id > 0:
                torch.nn.init.constant_(p, 0.)
            if p.size(0) == self.n_o2id+1 and self.n_o2id > 0:
                torch.nn.init.constant_(p, 0.)

    def forward(self, q_data, qa_data, target, pid_data=None, t_data=None, f_data=None,
                sd_data=None, rd_data=None, o1_data=None, o2_data=None):
        # Batch First
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  d_model# c_ct
        if self.separate_qa:
            # BS, seqlen, d_model #f_(ct,rt)
            qa_embed_data = self.qa_embed(qa_data)
        else:
            a_data = (qa_data-q_data)//self.n_question  # rt
            # BS, seqlen, d_model # c_ct+ g_rt =e_(ct,rt)
            # print(self.qa_embed(qa_data).size())
            qa_embed_data = self.qa_embed(a_data)+q_embed_data
        r_embed_x = q_embed_data
        r_embed_y = qa_embed_data

        c_reg_loss = 0.
        if self.n_pid > 10000:
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            r_embed_x += pid_embed_data * q_embed_diff_data
            qa_embed_diff_data = self.qa_embed_diff(a_data)
            if self.separate_qa:
                r_embed_y += pid_embed_data * qa_embed_diff_data
            else:
                r_embed_y += pid_embed_data * (qa_embed_diff_data+q_embed_diff_data)
        else:
            q_embed_diff_data = self.q_embed_diff(pid_data)
            qa_embed_diff_data = self.qa_embed_diff(a_data)+q_embed_diff_data
            r_embed_x = torch.cat([r_embed_x, q_embed_diff_data], dim=-1)
            r_embed_y = torch.cat([r_embed_y, qa_embed_diff_data], dim=-1)
        if self.n_tid > 0:
            t_embed_data = self.t_embed(t_data)
            ta_embed_data = self.ta_embed(a_data)+t_embed_data
            p_embed_x = t_embed_data
            p_embed_y = ta_embed_data
        if self.n_fid > 0:
            f_embed_data = self.f_embed(f_data)
            fa_embed_data = self.fa_embed(a_data)+f_embed_data
            p_embed_x = f_embed_data
            p_embed_y = fa_embed_data
        if self.n_tid > 0 and self.n_fid > 0:
            p_embed_x = torch.cat([t_embed_data, f_embed_data], dim=-1)
            p_embed_y = torch.cat([ta_embed_data, fa_embed_data], dim=-1)

        if self.n_sd > 0:
            sd_embed_data = self.sd_embed(sd_data)
            sda_embed_data = self.sda_embed(a_data)+sd_embed_data
            f_embed_x = sd_embed_data
            f_embed_y = sda_embed_data
        if self.n_rd > 0:
            rd_embed_data = self.rd_embed(rd_data)
            rda_embed_data = self.rda_embed(a_data)+rd_embed_data
            f_embed_x = rd_embed_data
            f_embed_y = rda_embed_data
        if self.n_sd > 0 and self.n_rd > 0:
            f_embed_x = torch.cat([sd_embed_data, rd_embed_data], dim=-1)
            f_embed_y = torch.cat([sda_embed_data, rda_embed_data], dim=-1)

        if self.n_o1id > 0:
            o1_embed_data = self.o1_embed(o1_data)
            o1a_embed_data = self.o1a_embed(a_data)+o1_embed_data
            o_embed_x = o1_embed_data
            o_embed_y = o1a_embed_data
        if self.n_o2id > 0:
            o2_embed_data = self.o2_embed(o2_data)
            o2a_embed_data = self.o2a_embed(a_data)+o2_embed_data
            o_embed_x = o2_embed_data
            o_embed_y = o2a_embed_data
        if self.n_o1id > 0 and self.n_o2id > 0:
            o_embed_x = torch.cat([o1_embed_data, o2_embed_data], dim=-1)
            o_embed_y = torch.cat([o1a_embed_data, o2a_embed_data], dim=-1)

        concat_embed_x = r_embed_x
        concat_embed_y = r_embed_y
        if self.n_tid > 0 or self.n_fid > 0:
            concat_embed_x = torch.cat([concat_embed_x, p_embed_x], dim=-1)
            concat_embed_y = torch.cat([concat_embed_y, p_embed_y], dim=-1)
        if self.n_sd > 0 or self.n_rd > 0:
            concat_embed_x = torch.cat([concat_embed_x, f_embed_x], dim=-1)
            concat_embed_y = torch.cat([concat_embed_y, f_embed_y], dim=-1)
        if self.n_o1id > 0 or self.n_o2id > 0:
            concat_embed_x = torch.cat([concat_embed_x, o_embed_x], dim=-1)
            concat_embed_y = torch.cat([concat_embed_y, o_embed_y], dim=-1)

        x_embed = self.x_press(concat_embed_x)
        y_embed = self.x_press(concat_embed_y)

        d_output = self.model(x_embed, y_embed)
        concat_q = torch.cat([d_output, x_embed], dim=-1)
        output = self.out(concat_q)
        labels = target.reshape(-1)
        m = nn.Sigmoid()
        preds = (output.reshape(-1))
        mask = labels > -0.9
        masked_labels = labels[mask].float()
        masked_preds = preds[mask]
        loss = nn.BCEWithLogitsLoss(reduction='none')
        output = loss(masked_preds, masked_labels)
        return output.sum()+c_reg_loss, m(preds), mask.sum()


class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'akt'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks*2)
            ])

    def forward(self, q_embed_data, qa_embed_data):
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        for block in self.blocks_1:  # encode qas
            y = block(mask=1, query=y, key=y, values=y)
        flag_first = True
        for block in self.blocks_2:
            if flag_first:
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False)
                flag_first = False
            else:
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
                flag_first = True
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True)
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        gammas = self.gammas
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad, gammas)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
             math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
        scores_ = scores_ * mask.float().to(device)
        distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1
        position_effect = torch.abs(
            x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen
        # bs, 8, sl, sl positive distance
        dist_scores = torch.clamp(
            (disttotal_scores-distcum_scores)*position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1,8,1,1
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    total_effect = torch.clamp(torch.clamp(
        (dist_scores*gamma).exp(), min=1e-5), max=1e5)
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)
