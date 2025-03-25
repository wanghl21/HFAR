import numpy as np
import torch
from torch import nn, sub
import torch.nn.functional as F
import os
import math
import torch.optim as optim
from process_utils import *
npratio=1
max_len = 50

class AttentionPooling(nn.Module):
    def __init__(self, d_h, hidden_size, drop_rate):
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size // 2)
        self.att_fc2 = nn.Linear(hidden_size // 2, 1)
        self.drop_layer = nn.Dropout(p=drop_rate)
        torch.nn.init.constant_(self.att_fc2.weight, 0.)


    def forward(self, x, attn_mask=None):

        bz = x.shape[0]
        #print(bz)
        e = self.att_fc1(x) # (bz, seq_len, 200)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e) # (bz, seq_len, 1)
        alpha = torch.exp(alpha)

        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)

        x = torch.bmm(x.permute(0, 2, 1), alpha)
        #print(x)
        x = torch.reshape(x, (bz, -1)) # (bz, 400)
        #x = torch.reshape(x, (bz,)) # (bz, 400)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True)  + 1e-8)
        
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model # 300
        self.n_heads = n_heads # 20
        self.d_k = d_k # 20
        self.d_v = d_v # 20
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads) # 300, 400
        self.W_K = nn.Linear(d_model, d_k * n_heads) # 300, 400
        self.W_V = nn.Linear(d_model, d_v * n_heads) # 300, 400
        
        self._initialize_weights()
                
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                
    def forward(self, Q, K, V, attn_mask=None):
        residual, batch_size = Q, Q.size(0)
        
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, max_len, max_len) 
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) 
        
        context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask) 
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) 
        return context
