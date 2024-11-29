from .pointnet_util import index_points, square_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


'''Enhanced transformer'''
class Enhanced_transformer(nn.Module):
    def __init__(self, in_points) -> None:
        super().__init__()
        dropout = 0.2,
        self.norm1 = nn.LayerNorm(in_points)
        self.norm2 = nn.LayerNorm(in_points)
        self.drop_out = nn.Dropout(0.2)    # if dropout > 0. else nn.Identity(),
        # self.transformer = TransformerBlock_pct(in_points)
        self.transformer = TransformerBlock_1(in_points)
        self.mlp = nn.Sequential(
            nn.Linear(in_points, in_points),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(in_points, in_points),
            nn.Dropout(0.2),
        )

    # xyz_list[-1] = [B,N,C], x = [B,C,N]
    def forward(self, xyz, x):   # xyz = [8,16,3], x = [8,1024,16]
        x_res1 = x                    # [8,1024,16]    [B,C,N]
        # x = x.permute(0,2,1)          # [8,1024,16] --> [8,16,1024]
        x = self.norm1(x)             # [8,1024,16]
        x = self.transformer(xyz,x.permute(0, 2, 1))   # [8,1024,16]
        x = self.drop_out(x)          # [8,1024,16]
        x = x + x_res1                # [8,1024,16]
        x_res2 = x                    # [8,1024,16]
        # x = x.permute(0,2,1)          # [8,16,1024]
        x = self.norm2(x)             # [8,1024,16]
        x = self.mlp(x)               # [8,1024,16]
        x = self.drop_out(x)          # [8,16,1024]
        x = x + x_res2          # [8,1024,16]

        # x = x + self.drop_out(self.TransformerBlock_Hengshuang(self.norm1(x)))
        # x = x + self.drop_out(self.mlp(self.norm2(x)))
        # x = x.permute(0, 2, 1)

        return x



class TransformerBlock_Hengshuang(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(nn.Linear(3, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.fc_gamma = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)

        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res


'''Menghao'''
class TransformerBlock_pct(nn.Module):
    def __init__(self, channels):
        super(TransformerBlock_pct, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)   #  // 4
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)   #  // 4
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, xyz, x):
        # xyz_list[-1] = [B, N, C], x = [B, N, C]
        # b, n, c    # xyz = [8,4096,3]     # x = [8,16,1024]   #将要卷积的维度放在第二维
        # x = x + xyz
        x = x.permute(0, 2, 1)     #   [8,16,1024]   单独跑时加上
        x_q = self.q_conv(x)    #  [8,16,1024]
        # b, c, n
        x_k = self.k_conv(x).permute(0, 2, 1)       #  [8,1024,16]
        x_v = self.v_conv(x).permute(0, 2, 1)        #  [8,1024,16]
        energy = torch.bmm(x_q, x_k)   # b, n, n     # [8,16,1024]*[8,1024,16] --> [8,16,16]
        attention = self.softmax(energy)             # [8,16,16]
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))          # [8,16,16]
        x_r = torch.bmm(x_v, attention)      # b, c, n         #  [8,1024,16]*[8,16,16] --> [8,1024,16]
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r.permute(0, 2, 1))))    #  [8,16,1024]
        x = (x + x_r).permute(0, 2, 1)              #  [8,1024,16]
        return x


'''Menghao'''
class TransformerBlock_1(nn.Module):
    def __init__(self, channels):
        super(TransformerBlock_1, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)   #  // 4
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)   #  // 4
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels // 4, channels, 1)
        self.trans_conv1 = nn.Conv1d(channels // 4, channels, 1)
        self.act = nn.GELU()
        self.after_norm = nn.BatchNorm1d(channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, xyz, x):
        # b, n, c    # xyz = [8,16,3] [B, N, C]     # x = [8,16,1024] [B, N, C]   #将要卷积的维度放在第二维
        # x = x.permute(0, 2, 1)    #   [8,16,1024]   单独跑时加上
        x_q = self.q_conv(x)    #  [8,256,16]   [8,4,1024]
        # b, c, n
        x_k = self.k_conv(x).permute(0, 2, 1)        # [8,16,256]   [8,1024,4]
        x_v = self.v_conv(x).permute(0, 2, 1)        # [8,16,1024]  [8,1024,16]
        energy = torch.bmm(x_q, x_k)   # b, n, n     # [8,256,16]*[8,16,256] --> [8,256,256]    [8,4,4]
        attention = self.trans_conv(energy)          # [8,1024,256]       [8,16,4]
        attention = self.act(attention)
        attention = self.trans_conv1(attention.permute(0, 2, 1))      #  [8,16,16]
        attention = self.softmax(attention)
        x = torch.bmm(x_v, attention)      # b, c, n         #  [8,1024,16]*[8,16,16] --> [8,1024,16]

        return x


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        attn = q @ k.transpose(-1, -2)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-1)
        output = attn @ v

        return output, attn


'''Nico'''
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model_q, d_model_kv, d_k, d_v):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model_q, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model_kv, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model_kv, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model_q, bias=False)

        self.attention = Attention()

        self.layer_norm1 = nn.LayerNorm(n_head * d_v, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model_q, eps=1e-6)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        b_size, n_q, n_k = q.size(0), q.size(1), k.size(1)

        residual = q

        # Pass through the pre-attention projection: b x k x (n*dv)
        # Separate different heads: b x k x n x dv
        q = self.w_qs(q).view(-1, n_q, n_head, d_k)
        k = self.w_ks(k).view(-1, n_k, n_head, d_k)
        v = self.w_vs(v).view(-1, n_k, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # get b x n x k x dv
        q, attn = self.attention(q, k, v)

        # b x k x ndv
        q = q.transpose(1, 2).contiguous().view(b_size, n_q, -1)
        s = self.layer_norm1(residual + q)
        res = self.layer_norm2(s + self.fc(s))

        return res, attn