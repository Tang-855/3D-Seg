import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy as np
import math


class Point_Transformer_partseg(nn.Module):
    def __init__(self, num_class=2):
        super(Point_Transformer_partseg, self).__init__()
        self.num_class = num_class
        self.conv1 = nn.Conv1d(6, 128, kernel_size=1, bias=False)  # normal时=6
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(negative_slope=0.2))

        self.label_conv = nn.Sequential(nn.Conv1d(1, 64, kernel_size=1, bias=False), nn.BatchNorm1d(64), nn.LeakyReLU(negative_slope=0.2))

        self.convs1 = nn.Conv1d(1024 * 3 + 64, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.num_class, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def forward(self, x, cls_label):    # x = [8,3,4096], cls_label = [8,1]
        batch_size, _, N = x.size()
        # print(x.shape)                # x = [8,6,4096]
        # print(cls_label.shape)        # cls_label = [8,1]
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N   [8,128,4096]
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N   [8,128,4096]
        x1 = self.sa1(x)  # B, D, N   [8,128,4096]
        x2 = self.sa2(x1) # B, D, N   [8,128,4096]
        x3 = self.sa3(x2) # B, D, N   [8,128,4096]
        x4 = self.sa4(x3) # B, D, N   [8,128,4096]
        x = torch.cat((x1, x2, x3, x4), dim=1)    # B, D, N   [8,512,4096]
        x = self.conv_fuse(x)                     # B, D, N   [8,1024,4096]
        x_max = x.max(dim=2, keepdim=False)[0]    # B, D      [8,1024]

        x_avg = x.mean(dim=2, keepdim=False)      # B, D      [8,1024]
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)    # B, D, N   [8,1024,4096]
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)    # B, D, N   [8,1024,4096]
        cls_label_one_hot = cls_label.view(batch_size,1,1)    # [8,1,1]
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)   # B, D, N   [8,64,4096]
        x_global_feature = torch.cat((x_max_feature, x_avg_feature, cls_label_feature), 1) # 1024 + 64   # B, D, N   [8,2112,4096]
        x = torch.cat((x, x_global_feature), 1) # 1024 * 3 + 64   # B, D, N   [8,3136,4096]
        x = self.relu(self.bns1(self.convs1(x)))                  # B, D, N   [8,512,4096]
        x = self.dp1(x)                                           # B, D, N   [8,512,4096]
        x = self.relu(self.bns2(self.convs2(x)))                  # B, D, N   [8,256,4096]
        '''   自监督时去掉  '''
        x = self.convs3(x)                                        # B, D, N   [8,2,4096]
        return x



class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c   [8,4096,32]
        x_k = self.k_conv(x)             # b, c, n    [8,32,4096]
        x_v = self.v_conv(x)             # [8,128,4096]
        energy = torch.bmm(x_q, x_k)     # b, n, n   [8,4096,4096]
        attention = self.softmax(energy) # b, n, n   [8,4096,4096]
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))    # [8,4096,4096]
        x_r = torch.bmm(x_v, attention)  # b, c, n   [8,128,4096]
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r))) # b, c, n   [8,128,4096]
        x = x + x_r    # b, c, n   [8,128,4096]
        return x
