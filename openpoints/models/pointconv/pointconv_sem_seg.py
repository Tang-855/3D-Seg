import torch.nn as nn
import torch.nn.functional as F
from .pointconv_util import PointConvDensitySetAbstraction,PointConvFeaturePropagation


class pointconv(nn.Module):
    def __init__(self):
        super(pointconv, self).__init__()
        # npoint, nsample, in_channel, mlp, bandwidth, group_all
        feature_dim = 6
        num_classes = 2
        self.sa1 = PointConvDensitySetAbstraction(npoint=1024, nsample=32, in_channel=feature_dim + 3, mlp=[32, 32, 64], bandwidth=0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=256, nsample=32, in_channel=64 + 3, mlp=[64, 64, 128],bandwidth=0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=64, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256], bandwidth=0.4, group_all=False)
        self.sa4 = PointConvDensitySetAbstraction(npoint=16, nsample=32, in_channel=256 + 3, mlp=[256, 256, 512], bandwidth=0.8, group_all=False)
        # npoint, nsample, in_channel, out_put, mlp, bandwidth, group_all
        self.fp4 = PointConvFeaturePropagation(npoint=512, nsample=32, in_channel=512+3, out_put=768, mlp=[512, 256], bandwidth=0.8, group_all=False)
        self.fp3 = PointConvFeaturePropagation(npoint=256, nsample=32, in_channel=256+3, out_put=384, mlp=[256, 256], bandwidth=0.4, group_all=False)
        self.fp2 = PointConvFeaturePropagation(npoint=256, nsample=32, in_channel=256+3, out_put=320, mlp=[256, 128], bandwidth=0.2, group_all=False)
        self.fp1 = PointConvFeaturePropagation(npoint=128, nsample=32, in_channel=128+3, out_put=128, mlp=[128, 128, 128], bandwidth=0.1, group_all=False)
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz       # (8,6,4096)
        l0_xyz = xyz[:,:3,:]  # (8,3,4096)

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # l1_xyz=(8,3,1024),l1_points=(8,64,1024)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # l1_xyz=(8,3,256),l1_points=(8,128,256)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # l1_xyz=(8,3,64),l1_points=(8,256,64)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # l1_xyz=(8,3,16),l1_points=(8,512,16)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)   # l1_points=(8,256,64)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)   # l1_points=(8,256,256)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)   # l1_points=(8,128,1024)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)        # l1_points=(8,128,4096)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))      # l1_points=(8,128,4096)
        '''自监督时去掉'''
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)   # (8,2,4096)
        return x
        # return x, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    import  torch
    model = pointconv(21)
    xyz = torch.rand(6, 6, 2048)
    (model(xyz))