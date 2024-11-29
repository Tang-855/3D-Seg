import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):  #计算两个点之间的欧几里得公式，即距离公式
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C] ，B（batch_size）, N(点的数量), C
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    #将得到的点的索引转换为值
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):  #最远点采样
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]  返回采样后的中心点
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long)    #B = 16，npoint = 1024 ，初始化一个矩阵
    distance = torch.ones(B, N).to(device) * 1e10  #构建距离定义，16 * 4096
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  #定义一个最远点初始化的索引，即随机选择第一个最远点
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  #得到当前采样点的坐标 B * 3
        dist = torch.sum((xyz - centroid) ** 2, -1)  #计算当前采样点与所有其他点的距离
        mask = dist < distance  #选择距离最近的点来更新距离（更新维护这个距离表）
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]  #重新计算得到的最远点索引（在更新的表中选择距离最大的点）
    return centroids   #返回采样的中心点


def query_ball_point(radius, nsample, xyz, new_xyz):  #确定却簇中心的半径，分组
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):  #确定样本点和分组
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):    #主要用在PointNetSetAbstraction
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        # print(xyz.shape)
        if points is not None:
            points = points.permute(0, 2, 1)
        # print(points.shape)

        #形成局部点的group
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        # print(new_points.shape)
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        #以下是pointnet操作：
        #对局部group中的每一个点做MLP操作；
        #利用1x1的2d卷积，相当于把每个group当成一个通道，共n point个通道，
        #对（C+D，nsample）的维度上做逐像素卷积，结果相当于对单个C+D维度做1d卷积
        # print(new_points.shape)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        # print(new_points.shape)

        new_points = torch.max(new_points, 2)[0]
        # print(new_points.shape)
        new_xyz = new_xyz.permute(0, 2, 1)
        # print(new_xyz.shape)
        return new_xyz, new_points

#PointNetSetAbstractionMsg类实现MSG方法的Set Abstraction：
#这里的radius_list输入的是一个list，例如[1,2,3]；
#对不同的半径做ball query，将不同半径下的点云特征保存在new_points_list中，最后再拼接在一起

#PointNetSetAbstraction类是实现普通的Set Abstraction：
#首先通过sample_and_group的操作形成局部group,
#然后对局部group中的每一个点做MLP操作，最后进行局部的最大池化，得到局部的全局特征。
class PointNetSetAbstractionMsg(nn.Module):
    #例如：npoint=1024,radius=[0.05,0.1],nsample=[16,32],in_channel=9,mlp=[[16, 16, 32], [32, 32, 64]],group_all=False
    #npoint=1024：最远点采样中采样的点；
    #radius=[0.05,0.1]：在局部区域内，簇中心点的采样搜索半径；
    #nsample=[16,32]：相应的搜索半径内对应的点；
    #in_channel=9：通道数
    #mlp=[[16, 16, 32], [32, 32, 64]]：每个点上MLP的输出大小
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]  原始信息。xyz，N是输入样本
            points: input points data, [B, D, N]   特征信息RGB ，D=3，N是输入样本
        Return:
            new_xyz: sampled points position data, [B, C, S]   可能要做采样的点，采样之后N变成S
            new_points_concat: sample points feature data, [B, D', S]   在区域中采样、提特征的点，将不同半径提的特征拼接在一起，拼接后D'的值会增大；S要采样的点，不同半径的点对应的采样点不同
        """
        xyz = xyz.permute(0, 2, 1)
        # print(xyz.shape)
        if points is not None:
            points = points.permute(0, 2, 1)   #维度的变换
        # print(points.shape)  #torch.Size([16, 4096, 9])，9代表原始数据的9个特征+3个定义的特征

        B, N, C = xyz.shape
        #print(xyz.shape)
        S = self.npoint  #在总的点中，用最远点采样选择1024个中心点
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))  #最远点采样，把xyz位置信息传进去，是采样后的点
        # print(new_xyz.shape)
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]   #相应的半径对应的圆中的点
            #按照输入的点云数据数据和索引返回索引的点云数据
            group_idx = query_ball_point(radius, K, xyz, new_xyz)  #radius半径, K对应半径的点, xyz原始坐标信息, new_xyz采样点信息，会得到相应的1024个中心点的圆
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)  #拼接点特征数据和点坐标数据
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
                # print(grouped_points.shape)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]维度变换
            # print(grouped_points.shape)

            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]  #卷积核，（16，9，1，1），（16，16，1，1）*2，（16，32，1，1）（9，32，1，1）（32，32，1，1）（32，64，1，1）
                bn = self.bn_blocks[i][j]    #   32，
                grouped_points =  F.relu(bn(conv(grouped_points)))  #最大池化，获得局部区域的全局特征

            # print(grouped_points.shape)
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            # print(new_points.shape)
            new_points_list.append(new_points)  #不同半径下的点云特征特征列表

        new_xyz = new_xyz.permute(0, 2, 1)  #拼接不同半径下的点云特征
        new_points_concat = torch.cat(new_points_list, dim=1)
        # print(new_points_concat.shape)
        return new_xyz, new_points_concat



#Feature Propagation的实现主要是用在分割的时候，做上采样用
#Feature Propagation的实现主要是通过线性差值和MLP完成；
#当点的个数只有一个时，采用repeat直接复制成N个点；
#当点的个数大于一个时，采用线性差值的方式进行上采样；
#拼接上下采样对应点的SA层的特征，再对拼接后的每一个点都做一个MLP
class PointNetFeaturePropagation(nn.Module):   #PointNet特征传播
    def __init__(self, in_channel, mlp):  #例如in_channel=384，mlp=[256,128]
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        # print(xyz1.shape)
        # print(xyz2.shape)

        points2 = points2.permute(0, 2, 1)
        # print(points2.shape)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            #当点的个数只有一个时，采用repeat直接复制成N个点；
            interpolated_points = points2.repeat(1, N, 1)
            # print(interpolated_points.shape)
        else:
            # 当点的个数大于一个时，采用线性差值的方式进行上采样；
            dists = square_distance(xyz1, xyz2)
            # print(dists.shape)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)  #距离越远的点，权重越小
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  #对于每一个点的权重再做一个全局的归一化
            # print(weight.shape)
            # print(index_points(points2, idx).shape)
            #获得插值点
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
            # print(interpolated_points.shape)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)   #拼接上下采样前对应点SA层的特征
        else:
            new_points = interpolated_points
        # print(new_points.shape)

        new_points = new_points.permute(0, 2, 1)
        # print(new_points.shape)
        for i, conv in enumerate(self.mlp_convs):   #对拼接后每一个点都在MLP
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        # print(new_points.shape)
        return new_points   #得到处理后的new_point

