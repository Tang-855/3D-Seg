
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
#from pointnet2_ops import pointnet2_utils
from openpoints.models.layers import create_grouper,furthest_point_sample
from .transformer import Enhanced_transformer, TransformerBlock_pct, TransformerBlock_1


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'leakyrelu0.2':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
        return nn.ReLU(inplace=True)

# 计算两个点之间的距离
def square_distance(src, dst):
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

# 找到采样后输入点的索引
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
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


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

# 选择局部邻点  Geometric Affine 模块
class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="anchor", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number        分组数量
        :param kneighbors: k-nerighbors     每组的邻点数量
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()

        self.sample_fn = furthest_point_sample

        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel=3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1,channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
        # fps_idx = farthest_point_sample(xyz, self.groups).long()
        fps_idx = self.sample_fn(xyz, self.groups).long()  # [B, npoint]
        #fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        idx = knn_point(self.kneighbors, xyz, new_xyz)   # [B, npoint=1024, k=32]
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k=32, 3]
        '''# 得到Geometric Affine的上面分支一'''
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize =="anchor":
                mean = torch.cat([new_points, new_xyz],dim=-1) if self.use_xyz else new_points     # [B, npoint, d+3]
                '''# 得到Geometric Affine的下面分支二'''
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            '''# 得到权重因子σ'''
            std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)    # [8, 1, 1, 1]
            '''上下两个分支之间相减，再除以权重因子σ'''
            grouped_points = (grouped_points-mean)/(std + 1e-5)    # [B, npoint, k, d+3]
            # 乘以alpha，加上beta
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta    # [B, npoint, k, d+3] [8,256,32,131]

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)   # [B, npoint, k, d+3] [8,256,32,259]
        return new_xyz, new_points


# 一维的Conv、BN和ReLU
class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act )

    def forward(self, x):
        return self.net(x)


# 残差结构的一维的Conv、BN和ReLU
class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion), kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel, kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel, kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels,  blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3+2*channels if use_xyz else 2*channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation) )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)   # [2048, 259, 32]
        '''第一次MLP、BN和ReLU'''
        x = self.transfer(x)      # [2048, 256, 32]
        batch_size, _, _ = x.size()
        '''第一次MLP、BN残差结构之后，再进行ReLU'''
        x = self.operation(x)  # [b, d, k]     # [8192, 128, 32] [2048, 256, 32]
        '''结构图中尚未体现，对残差结构之后，x的最后一维进行池化,再使用view将最后一个为1的维度去掉'''
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)   # [8192, 128, 1] --> [8192, 128]
        '''利用reshape函数将向量维度变为(b, n, c=128)'''
        x = x.reshape(b, n, -1).permute(0, 2, 1)     # [8, 256, 256]
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation))
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(PointNetFeaturePropagation, self).__init__()
        self.fuse = ConvBNReLU1D(in_channel, out_channel, 1, bias=bias)
        self.extraction = PosExtraction(out_channel, blocks, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)


    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        new_points = self.fuse(new_points)
        new_points = self.extraction(new_points)
        return new_points


class PointMLP(nn.Module):
    def __init__(self, num_classes,points=4096, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 256, 128, 128], de_blocks=[2,2,2,2],
                 gmp_dim=64,cls_dim=64, **kwargs):
        super(PointMLP, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = num_classes
        self.points = points
        self.embedding = ConvBNReLU1D(6, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        en_dims = [last_channel]
        ### Building Encoder #####
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel
            en_dims.append(last_channel)

        ### Building transformer #####
        # self.transformer = Enhanced_transformer(16)   # last_channel=1024, out_channel=1024
        # self.transformer = TransformerBlock_pct(16)
        # self.transformer = TransformerBlock_1(16)

        ### Building Decoder #####
        self.decode_list = nn.ModuleList()
        en_dims.reverse()
        de_dims.insert(0,en_dims[0])
        assert len(en_dims) ==len(de_dims) == len(de_blocks)+1
        for i in range(len(en_dims)-1):
            self.decode_list.append(
                PointNetFeaturePropagation(de_dims[i]+en_dims[i+1], de_dims[i+1],
                                           blocks=de_blocks[i], groups=groups, res_expansion=res_expansion,
                                           bias=bias, activation=activation) )

        self.act = get_activation(activation)

        # class label mapping  类别标签映射
        self.cls_map = nn.Sequential(
            ConvBNReLU1D(1, cls_dim, bias=bias, activation=activation),
            ConvBNReLU1D(cls_dim, cls_dim, bias=bias, activation=activation) )
        # global max pooling mapping
        self.gmp_map_list = nn.ModuleList()
        for en_dim in en_dims:
            self.gmp_map_list.append(ConvBNReLU1D(en_dim, gmp_dim, bias=bias, activation=activation))
        self.gmp_map_end = ConvBNReLU1D(gmp_dim*len(en_dims), gmp_dim, bias=bias, activation=activation)

        # # classifier  gmp_dim=64   cls_dim=64    de_dims=[512, 256, 128, 128]
        self.classifier = nn.Sequential(
            nn.Conv1d(gmp_dim+cls_dim+de_dims[-1], 128, 1, bias=bias),    # gmp_dim+cls_dim+de_dims[-1]=256
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.Conv1d(128, num_classes, 1, bias=bias) )
        '''自监督时使用'''
        # self.classifier = nn.Sequential(
        #     nn.Conv1d(gmp_dim + cls_dim + de_dims[-1], 128, 1, bias=bias),  # gmp_dim+cls_dim+de_dims[-1]=256
        #     nn.BatchNorm1d(128),
        #     nn.Dropout())
        self.en_dims = en_dims

    def forward(self, x, norm_plt, cls_label):    # norm_plt = xyz = [8,3,4096]     cls_labelv = [8,1]
        xyz = x.permute(0, 2, 1)              # [8,4096,3]
        x = torch.cat([x,norm_plt],dim=1)     # [8,6,4096]
        x = self.embedding(x)  # B,D,N        # [8,64,4096]

        xyz_list = [xyz]  # [B, N, 3]
        x_list = [x]  # [B, D, N]

        # here is the encoder
        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # [b=8,g=16,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]   [8, 512, 64]
            x = self.pos_blocks_list[i](x)  # [b,d,g]    # [8, 256, 256]-->[8, 512, 64]    xyz = [8, 256, 3]
            xyz_list.append(xyz)  # [8, 4096, 3] --> [8, 1024, 3] --> [8, 256, 3] --> [8, 64, 3] --> [8, 16, 3]
            x_list.append(x)     # [8, 64, 4096] --> [8, 128, 1024] --> [8, 256, 256] --> [8, 512, 64] --> [8, 1024, 16]


        # here is the decoder
        xyz_list.reverse()   # 将列表中顺序倒过来   [8, 16, 3] --> [8, 64, 3] --> [8, 256, 3] --> [8, 1024, 3] --> [8, 4096, 3]   [B,N,C]
        x_list.reverse()     # 将列表中顺序倒过来   [8, 1024, 16] --> [8, 512, 64] --> [8, 256, 256] --> [8, 128, 1024] --> [8, 64, 4096]  [B,C,N]

        '''原始的x是列表的第一行'''
        x = x_list[0]        #   [8, 1024, 16]  [B,C,N]
        '''新的x是经过Enhanced_transformer here is the transformer'''
        # x = self.transformer(xyz_list[0],x)   # xyz_list[-1] = [B,N,C], x = [B,C,N]

        for i in range(len(self.decode_list)):
            x = self.decode_list[i](xyz_list[i+1], xyz_list[i], x_list[i+1],x)    # [8, 1024, 16] --> [8, 512, 64] --> [8, 256, 256] --> [8, 128, 1024] --> [8, 128, 4096]

        # here is the global context    # 提取全局特征
        gmp_list = []   #   [8,64,1] --> [8,64,1] --> [8,64,1] --> [8,64,1] --> [8,64,1]
        for i in range(len(x_list)):
            ''' ConvBNReLU1将x_list中每个tensor的第二个通道维度都MLP为64通道，再利用MaxPooling将第三个通道池化为1'''
            x_pool = self.gmp_map_list[i](x_list[i])
            gmp_list.append(F.adaptive_max_pool1d(x_pool, 1))
        ''' torch.cat将gmp_list中每个tensor沿第二个通道拼接，再利用convBNReLU1将拼接后tensor的第二个通道维度从320变为为64通道'''
        global_context = self.gmp_map_end(torch.cat(gmp_list, dim=1)) # [b, gmp_dim, 1]   #[8,64,1]

        ''' here is the cls_token'''
        '''将cls_label=[8,1]的维度扩大一维变成[8,1,1]，然后利用MLP将特征维度从一维映射到64维'''#  无监督学习时，将分割头去掉
        cls_token = self.cls_map(cls_label.unsqueeze(dim=-1))  # [b, cls_dim, 1]     #   [8,64,1]
        # #x=[8,128,4096]   global_context=[8,64,4096]    cls_token=[8,64,4096]
        x = torch.cat([x, global_context.repeat([1, 1, x.shape[-1]]), cls_token.repeat([1, 1, x.shape[-1]])], dim=1)   # [8,256,4096]
        #  '''softmax按照行或者列来做归一化,log_softmax在softmax的结果上再做多一次log运算'''
        #  '''x指的是输入矩阵,dim指的是归一化的方式，如果为0是对列做归一化，1是对行做归一化'''

        x = self.classifier(x)    # [8,2,4096]
        '''自监督时去掉'''
        x = F.log_softmax(x, dim=1)   # log_softmax计算输出和梯度        # [8,2,4096]
        # print(x.shape)
        # #x = x.permute(0, 2, 1)
        return x


def pointMLP(num_classes, points, **kwargs) -> PointMLP:   # num_classes=2, points=4096
    return PointMLP(num_classes=num_classes, points=points, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 256, 128, 128], de_blocks=[4,4,4,4],
                 gmp_dim=64,cls_dim=64, **kwargs)


if __name__ == '__main__':
    data = torch.rand(2, 3, 2048)
    norm = torch.rand(2, 3, 2048)
    cls_label = torch.rand([2, 16])
    print("===> testing modelD ...")
    model = pointMLP(50)
    out = model(data, cls_label)  # [2,2048,50]
    print(out.shape)
