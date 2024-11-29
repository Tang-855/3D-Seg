# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
from audtorch.metrics.functional import pearsonr
import numpy as np
from openpoints.models.layers import create_grouper,furthest_point_sample
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import pairwise_distances


#   正则化
class TotalCodingRate(nn.Module):
    def __init__(self, bs, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.b = bs
        self.eps = eps

    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""

        p, m = W.shape  # [d, B]
        m = self.b
        I = torch.eye(p, device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def forward(self, X):
        return - self.compute_discrimn_loss(X.T)

# 余弦相似度
class Similarity_Loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass

    def forward(self, z_list, z_avg):
        z_sim = 0
        num_patch = len(z_list)                     # 33
        z_list = torch.stack(list(z_list), dim=0)   # z_list = [33,128]
        z_avg = z_list.mean(dim=0)                  # 128
        z_sim = 0
        for i in range(num_patch):
            z_sim += F.cosine_similarity(torch.unsqueeze(z_list[i], 0), torch.unsqueeze(z_avg,0), dim=1).mean()

        z_sim = z_sim / num_patch
        z_sim_out = z_sim.clone().detach()

        return -z_sim, z_sim_out

# 余弦相似度（groupouter）
class Similarity_Loss_1(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass

    def forward(self, z_list, z_avg):
        z_sim = 0
        num_patch = len(z_list)                     # 33
        z_list = torch.stack(list(z_list), dim=0)   # z_list = [33,128]
        z_avg_1 = z_list.mean(dim=0)                  # 128
        z_sim = 0
        for i in range(num_patch):
            z_sim += F.cosine_similarity(torch.unsqueeze(z_list[i], 0), torch.unsqueeze(z_avg,0), dim=1).mean()

        z_sim = z_sim / num_patch
        z_sim_out = z_sim.clone().detach()

        return -z_sim, z_sim_out

# 引用上面的正则
def cal_TCR(z, criterion, num_patches):
    z_list = z    # z = [33, 128]   num_patches = 33
    loss = 0
    for i in range(num_patches):
        loss += criterion(torch.unsqueeze(z_list[i], 0))
    loss = loss/num_patches
    return loss


# K-Means++聚类——初始化中心点
'''X：进行聚类的矩阵，K：聚类中心点
    返回的是初始中心点的索引'''
def InitialCentroid(x, K):
    '''x：tensor [256,3]'''
    # 在列表或数组中，生成介于0-数组长度（不含）之间的随机浮点数，然后使用uniform将其转换为整数
    # len(x) = 256        x.shape[1] = 3
    c0_idx = int(np.random.uniform(0, x.shape[0]))
    centroid = x[c0_idx,:].reshape(1,-1)  # 选择第一个簇中心    [1, 3]
    k = 1
    n = x.shape[0]
    while k < K:
        d2 = []
        for i in range(n):
            distance = torch.sum((centroid - x[i, :3]) ** 2)    # 按顺序依次从特定向量中减去质心向量，求出subs的平方,最后求出xyz三个轴相加的和
            d2.append(torch.min(distance))

        # ---- 直接选择概率值最大的 ------
        # new_c_idx = np.argmax(d2)
        # ---- 依照概率分布进行选择 -----
        # 计算每个元素的概率
        # prob = np.array(d2) / np.sum(np.array(d2))
        # new_c_idx = np.random.choice(n, p=prob)
        ''' Softmax 函数对输入进行指数运算，对相对较大的值进行放大，同时对相对较小的值进行抑制。这在某些情况下可以带来更显著的差异，使得概率分布更加突出。
            当你强调相对大的值，并在分布中保留更多的差异性时，使用Softmax通常更合适。特别是在分类问题中，Softmax常用于将模型的输出转换为概率分布'''
        # probabilities = torch.nn.functional.softmax(torch.stack(d2), dim=0)
        '''   - 直接除以总和,不引入指数运算，更直观，可能在某些情况下计算效率更高。当只需要一个相对简单的概率分布，并且不强调对相对大小的微小差异时，直接除法方法可能更适用
              - 如果目标是将输出映射为概率分布，而且相对大小的差异，Softmax通常更适用
              - 如果只需要简单的概率分布，不需要强调相对大小的微小差异，直接除法方法可能更简单和直观'''
        probabilities = torch.stack(d2) / torch.stack(d2).sum()

        new_c_idx = torch.argmax(probabilities)                  # 直接选择概率值最大的
        # new_c_idx = torch.multinomial(probabilities, 1).item()     # 从索引范围 [0, n) 中随机选择一个索引，考虑到概率分布
        # w = x[new_c_idx] = [3]        w1 = x[new_c_idx].reshape(1, -1) = [1, 3]
        # centroid = torch.stack([centroid, x[new_c_idx].reshape(1, -1)], dim=0)      # [k,1,3]
        centroid = torch.cat([centroid, x[new_c_idx].reshape(1, -1)], dim=0)      # [k,3]
        k += 1
    centroid_1 = centroid.reshape(K, 1, 3)                                        # [k,1,3]
    return centroid


# 簇分配矩阵的实现，遍历每个样本点，计算其到每个簇中心的聚类，选择较近的簇中心点
'''X = tensor[N(256), C(3)]：输入矩阵, centroid = tensor[K(20), 3]：中心点
    返回的是每个样本点到最近中心点的索引'''
def findClostestCentroids(X, centroid):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    idx = torch.zeros(X.shape[0], dtype=torch.int)     # tensor [256]
    n, m_c, m_x = X.shape[0], centroid.shape[-1], X.shape[-1]  # n 表示样本个数

    for i in range(n):        # 遍历每一个样本点
        # distance = torch.sum((centroid - X[i, :3]) ** 2, dim=1)
        '''考虑加入语义信息的情况     '''
        if m_c != m_x:
            distance = torch.sum((centroid - X[i, :3]) ** 2, dim=1) # [tensor:20]当前点到所有中心点的距离总和
        else:
            distance = torch.sum((centroid - X[i, :]) ** 2, dim=1)  # [tensor:20]当前点到所有中心点的距离总和

        # distance[torch.isnan(distance)] = torch.inf                           # 检测 NaN 值并替换为指定数值
        # idx[i] = torch.where(distance == distance.min())[0][0]    # 找到最小值的第二种方法：选择最小距离所对应的簇索引
        idx[i] = torch.argmin(distance)
        min_value = torch.min(distance)                           # 找到向量中的最小值

    return idx.to(device)


# 簇中心矩阵的实现，重新计算每个簇的中心点。遍历K个簇，然后分别计算每个簇中所有样本点的平均中心
'''X：[256, 3] (输入矩阵), idx：[256] (最近簇中心点以及簇内点的索引), K：20 (聚簇数)
   返回的重新计算后的聚簇中心点的索引'''
def computeCentroids(X, centroids_idx, idx, K, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # n, m = X.shape[0], centroids_idx.shape[-1]                                # n = 256, m = 3
    n, m = X.shape
    sse, similarity, sim, TCR, loss_TCR = 0, 0, 0, 0, 0
    centriod = torch.zeros((K, m), dtype=torch.float32)     # [20, 3]  [20, 131]
    for k in range(K):
        index = torch.where(idx == k)[0]            # 找到索引为k的所有点，一个簇一个簇的分开来计算
        # temp = X[index, :3]
        '''考虑语义信息的情况      '''
        if m == 3:
            temp = X[index, :3]                     # [index, 3] 取出同一个中心点的所有点   每次先取出一个簇中的所有样本
        else:
            temp = X[index, :]                      # [index, 3] 取出同一个中心点的所有点   每次先取出一个簇中的所有样本

        s = torch.sum(temp, axis=0)                 # [3] [x1, y1, z1] 计算该簇所有点按index分别相加XYZ轴的和，如(x1, y1, z1)
        # s = torch.sum(temp, axis=1)               # [index]计算该簇所有点按3分别相加index轴的和，如(index)
        centriod[k, :] = s / index.size()[0]        # index.size()[0] = index
        centriod = centriod.to(device)
        # q, w, e, t = temp[:, :3], centriod[k, :3], centriod[k, :3].unsqueeze(0), (temp[:, :3] - centriod[k, :3].unsqueeze(0)) ** 2
        sse_k = torch.sum((temp[:, :3] - centriod[k, :3].unsqueeze(0)) ** 2)       # 计算sse损失
        sse = sse + sse_k

        # # 计算类间相似度
    #     if len(index) != 0:
    #         for i in range(len(index)):
    #             # q, w, e = len(index), torch.unsqueeze(temp[i, :], 0), torch.unsqueeze(centriod[k, :], 0)
    #             sim += F.cosine_similarity(torch.unsqueeze(temp[i, :], 0), torch.unsqueeze(centriod[k, :], 0), dim=1).mean()
    #             TCR += criterion(torch.unsqueeze(temp[i, :], 0))
    #         TCR = TCR / len(index)
    #         sim = sim / len(index)
    #     else:
    #         TCR = 0
    #         sim = 0
    #
    #     loss_TCR += TCR
    #     similarity += sim
    #
    # loss_TCR = loss_TCR / K
    # similarity = similarity / K

    sse = torch.sqrt(sse)
    centriod = centriod.float()
    rows_nan = torch.any(torch.isnan(centriod), dim=1) | torch.any(torch.isinf(centriod), dim=1)
    rows_no = ~torch.any(torch.isnan(centriod), dim=1) & ~torch.any(torch.isinf(centriod), dim=1)
    indices_nan = torch.where(rows_nan)[0]       # 取出包含这些值的第一维的索引
    indices_no = torch.where(rows_no)[0]         # 取出包含这些值的第一维的索引
    if len(indices_nan) != 0:
        centriod_nan = centriod[indices_nan, :]  # 从centriod向量中取出索引对应的向量
        centriod_no = centriod[indices_no, :]    # 从centriod向量中取出索引对应的向量
        '''用平均点求
        centriod_avg = torch.mean(centriod_no, dim=0)
        centriod_avg = centriod_avg.unsqueeze(0)[:, :3]
        distances = torch.cdist(X[:, :3], centriod_avg).squeeze(-1)
        '''
        distances = torch.cdist(X, centriod_no)
        max_dis = torch.sum(distances, dim=-1)
        farthest_indices = torch.topk(max_dis, k=len(indices_nan)).indices  # 找到距离最远的n个点的索引
        if m == 3:
            centriod[indices_nan, :] = X[farthest_indices][:, :3]               # 找到距离最远的n个点的坐标
        else:
            centriod[indices_nan, :] = X[farthest_indices][:, :]

    return centriod, sse


def silhouette_coefficient(X, labels):
     K = torch.unique(labels).shape[0]  # 找到簇的数量, 找到labels中出现的唯一的数字，即
     s = []
     for k in range(K):                 # 遍历每一个簇
         index = (labels == k)          # 取对应簇所有样本的索引
         x_in_cluster = X[index]        # [k, 131] 取对应簇中的所有样本
         for sample in x_in_cluster:    # [131] 计算每个样本的轮廓系数
             '''# (sample - x_in_cluster)=[k, 131], ((sample - x_in_cluster) ** 2)=[k, 131]'''
             # a = ((sample - x_in_cluster) ** 2).sum(axis=1)   # a=[k(簇中点的数量)] （样本点-簇中所有点）**2，再把xyz三列相加
             a = torch.sum(((sample - x_in_cluster) ** 2),dim=1)
             '''# 将a中每一行的值开均方，再把所有行相加   torch.sqrt(a).sum()=[值]'''
             a = torch.sqrt(a).sum() / (len(a) - 1)  # 去掉当前样本点与当前样本点的组合计数
             nearest_cluster_id = None
             min_dist2 = torch.inf                   # 这个值表示正无穷大
             for c in range(K):                      # 寻找距离当前样本点最近的簇
                 if k == c:
                     continue
                 # centroid = X[labels == c].mean(axis=0)         # [131]
                 centroid = torch.mean(X[labels == c], dim=0) # [131]
                 # dist2 = ((sample - centroid) ** 2).sum()       # [值]
                 dist2 = torch.sum((sample - centroid) ** 2)    # [值]
                 if dist2 < min_dist2:
                     nearest_cluster_id = c
                     min_dist2 = dist2
             x_nearest_cluster = X[labels == nearest_cluster_id]   # [m(该簇中所有的点), 131] 找到离该样本点最近的簇中所有点
             # b = ((sample - x_nearest_cluster) ** 2).sum(axis=1)   # [m(该簇中所有的点)]（样本点-该簇中所有点的距离）**2，再把每一行的XYZ坐标相加
             b = torch.sum(((sample - x_nearest_cluster) ** 2), dim=1)
             b = torch.sqrt(b).mean()
             '''torch.max()：返回输入张量中所有元素的最大值，一个张量则返回该张量中的最大值，两个张量则返回两个张量中对应位置元素的最大值张量'''
             s.append((b - a) / torch.max(a, b))  # s为列表，有256个tensor向量
     s_c = torch.stack(s)                         # 将列表中的张量堆叠成一个新的张量
     s_c[torch.isnan(s_c)] = 0                    # 检测 NaN 值并替换为指定数值
     silhouette = torch.mean(s_c, dim=0)          # 计算平均  silhouette_1 = torch.mean(s_c)  # 计算平均

     return silhouette


def calinski_harabasz(X, labels):
    # X = X[:,:3]
    global b
    n_samples = X.shape[0]                     # 得到样本数，256
    n_clusters = torch.unique(labels).shape[0] # 得到簇数，10
    betw_disp = 0.    # 所有的簇间距离和
    within_disp = 0.  # 所有的簇内距离和
    global_centroid = torch.mean(X, axis=0)    # 全局簇中心 [131]
    for k in range(n_clusters):                # 遍历每一个簇
        x_in_cluster = X[labels == k]          # 取出当前簇中的所有样本     [m(簇内样本点数),131]
        centroid = torch.mean(x_in_cluster, axis=0)  # 计算当前簇的簇中心  [131]
        within_disp += torch.sum((x_in_cluster - centroid) ** 2)  # 每个簇对应簇内距离的总和
        betw_disp += len(x_in_cluster) * torch.sum((centroid - global_centroid) ** 2)  # 每个簇中心到全局中心的距离总和
    if within_disp == 0.:
        calinski = 1.
    else:
        a = betw_disp * (n_samples - n_clusters)
        b = within_disp * (n_clusters - 1.) + 1e-4
        calinski_1 = a / b
        print('betw_disp:', betw_disp, '\nn_samples:', n_samples, '\nn_clusters:', n_clusters, '\nwithin_disp:',
              within_disp, '\na:', a, '\nb:', b)
        print('calinski_1:', calinski_1)
    # calinski = (1. if within_disp == 0. else betw_disp * (n_samples - n_clusters) / (within_disp * (n_clusters - 1.) + 1e-4))
    if within_disp == 0.:
        calinski = 1
    else:
        calinski = a / b
        print('calinski_1_1:', calinski)
        if torch.isnan(calinski).any():  # 判断是否存在 NaN 值
            nan_indices_1 = torch.isnan(calinski)  # 检查是否有 NaN 值
            calinski[nan_indices_1] = 1e-5
        print('calinski_1_2:', calinski)

    return calinski


def davies_bouldin(X, labels):
    # X = X[:, :3]
    n_clusters = torch.unique(labels).shape[0]
    centroids = torch.zeros((n_clusters, len(X[0])), dtype=float)  # 簇中心矩阵
    s_i = torch.zeros(n_clusters)      # 初始化簇内直径向量

    for k in range(n_clusters):        # 遍历每一个簇
        x_in_cluster = X[labels == k]  # 取当前簇中的所有样本 [m, 131]
        centroids[k] = torch.mean(x_in_cluster, axis=0)  # 计算当前簇的簇中心
        centroids_1, x_in_cluster_c = centroids[k].unsqueeze(0), x_in_cluster.cpu()   # 将gpu向量转换到cpu向量
        x_in_cluster_n, centroids_n = x_in_cluster_c.detach().numpy(), centroids_1.detach().numpy()  # 将cpu向量转换为numpy数组
        # centroids[k]=tensor[131], [centroids[k]]=列表[tensor[131]]   pairwise_distances(x_in_cluster_n, centroids_n)=[159]
        if x_in_cluster_n.shape[0] != 0:
            s_i[k] = pairwise_distances(x_in_cluster_n, centroids_n).mean()   # 用pairwise_distances函数求出两个向量组的空间距离，再求均值
    centroids = centroids.detach().numpy()              # tensor[k,131] -->  numpy[10,131]  # 将cpu向量转换为numpy数组
    centroid_distances = pairwise_distances(centroids)  # [K,K]  求出中心点的空间距离
    combined_s_i_j = s_i[:, None] + s_i                 # tensor[K,k]
    centroid_distances[centroid_distances == 0] = torch.inf  # 把centroid_distances中为0的元素变为inf
    centroid_distances = torch.tensor(centroid_distances)    # 将numpy数组转换为cpu向量   tensor[K,k]
    scores = torch.max(combined_s_i_j / centroid_distances, dim=1)     # combined_s_i_j / centroid_distances = tensor[K,k]，再取第2维的最大值
    D_B = torch.mean(scores[0])     # scores[0], scores[1]= tensor[第二维的最大值], tensor[第二维最大值的索引]
    return D_B

# 监督损失  全局损失
class My_super_loss_kmeans2(nn.Module):

    def __init__(self, bs):

        super().__init__()
        self.sample_fn = furthest_point_sample  # 最远点采样
        self.nsample = 32  # KNN分组后每个组点的数量
        group_args = {'NAME': 'knn', 'radius': 0.08, 'nsample': self.nsample,
                      'return_only_idx': False}  # KNN分组'radius': 0.08,半径
        # group_args = {'NAME': 'ballquery', 'radius': 0.08,  'nsample': self.nsample, 'return_only_idx': False}   # 球半径分组'radius': 0.08,半径
        self.grouper = create_grouper(group_args)

        self.w = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda()
        self.w.data.fill_(0.75).cuda()

        self.downsample = None

        self.b = bs  # 8
        self.n = None  # 4096
        self.c = None  # 128
        self.contractive_loss = Similarity_Loss()

        self.K = 5  # K-Means聚类的中心点数量
        self.max_iter = 100  # K-Means聚类的最大迭代次数
        self.tolerance = 0.001  # 簇内误方差SSE的阈值，小于该阈值时，迭代中止

        self.criterion = TotalCodingRate(self.b, eps=0.2)

    def forward(self, logits, logits1, p0first, p0sec, orixyz):
        '''logits:分支一经过网络预测后的语义特征    p0first：分支一不经过网络预测后的位置特征'''
        b, n, c = logits.shape
        # self.b = b
        self.n = n
        self.c = c
        self.downsample = self.n // 16  # 下采样的倍数
        torch.autograd.set_detect_anomaly(True)

        ##########global points loss 计算两个分支的余弦相似度#####################
        globalpointloss = 0
        for index in range(self.b):
            data = logits[index]  # [4096, 128]   [N, C]
            data1 = logits1[index]  # [4096, 128]   [N, C]
            globalpointloss += F.cosine_similarity(data, data1, dim=1).mean()  # 计算两个分支的余弦相似度

        globalpointloss = globalpointloss / self.b
        globalpointloss = -globalpointloss  # -0.1934
        print('globalpointloss:', globalpointloss)

        ##########FPS 下采样 #####################
        idx = self.sample_fn(p0first, self.downsample).long()  # # 最远点采样，下采样后新的索引   [8, 256]    [B, N]      tensor
        idx1 = self.sample_fn(p0sec, self.downsample).long()  # # 最远点采样，下采样后新的索引   [8, 256]    [B, N]      tensor
        new_p = torch.gather(p0first, 1,
                             idx.unsqueeze(-1).expand(-1, -1, 3))  # 下采样后新的坐标     [8,256,3]    [B, N, C]      tensor
        new_p1 = torch.gather(p0sec, 1,
                              idx1.unsqueeze(-1).expand(-1, -1, 3))  # 下采样后新的坐标     [8,256,3]    [B, N, C]      tensor

        logits = logits.permute(0, 2, 1)  # 输入的特征  [8,128,4096]    [B, C, N]        p0first = [8,4096,3]      tensor
        logits1 = logits1.permute(0, 2, 1)  # 输入的特征  [8,128,4096]    [B, C, N]        p0sec = [8,4096,3]        tensor
        fi = torch.gather(logits, -1, idx.unsqueeze(1).expand(-1, logits.shape[1], -1))  # fi b c n   采样后的特征，对应new_p    [8,128,256]    [B, C, N]      tensor
        fi1 = torch.gather(logits1, -1, idx1.unsqueeze(1).expand(-1, logits1.shape[1], -1))  # fi b c n   采样后的特征，对应new_p1   [8,128,256]    [B, C, N]      tensor


        ##########  K-Means聚类，计算类间损失  #####################
        groupinnerloss_k = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
        groupinnerloss_k_1 = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
        groupouterloss = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
        groupouterloss_1 = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
        for index in range(self.b):
            pos = new_p[index]  # [256, 3]    [N, C]
            pos1 = new_p1[index]  # [256, 3]    [N, C]
            fea = fi.permute(0, 2, 1)[index]  # [256, 128]  [N, C]         fi = [8,256,128]    [B, N, C]
            fea1 = fi1.permute(0, 2, 1)[index]  # [256, 128]  [N, C]

            centroids_idx = InitialCentroid(pos, self.K)  # 分支一：[20, 3] 用K-Means++方法初始化选择聚类中心    centroids_idx = [K, 3]([K, 1, 3])
            centroids_idx_1 = InitialCentroid(pos1, self.K)  # 分支二：
            # idx = None                                         # 初始化样本点的索引
            idx = findClostestCentroids(pos, centroids_idx)  # [256] 计算每个样本点的最近簇中心点,返回到中心点的索引
            idx_1 = findClostestCentroids(pos1, centroids_idx_1)  # 分支二：

            centroids_idx, sse_n = computeCentroids(pos, centroids_idx, idx, self.K, self.criterion)  # [20, 3] 重新计算每个簇的样本中心点
            centroids_idx_1, sse_n_1 = computeCentroids(pos1, centroids_idx_1, idx_1, self.K, self.criterion)  # [20, 3] 重新计算每个簇的样本中心点

            new_feature = torch.cat((pos, fea), dim=-1)  # [256, 131] 拼接XYZ特征和feature特征为新的高维特征    [N, C]
            new_feature_1 = torch.cat((pos1, fea1), dim=-1)  # [256, 131] 拼接XYZ特征和feature特征为新的高维特征    [N, C]

            # 分支一：聚类结果，找到该样本中心点在高维特征中的对应的索引
            sse_pre = 0
            for i in range(self.max_iter):
                prev_centers = centroids_idx  # 保存上一轮的簇中心点的数据
                idx = findClostestCentroids(fea, centroids_idx)  # 计算每个样本点的最近簇中心点
                centroids_idx, sse_curr = computeCentroids(fea, centroids_idx, idx, self.K,
                                                                                 self.criterion)  # 重新计算每个簇的样本中心点，返回中心点坐标
                # '中心点'是否在误差范围
                ''' # 只利用前后两个中心点的变化作为sse
                q = (prev_centers - centroids_idx)                  # 计算sse损失
                w = (prev_centers - centroids_idx) ** 2             # 计算sse损失
                sse = torch.sum((prev_centers - centroids_idx) ** 2)  # 计算sse损失
                sse[torch.isnan(sse)] = 0  '''
                if sse_pre is not None and abs(sse_curr - sse_pre) < self.tolerance:
                    break
                sse_pre = sse_curr

            # 分支二：聚类结果，找到该样本中心点在高维特征中的对应的索引
            sse_pre_1 = 0
            for j in range(self.max_iter):
                prev_centers_1 = centroids_idx_1  # 保存上一轮的簇中心点的数据
                idx_1 = findClostestCentroids(fea1, centroids_idx_1)  # 计算每个样本点的最近簇中心点
                centroids_idx_1, sse_curr_1 = computeCentroids(fea1, centroids_idx_1, idx_1,
                                                                                         self.K,
                                                                                         self.criterion)  # 重新计算每个簇的样本中心点，返回中心点坐标
                # '中心点'是否在误差范围
                if sse_pre_1 is not None and abs(sse_curr_1 - sse_pre_1) < self.tolerance:
                    break
                sse_pre_1 = sse_curr_1

            '''  Silhouette Score：衡量每个样本点到其簇内样本的距离与其最近簇结构之间距离的比值
                     比值越小，说明该样本点所在的簇结构与其最近簇结构之间的距离越远，聚类结果越好。(接近1表簇类紧密度高，与相邻簇分离度高[-1,1])
                 Calinski-Harabasz Index：簇间距离与簇内距离的比值，衡量簇类的紧密度与簇间的分离度，取值越高，聚类效果越好，最大化index
                 Davies-Bouldin Index：计算每个簇与最相似簇之间相似度.簇与簇之间的相似度越高（DB指数偏高），
                   簇与簇之间的距离越小（直观上理解只有相距越近的事物才会约相似），聚类结果就越好'''

            '''方法一：利用sklearn，实现计算轮廓系数、C_H系数和D_B系数三个指标函数的代码'''
            new_feature_c_1, idx_c_1 = new_feature_1.cpu(), idx_1.cpu()  # 将gpu向量转为cpu向量上
            new_feature_n_1, idx_n_1 = new_feature_c_1.detach().numpy(), idx_c_1.detach().numpy()  # 将cpu向量转为numpy数组
            new_feature_c, idx_c = new_feature.cpu(), idx.cpu()  # 将gpu向量转为cpu向量上
            new_feature_n, idx_n = new_feature_c.detach().numpy(), idx_c.detach().numpy()  # 将cpu向量转为numpy数组

            fea_c_1, idx_c_1 = fea1.cpu(), idx_1.cpu()  # 将gpu向量转为cpu向量上
            fea_n_1, idx_n_1 = fea_c_1.detach().numpy(), idx_c_1.detach().numpy()  # 将cpu向量转为numpy数组
            fea_c, idx_c = fea.cpu(), idx.cpu()  # 将gpu向量转为cpu向量上
            fea_n, idx_n = fea_c.detach().numpy(), idx_c.detach().numpy()  # 将cpu向量转为numpy数组

            # silhouette = silhouette_score(new_feature_n, idx_n)
            # silhouette[torch.isnan(silhouette)] = 0
            # calinski = calinski_harabasz_score(fea_n, idx_n)
            # davies = davies_bouldin_score(fea_n, idx_n)

            # silhouette_1 = silhouette_score(new_feature_n_1, idx_n_1)
            # calinski_1 = calinski_harabasz_score(new_feature_n_1, idx_n_1)
            # davies_1 = davies_bouldin_score(new_feature_n_1, idx_n_1)

            '''方法二：利用python代码，实现计算轮廓系数、C_H系数和D_B系数三个指标函数的代码'''
            # S_C = silhouette_coefficient(new_feature, idx)  # S_C(-1, 1)，1表示聚类效果最好
            # S_C = 1 - (S_C + 1) / 2  # 将原始轮廓系数做一个线性变换，值越小表示效果越好
            # C_H = calinski_harabasz(fea, idx)  # C_H(0,无穷大)，值越大表示聚类效果越好
            with torch.no_grad():
                C_H = calinski_harabasz(fea, idx)
                C_H = torch.tensor(C_H)
                if torch.isnan(C_H).any():
                    C_H = C_H.detach()
                    nan_indices = torch.isnan(C_H)  # 检查是否有 NaN 值
                    C_H[nan_indices] = 0
                    print('C_H:',C_H)
                    print(f"Skipping iteration {index} because C_H value is NaN")
            # C_H = calinski_harabasz(new_feature, idx)  # C_H(0,无穷大)，值越大表示聚类效果越好
            C_H = 1 / (1 + C_H)  # 将D_B系数做一个线性变换，值越小表示效果越好
            # D_B = davies_bouldin(fea, idx)  # D_B(0,无穷大)，值越小表示聚类效果越好
            # D_B = davies_bouldin(new_feature, idx)  # D_B(0,无穷大)，值越小表示聚类效果越好

            # S_C_1 = silhouette_coefficient(new_feature_1, idx_1)  # S_C(-1, 1)，1表示聚类效果最好
            # S_C_1 = 1 - (S_C_1 + 1) / 2  # 将原始轮廓系数做一个线性变换，值越小表示效果越好
            # C_H_1 = calinski_harabasz(fea1, idx_1)  # C_H(0,无穷大)，值越大表示聚类效果越好
            with torch.no_grad():
                C_H_1 = calinski_harabasz(fea1, idx_1)
                C_H_1 = torch.tensor(C_H_1)
                if torch.isnan(C_H_1).any():
                    C_H_1 = C_H_1.detach()
                    nan_indices_1 = torch.isnan(C_H_1)  # 检查是否有 NaN 值
                    C_H_1[nan_indices_1] = 0
                    print('C_H_1:',C_H_1)
                    print(f"Skipping iteration {index} because C_H value is NaN")
            # C_H_1 = calinski_harabasz(new_feature_1, idx_1)  # C_H(0,无穷大)，值越大表示聚类效果越好
            C_H_1 = 1 / (1 + C_H_1)  # 将D_B系数做一个线性变换，值越小表示效果越好
            # D_B_1 = davies_bouldin(fea1, idx_1)  # D_B(0,无穷大)，值越小表示聚类效果越好
            # D_B_1 = davies_bouldin(new_feature_1, idx_1)  # D_B(0,无穷大)，值越小表示聚类效果越好

            groupouterloss += C_H  # 分支一：在KMeans方法中计算类间相似度
            groupouterloss_1 += C_H_1  # 分支一：在KMeans方法中计算类间相似度

            # groupinnerloss_k += 200 * similarity + 1 * loss_TCR  # 分支一：在KMeans方法中计算类间相似度
            # groupinnerloss_k_1 += 200 * similarity_1 + 1 * loss_TCR_1  # 分支一：在KMeans方法中计算类间相似度

        groupouterloss = ((groupouterloss + groupouterloss_1) / 2) / self.b
        print('groupouterloss:', groupouterloss)
        # groupinnerloss_k = ((groupinnerloss_k + groupinnerloss_k_1) / 2) / self.b
        # print('groupinnerloss_k:', groupinnerloss_k)


        ##########  KNN分组  #####################
        # dp, fj = self.grouper(new_p, p0first, logits)  #fj b c n k    fj    p0first = [8,4096,3]     KNN分组
        # dp1, fj1 = self.grouper(new_p1, p0sec, logits1)  # fj b c n k         KNN分组
        # dp = [8, 3, 256, 32][B,C,N,S]  fj = [8,128,256,32]    [B, C, S, N]
        dp, fj = self.grouper(new_p, p0first, logits)  # fj b c n k      KNN分组
        dp1, fj1 = self.grouper(new_p1, p0sec, logits1)  # fj b c n k      KNN分组
        fi = fi.permute(0, 2, 1)  # [8,256,128]    [B, N, C]
        fj = fj.permute(0, 2, 3, 1)  # [8,256,32,128]    [B, N, S, C]
        fi1 = fi1.permute(0, 2, 1)  # [8,256,128]    [B, N, C]
        fj1 = fj1.permute(0, 2, 3, 1)  # [8,256,32,128]    [B, N, S, C]
        # logits = logits.permute(0, 2, 1)
        # logits1 = logits1.permute(0, 2, 1)

        ##################groupinnerloss  计算KNN分组后的组内损失####################
        groupinnerloss = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
        for index in range(self.b):
            fii = fi[index]  # [256, 128] [N, C]         fi = [8,256,128]    [B, N, C]
            fjj = fj[index]  # [256, 32, 128] [N, S, C]  fj =  [8,256,32,128]    [B, N, S, C]
            fii1 = fi1[index]  # [256, 128] [N, C]
            fjj1 = fj1[index]  # [256, 32, 128] [N, S, C]

            # 两个分支分别进行，KNN聚类后组内的中心点和邻点之间的余弦相似度,共256个分组,每个聚类组中包含33个点
            for pointindex, points in enumerate(fii):
                # pointindex = 0--256  points = 128
                datatemp = torch.unsqueeze(points, 0)  # [1, 128] [N, C]
                datatemp = torch.cat((fjj[pointindex, :, :], datatemp), 0)  # [33, 128] [1+S, C]
                avg = datatemp.mean(0)  # 128
                loss_contract, _ = self.contractive_loss(datatemp, avg)  # 计算两个分支的相似度
                loss_TCR = cal_TCR(datatemp, self.criterion, len(datatemp))  # 计算正则
                groupinnerloss += 200 * loss_contract + 1 * loss_TCR

            for pointindex, points in enumerate(fii1):
                datatemp = torch.unsqueeze(points, 0)
                datatemp = torch.cat((fjj1[pointindex, :, :], datatemp), 0)
                avg = datatemp.mean(0)
                loss_contract, _ = self.contractive_loss(datatemp, avg)
                loss_TCR = cal_TCR(datatemp, self.criterion, len(datatemp))
                groupinnerloss += 200 * loss_contract + 1 * loss_TCR

            groupinnerloss = groupinnerloss / self.downsample

        groupinnerloss = groupinnerloss / self.b
        print('groupinnerloss:', groupinnerloss)

        ##################grouploss####################
        # loss = groupinnerloss + groupouterloss
        loss = globalpointloss + groupinnerloss + groupouterloss
        # loss = globalpointloss + groupinnerloss_k + groupouterloss

        return loss



# 监督损失  全局损失
class My_super_loss_kmeans2_1(nn.Module):

    def __init__(self, bs):

        super().__init__()
        self.sample_fn = furthest_point_sample   # 最远点采样
        self.nsample = 32  # KNN分组后每个组点的数量
        group_args = {'NAME': 'knn', 'radius': 0.08,  'nsample': self.nsample, 'return_only_idx': False}   # KNN分组'radius': 0.08,半径
        # group_args = {'NAME': 'ballquery', 'radius': 0.08,  'nsample': self.nsample, 'return_only_idx': False}   # 球半径分组'radius': 0.08,半径
        self.grouper = create_grouper(group_args)

        self.w = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda()
        self.w.data.fill_(0.75).cuda()

        self.downsample = None

        self.b = bs       # 8
        self.n = None     # 4096
        self.c = None     # 128
        self.contractive_loss = Similarity_Loss()

        self.K = 5              # K-Means聚类的中心点数量
        self.max_iter = 100      # K-Means聚类的最大迭代次数
        self.tolerance  =0.001         # 簇内误方差SSE的阈值，小于该阈值时，迭代中止

        self.criterion = TotalCodingRate(self.b, eps=0.2)

    def forward(self, logits, logits1,p0first,p0sec,orixyz):
        '''logits:分支一经过网络预测后的语义特征    p0first：分支一不经过网络预测后的位置特征'''
        b, n, c = logits.shape
        # self.b = b
        self.n = n
        self.c = c
        self.downsample = self.n // 16   #下采样的倍数
        torch.autograd.set_detect_anomaly(True)
        ##########global points loss 计算两个分支的余弦相似度#####################
        globalpointloss = 0
        for index in range(self.b):
            data = logits[index]      # [4096, 128]   [N, C]
            data1 = logits1[index]    # [4096, 128]   [N, C]
            globalpointloss += F.cosine_similarity(data, data1, dim=1).mean()   # 计算两个分支的余弦相似度

        globalpointloss = globalpointloss / self.b
        globalpointloss = -globalpointloss             # -0.1934
        print('globalpointloss:', globalpointloss)


        ##########FPS 下采样 #####################
        idx = self.sample_fn(p0first, self.downsample).long()   # # 最远点采样，下采样后新的索引   [8, 256]    [B, N]      tensor
        idx1 = self.sample_fn(p0sec, self.downsample).long()    # # 最远点采样，下采样后新的索引   [8, 256]    [B, N]      tensor
        new_p = torch.gather(p0first, 1, idx.unsqueeze(-1).expand(-1, -1, 3))   # 下采样后新的坐标     [8,256,3]    [B, N, C]      tensor
        new_p1 = torch.gather(p0sec, 1, idx1.unsqueeze(-1).expand(-1, -1, 3))   # 下采样后新的坐标     [8,256,3]    [B, N, C]      tensor

        logits = logits.permute(0, 2, 1)        # 输入的特征  [8,128,4096]    [B, C, N]        p0first = [8,4096,3]      tensor
        logits1 = logits1.permute(0, 2, 1)    # 输入的特征  [8,128,4096]    [B, C, N]        p0sec = [8,4096,3]        tensor
        fi = torch.gather(logits, -1, idx.unsqueeze(1).expand(-1, logits.shape[1], -1))      # fi b c n   采样后的特征，对应new_p    [8,128,256]    [B, C, N]      tensor
        fi1 = torch.gather(logits1, -1, idx1.unsqueeze(1).expand(-1, logits1.shape[1], -1))  # fi b c n   采样后的特征，对应new_p1   [8,128,256]    [B, C, N]      tensor


        ##########  K-Means聚类，计算类间损失  #####################
        groupinnerloss_k = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
        groupinnerloss_k_1 = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
        groupouterloss = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
        groupouterloss_1 = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
        for index in range(self.b):
            pos = new_p[index]                                # [256, 3]    [N, C]
            pos1 = new_p1[index]                              # [256, 3]    [N, C]
            fea = fi.permute(0, 2, 1)[index]                    # [256, 128]  [N, C]         fi = [8,256,128]    [B, N, C]
            fea1 = fi1.permute(0, 2, 1)[index]                  # [256, 128]  [N, C]

            centroids_idx = InitialCentroid(pos, self.K)         # 分支一：[20, 3] 用K-Means++方法初始化选择聚类中心    centroids_idx = [K, 3]([K, 1, 3])
            centroids_idx_1 = InitialCentroid(pos1, self.K)      # 分支二：
            # idx = None                                         # 初始化样本点的索引
            idx = findClostestCentroids(pos, centroids_idx)      # [256] 计算每个样本点的最近簇中心点,返回到中心点的索引
            idx_1 = findClostestCentroids(pos1, centroids_idx_1) # 分支二：

            centroids_idx, sse_n, similarity_n, loss_TCR_n = computeCentroids(pos, centroids_idx, idx, self.K, self.criterion)      # [20, 3] 重新计算每个簇的样本中心点
            centroids_idx_1, sse_n_1, similarity_n_1, loss_TCR_n_1 = computeCentroids(pos1, centroids_idx_1, idx_1, self.K, self.criterion)  # [20, 3] 重新计算每个簇的样本中心点

            new_feature = torch.cat((pos,fea),dim=-1)       # [256, 131] 拼接XYZ特征和feature特征为新的高维特征    [N, C]
            new_feature_1 = torch.cat((pos1,fea1),dim=-1)   # [256, 131] 拼接XYZ特征和feature特征为新的高维特征    [N, C]

            # 分支一：聚类结果，找到该样本中心点在高维特征中的对应的索引
            sse_pre = 0
            for i in range(self.max_iter):
                prev_centers = centroids_idx     # 保存上一轮的簇中心点的数据
                idx = findClostestCentroids(fea, centroids_idx)                           # 计算每个样本点的最近簇中心点
                centroids_idx, sse_curr, similarity, loss_TCR = computeCentroids(fea, centroids_idx, idx, self.K, self.criterion)    # 重新计算每个簇的样本中心点，返回中心点坐标
                # '中心点'是否在误差范围
                ''' # 只利用前后两个中心点的变化作为sse
                q = (prev_centers - centroids_idx)                  # 计算sse损失
                w = (prev_centers - centroids_idx) ** 2             # 计算sse损失
                sse = torch.sum((prev_centers - centroids_idx) ** 2)  # 计算sse损失
                sse[torch.isnan(sse)] = 0  '''
                if sse_pre is not None and abs(sse_curr - sse_pre) < self.tolerance:
                    break
                sse_pre = sse_curr

            # 分支二：聚类结果，找到该样本中心点在高维特征中的对应的索引
            sse_pre_1 = 0
            for j in range(self.max_iter):
                prev_centers_1 = centroids_idx_1  # 保存上一轮的簇中心点的数据
                idx_1 = findClostestCentroids(fea1, centroids_idx_1)  # 计算每个样本点的最近簇中心点
                centroids_idx_1, sse_curr_1, similarity_1, loss_TCR_1 = computeCentroids(fea1, centroids_idx_1, idx_1, self.K, self.criterion)  # 重新计算每个簇的样本中心点，返回中心点坐标
                # '中心点'是否在误差范围
                if sse_pre_1 is not None and abs(sse_curr_1 - sse_pre_1) < self.tolerance:
                    break
                sse_pre_1 = sse_curr_1

            '''  Silhouette Score：衡量每个样本点到其簇内样本的距离与其最近簇结构之间距离的比值
                     比值越小，说明该样本点所在的簇结构与其最近簇结构之间的距离越远，聚类结果越好。(接近1表簇类紧密度高，与相邻簇分离度高[-1,1])
                 Calinski-Harabasz Index：簇间距离与簇内距离的比值，衡量簇类的紧密度与簇间的分离度，取值越高，聚类效果越好，最大化index
                 Davies-Bouldin Index：计算每个簇与最相似簇之间相似度.簇与簇之间的相似度越高（DB指数偏高），
                   簇与簇之间的距离越小（直观上理解只有相距越近的事物才会约相似），聚类结果就越好'''

            '''方法一：利用sklearn，实现计算轮廓系数、C_H系数和D_B系数三个指标函数的代码'''
            new_feature_c_1, idx_c_1 = new_feature_1.cpu(), idx_1.cpu()  # 将gpu向量转为cpu向量上
            new_feature_n_1, idx_n_1 = new_feature_c_1.detach().numpy(), idx_c_1.detach().numpy()  # 将cpu向量转为numpy数组
            new_feature_c, idx_c = new_feature.cpu(), idx.cpu()  # 将gpu向量转为cpu向量上
            new_feature_n, idx_n = new_feature_c.detach().numpy(), idx_c.detach().numpy()  # 将cpu向量转为numpy数组

            fea_c_1, idx_c_1 = fea1.cpu(), idx_1.cpu()  # 将gpu向量转为cpu向量上
            fea_n_1, idx_n_1 = fea_c_1.detach().numpy(), idx_c_1.detach().numpy()  # 将cpu向量转为numpy数组
            fea_c, idx_c = fea.cpu(), idx.cpu()  # 将gpu向量转为cpu向量上
            fea_n, idx_n = fea_c.detach().numpy(), idx_c.detach().numpy()  # 将cpu向量转为numpy数组

            # silhouette = silhouette_score(new_feature_n, idx_n)
            # silhouette[torch.isnan(silhouette)] = 0
            # calinski = calinski_harabasz_score(fea_n, idx_n)
            davies = davies_bouldin_score(fea_n, idx_n)

            # silhouette_1 = silhouette_score(new_feature_n_1, idx_n_1)
            # calinski_1 = calinski_harabasz_score(new_feature_n_1, idx_n_1)
            # davies_1 = davies_bouldin_score(new_feature_n_1, idx_n_1)

            '''方法二：利用python代码，实现计算轮廓系数、C_H系数和D_B系数三个指标函数的代码'''
            # S_C = silhouette_coefficient(new_feature, idx)  # S_C(-1, 1)，1表示聚类效果最好
            # S_C = 1 - (S_C + 1) / 2  # 将原始轮廓系数做一个线性变换，值越小表示效果越好
            C_H = calinski_harabasz(fea, idx)  # C_H(0,无穷大)，值越大表示聚类效果越好
            # C_H = calinski_harabasz(new_feature, idx)  # C_H(0,无穷大)，值越大表示聚类效果越好
            C_H = 1 / (1 + C_H)      # 将D_B系数做一个线性变换，值越小表示效果越好
            D_B = davies_bouldin(fea, idx)  # D_B(0,无穷大)，值越小表示聚类效果越好
            # D_B = davies_bouldin(new_feature, idx)  # D_B(0,无穷大)，值越小表示聚类效果越好

            # S_C_1 = silhouette_coefficient(new_feature_1, idx_1)  # S_C(-1, 1)，1表示聚类效果最好
            # S_C_1 = 1 - (S_C_1 + 1) / 2  # 将原始轮廓系数做一个线性变换，值越小表示效果越好
            C_H_1 = calinski_harabasz(fea1, idx_1)  # C_H(0,无穷大)，值越大表示聚类效果越好
            # C_H_1 = calinski_harabasz(new_feature_1, idx_1)  # C_H(0,无穷大)，值越大表示聚类效果越好
            C_H_1 = 1 / (1 + C_H_1)      # 将D_B系数做一个线性变换，值越小表示效果越好
            D_B_1 = davies_bouldin(fea1, idx_1)  # D_B(0,无穷大)，值越小表示聚类效果越好
            # D_B_1 = davies_bouldin(new_feature_1, idx_1)  # D_B(0,无穷大)，值越小表示聚类效果越好

            groupouterloss += C_H   # 分支一：在KMeans方法中计算类间相似度
            groupouterloss_1 += C_H_1  # 分支一：在KMeans方法中计算类间相似度

            groupinnerloss_k += 200 * similarity + 1 * loss_TCR   # 分支一：在KMeans方法中计算类间相似度
            groupinnerloss_k_1 += 200 * similarity_1 + 1 * loss_TCR_1  # 分支一：在KMeans方法中计算类间相似度

        groupouterloss = ((groupouterloss + groupouterloss_1) / 2) / self.b
        print('groupouterloss:', groupouterloss)
        groupinnerloss_k = ((groupinnerloss_k + groupinnerloss_k_1) / 2) / self.b
        print('groupinnerloss_k:', groupinnerloss_k)


        ##########  KNN分组  #####################
        # dp, fj = self.grouper(new_p, p0first, logits)  #fj b c n k    fj    p0first = [8,4096,3]     KNN分组
        # dp1, fj1 = self.grouper(new_p1, p0sec, logits1)  # fj b c n k         KNN分组
        # dp = [8, 3, 256, 32][B,C,N,S]  fj = [8,128,256,32]    [B, C, S, N]
        dp, fj = self.grouper(new_p, p0first, logits)     # fj b c n k      KNN分组
        dp1, fj1 = self.grouper(new_p1, p0sec, logits1)   # fj b c n k      KNN分组
        fi = fi.permute(0, 2, 1)         #     [8,256,128]    [B, N, C]
        fj = fj.permute(0, 2, 3, 1)      #     [8,256,32,128]    [B, N, S, C]
        fi1 = fi1.permute(0, 2, 1)       #     [8,256,128]    [B, N, C]
        fj1 = fj1.permute(0, 2, 3, 1)    #     [8,256,32,128]    [B, N, S, C]
        # logits = logits.permute(0, 2, 1)
        # logits1 = logits1.permute(0, 2, 1)

        ##################grouploss  计算KNN分组后的组内损失####################
        groupinnerloss = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
        for index in range(self.b):
            fii = fi[index]       # [256, 128] [N, C]         fi = [8,256,128]    [B, N, C]
            fjj = fj[index]       # [256, 32, 128] [N, S, C]  fj =  [8,256,32,128]    [B, N, S, C]
            fii1 = fi1[index]     # [256, 128] [N, C]
            fjj1 = fj1[index]     # [256, 32, 128] [N, S, C]

            # 两个分支分别进行，KNN聚类后组内的中心点和邻点之间的余弦相似度,共256个分组,每个聚类组中包含33个点
            for pointindex,points in enumerate(fii):
                # pointindex = 0--256  points = 128
                datatemp = torch.unsqueeze(points, 0)                          # [1, 128] [N, C]
                datatemp = torch.cat((fjj[pointindex, :, :],datatemp),0)       # [33, 128] [1+S, C]
                avg = datatemp.mean(0)     # 128
                loss_contract, _ = self.contractive_loss(datatemp, avg)        # 计算两个分支的相似度
                loss_TCR = cal_TCR(datatemp, self.criterion, len(datatemp))    #计算正则
                groupinnerloss += 200 * loss_contract + 1 * loss_TCR

            for pointindex, points in enumerate(fii1):
                datatemp = torch.unsqueeze(points, 0)
                datatemp = torch.cat((fjj1[pointindex, :, :], datatemp), 0)
                avg = datatemp.mean(0)
                loss_contract, _ = self.contractive_loss(datatemp, avg)
                loss_TCR = cal_TCR(datatemp, self.criterion, len(datatemp))
                groupinnerloss += 200 * loss_contract + 1 * loss_TCR

            groupinnerloss = groupinnerloss / self.downsample

        groupinnerloss = groupinnerloss / self.b
        print('groupinnerloss:', groupinnerloss)


        ##################grouploss####################
        loss = globalpointloss + groupinnerloss + groupouterloss
        # loss = globalpointloss + groupinnerloss_k + groupouterloss

        return loss


# 监督损失  全局损失
class My_super_loss_groupouter(nn.Module):

    def __init__(self, bs):

        super().__init__()
        self.sample_fn = furthest_point_sample   # 最远点采样
        self.nsample = 32  # KNN分组后每个组点的数量
        group_args = {'NAME': 'knn', 'radius': 0.08,  'nsample': self.nsample, 'return_only_idx': False}   # KNN分组'radius': 0.08,半径
        # group_args = {'NAME': 'ballquery', 'radius': 0.08,  'nsample': self.nsample, 'return_only_idx': False}   # 球半径分组'radius': 0.08,半径
        self.grouper = create_grouper(group_args)

        self.w = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda()
        self.w.data.fill_(0.75).cuda()

        self.downsample = None

        self.b = bs       # 8
        self.n = None     # 4096
        self.c = None     # 128
        self.contractive_loss = Similarity_Loss()
        self.contractive_loss_1 = Similarity_Loss_1()
        self.criterion = TotalCodingRate(self.b, eps=0.2)

    def forward(self, logits, logits1,p0first,p0sec,orixyz):
        b, n, c = logits.shape
      #  self.b = b
        self.n = n
        self.c = c
        self.downsample = self.n // 16   #下采样的倍数

        ##########global points loss 计算两个分支的余弦相似度#####################
        globalpointloss = 0
        for index in range(self.b):
            data = logits[index]      # [4096, 128]   [N, C]
            data1 = logits1[index]    # [4096, 128]   [N, C]
            globalpointloss += F.cosine_similarity(data, data1, dim=1).mean()   # 计算两个分支的余弦相似度

        globalpointloss = globalpointloss / self.b
        globalpointloss = -globalpointloss             # -0.1934


        ##########FPS 下采样 #####################

        idx = self.sample_fn(p0first, self.downsample).long()   # # 最远点采样，下采样后新的索引   [8, 256]    [B, N]
        idx1 = self.sample_fn(p0sec, self.downsample).long()    # # 最远点采样，下采样后新的索引   [8, 256]    [B, N]
        new_p = torch.gather(p0first, 1, idx.unsqueeze(-1).expand(-1, -1, 3))   # 下采样后新的坐标     [8,256,3]    [B, N, C]
        new_p1 = torch.gather(p0sec, 1, idx1.unsqueeze(-1).expand(-1, -1, 3))   # 下采样后新的坐标     [8,256,3]    [B, N, C]

        logits = logits.permute(0,2,1)        # 输入的特征  [8,128,4096]    [B, C, N]        p0first = [8,4096,3]
        logits1 = logits1.permute(0, 2, 1)    # 输入的特征  [8,128,4096]    [B, C, N]        p0sec = [8,4096,3]
        fi = torch.gather(logits, -1, idx.unsqueeze(1).expand(-1, logits.shape[1], -1))      # fi b c n   采样后的特征，对应new_p    [8,128,256]    [B, C, N]
        fi1 = torch.gather(logits1, -1, idx1.unsqueeze(1).expand(-1, logits1.shape[1], -1))  # fi b c n   采样后的特征，对应new_p1   [8,128,256]    [B, C, N]
        # dp, fj = self.grouper(new_p, p0first, logits)  #fj b c n k    fj    p0first = [8,4096,3]     KNN分组
        # dp1, fj1 = self.grouper(new_p1, p0sec, logits1)  # fj b c n k         KNN分组
        # dp = [8, 3, 256, 32] [B,C,N,S]      fj = [8,128,256,32] [B, C, S, N]
        dp, fj = self.grouper(new_p, p0first, logits)     # fj b c n k      KNN分组
        dp1, fj1 = self.grouper(new_p1, p0sec, logits1)   # fj b c n k      KNN分组
        fi = fi.permute(0, 2, 1)         #     [8,256,128]    [B, N, C]
        fj = fj.permute(0, 2, 3, 1)      #     [8,256,32,128]    [B, N, S, C]
        fi1 = fi1.permute(0, 2, 1)       #     [8,256,128]    [B, N, C]
        fj1 = fj1.permute(0, 2, 3, 1)    #     [8,256,32,128]    [B, N, S, C]
        # print(fi.shape)
        # print(fi1.shape)
        # logits = logits.permute(0, 2, 1)
        # logits1 = logits1.permute(0, 2, 1)

        ##################  计算KNN分组后的组内损失groupinnerloss  ####################
        groupinnerloss = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
        #groupinnerloss1 = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()


        for index in range(self.b):
            fii = fi[index]       # [256, 128] [N, C]         fi = [8,256,128]    [B, N, C]
            fjj = fj[index]       # [256, 32, 128] [N, S, C]  fj =  [8,256,32,128]    [B, N, S, C]
            fii1 = fi1[index]     # [256, 128] [N, C]
            fjj1 = fj1[index]     # [256, 32, 128] [N, S, C]

            # 两个分支分别进行，KNN聚类后组内的中心点和邻点之间的余弦相似度,共256个分组,每个聚类组中包含33个点
            for pointindex,points in enumerate(fii):
                # pointindex = 0--256  points = 128
                datatemp = torch.unsqueeze(points, 0)                          # [1, 128] [N, C]
                datatemp = torch.cat((fjj[pointindex, :, :],datatemp),0)       # [33, 128] [1+S, C]
                avg = datatemp.mean(0)     # 128
                loss_contract, _ = self.contractive_loss(datatemp, avg)        # 计算两个分支的相似度
                loss_TCR = cal_TCR(datatemp, self.criterion, len(datatemp))    #计算正则
                groupinnerloss += 200 * loss_contract + 1 * loss_TCR

            for pointindex, points in enumerate(fii1):
                datatemp = torch.unsqueeze(points, 0)
                datatemp = torch.cat((fjj1[pointindex, :, :], datatemp), 0)
                avg = datatemp.mean(0)
                loss_contract, _ = self.contractive_loss(datatemp, avg)
                loss_TCR = cal_TCR(datatemp, self.criterion, len(datatemp))
                groupinnerloss += 200 * loss_contract + 1 * loss_TCR

            groupinnerloss = groupinnerloss / self.downsample

        groupinnerloss = groupinnerloss / self.b

        ##################groupouterloss类之间的相似度####################

        groupouterloss = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
        for index in range(self.b):
            fii = fi[index]       # [256, 128] [N, C]         fi = [8,256,128]    [B, N, C]
            fjj = fj[index]       # [256, 32, 128] [N, S, C]  fj =  [8,256,32,128]    [B, N, S, C]
            fii1 = fi1[index]     # [256, 128] [N, C]
            fjj1 = fj1[index]     # [256, 32, 128] [N, S, C]

            # KNN聚类后组内的中心点和最近中心点的邻点之间的余弦相似度,共256个分组,每个聚类组中包含33个点
            for pointindex,points in enumerate(fii):
                # pointindex = 0--256  points = 128
                datatemp = torch.unsqueeze(points, 0)     # [1, 128] [N, C]
                # datatemp = [1, 128]
                # fjj[j, :, :] = [32, 128]      fjj[pointindex, :, :] = [32, 128]
                # fjj[j,:,:].unsqueeze(0) = [1, 32, 128]
                # fjj[pointindex, j] = [128]    fjj[pointindex, j].unsqueeze(0) = [1, 128]

                distances = F.pairwise_distance(datatemp, fii, p=2, keepdim=True)       # 计算 pairwise_distance
                # min_dis_index = torch.argmin(distances, dim=0)                        # 找到最小距离对应的索引

                _, indices = torch.topk(distances, k=1, dim=0, largest=False)           # 找到距离最小对应的索引，K对应要找的是前几位数
                nearest_point = torch.squeeze(fjj[indices[0], :, :],0)                  # 获取在第一个维度上最近的点

                datatemp_1 = torch.cat((nearest_point, datatemp), 0)           # [33, 128] [1+S, C]
                avg = datatemp.mean(0)
                loss_contract, _ = self.contractive_loss_1(nearest_point, datatemp)  # 计算两个分支的相似度
                loss_TCR = cal_TCR(datatemp_1, self.criterion, len(datatemp_1))  # 计算正则
                groupouterloss += 200 * loss_contract + 1 * loss_TCR

            groupouterloss = groupouterloss / self.downsample
            # groupouterloss = groupouterloss / fii[0]

        groupouterloss = groupouterloss / self.b


        ##################grouploss####################
        loss = globalpointloss + groupinnerloss + groupinnerloss
        # loss = globalpointloss + groupinnerloss

        return loss


# 监督损失  全局损失
class My_super_loss(nn.Module):

    def __init__(self, bs):

        super().__init__()
        self.sample_fn = furthest_point_sample   # 最远点采样
        self.nsample = 32  # KNN分组后每个组点的数量
        group_args = {'NAME': 'knn', 'radius': 0.08,  'nsample': self.nsample, 'return_only_idx': False}   # KNN分组'radius': 0.08,半径
        # group_args = {'NAME': 'ballquery', 'radius': 0.08,  'nsample': self.nsample, 'return_only_idx': False}   # 球半径分组'radius': 0.08,半径
        self.grouper = create_grouper(group_args)

        self.w = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda()
        self.w.data.fill_(0.75).cuda()

        self.downsample = None

        self.b = bs       # 8
        self.n = None     # 4096
        self.c = None     # 128
        self.contractive_loss = Similarity_Loss()
        self.contractive_loss_1 = Similarity_Loss_1()
        self.criterion = TotalCodingRate(self.b, eps=0.2)

    def forward(self, logits, logits1,p0first,p0sec,orixyz):
        b, n, c = logits.shape
      #  self.b = b
        self.n = n
        self.c = c
        self.downsample = self.n // 16   #下采样的倍数

        ##########global points loss 计算两个分支的余弦相似度#####################
        globalpointloss = 0
        for index in range(self.b):
            data = logits[index]      # [4096, 128]   [N, C]
            data1 = logits1[index]    # [4096, 128]   [N, C]
            globalpointloss += F.cosine_similarity(data, data1, dim=1).mean()   # 计算两个分支的余弦相似度

        globalpointloss = globalpointloss / self.b
        globalpointloss = -globalpointloss             # -0.1934


        ##########FPS 下采样 #####################

        idx = self.sample_fn(p0first, self.downsample).long()   # # 最远点采样，下采样后新的索引   [8, 256]    [B, N]
        idx1 = self.sample_fn(p0sec, self.downsample).long()    # # 最远点采样，下采样后新的索引   [8, 256]    [B, N]
        new_p = torch.gather(p0first, 1, idx.unsqueeze(-1).expand(-1, -1, 3))   # 下采样后新的坐标     [8,256,3]    [B, N, C]
        new_p1 = torch.gather(p0sec, 1, idx1.unsqueeze(-1).expand(-1, -1, 3))   # 下采样后新的坐标     [8,256,3]    [B, N, C]

        logits = logits.permute(0,2,1)        # 输入的特征  [8,128,4096]    [B, C, N]        p0first = [8,4096,3]
        logits1 = logits1.permute(0, 2, 1)    # 输入的特征  [8,128,4096]    [B, C, N]        p0sec = [8,4096,3]
        fi = torch.gather(logits, -1, idx.unsqueeze(1).expand(-1, logits.shape[1], -1))      # fi b c n   采样后的特征，对应new_p    [8,128,256]    [B, C, N]
        fi1 = torch.gather(logits1, -1, idx1.unsqueeze(1).expand(-1, logits1.shape[1], -1))  # fi b c n   采样后的特征，对应new_p1   [8,128,256]    [B, C, N]
        # dp, fj = self.grouper(new_p, p0first, logits)  #fj b c n k    fj    p0first = [8,4096,3]     KNN分组
        # dp1, fj1 = self.grouper(new_p1, p0sec, logits1)  # fj b c n k         KNN分组
        # dp = [8, 3, 256, 32][B,C,N,S]  fj = [8,128,256,32]    [B, C, S, N]
        dp, fj = self.grouper(new_p, p0first, logits)     # fj b c n k      KNN分组
        dp1, fj1 = self.grouper(new_p1, p0sec, logits1)   # fj b c n k      KNN分组
        fi = fi.permute(0, 2, 1)         #     [8,256,128]    [B, N, C]
        fj = fj.permute(0, 2, 3, 1)      #     [8,256,32,128]    [B, N, S, C]
        fi1 = fi1.permute(0, 2, 1)       #     [8,256,128]    [B, N, C]
        fj1 = fj1.permute(0, 2, 3, 1)    #     [8,256,32,128]    [B, N, S, C]
        # print(fi.shape)
        # print(fi1.shape)
        # logits = logits.permute(0, 2, 1)
        # logits1 = logits1.permute(0, 2, 1)

        ##################grouploss  计算KNN分组后的组内损失####################
        groupinnerloss = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
        #groupinnerloss1 = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()


        for index in range(self.b):
            fii = fi[index]       # [256, 128] [N, C]         fi = [8,256,128]    [B, N, C]
            fjj = fj[index]       # [256, 32, 128] [N, S, C]  fj =  [8,256,32,128]    [B, N, S, C]
            fii1 = fi1[index]     # [256, 128] [N, C]
            fjj1 = fj1[index]     # [256, 32, 128] [N, S, C]

            # 两个分支分别进行，KNN聚类后组内的中心点和邻点之间的余弦相似度,共256个分组,每个聚类组中包含33个点
            for pointindex,points in enumerate(fii):
                # pointindex = 0--256  points = 128
                datatemp = torch.unsqueeze(points, 0)                          # [1, 128] [N, C]
                datatemp = torch.cat((fjj[pointindex, :, :],datatemp),0)       # [33, 128] [1+S, C]
                avg = datatemp.mean(0)     # 128
                loss_contract, _ = self.contractive_loss(datatemp, avg)        # 计算两个分支的相似度
                loss_TCR = cal_TCR(datatemp, self.criterion, len(datatemp))    #计算正则
                groupinnerloss += 200 * loss_contract + 1 * loss_TCR

            for pointindex, points in enumerate(fii1):
                datatemp = torch.unsqueeze(points, 0)
                datatemp = torch.cat((fjj1[pointindex, :, :], datatemp), 0)
                avg = datatemp.mean(0)
                loss_contract, _ = self.contractive_loss(datatemp, avg)
                loss_TCR = cal_TCR(datatemp, self.criterion, len(datatemp))
                groupinnerloss += 200 * loss_contract + 1 * loss_TCR

            groupinnerloss = groupinnerloss / self.downsample

        groupinnerloss = groupinnerloss / self.b


        ##################grouploss####################
        loss = globalpointloss + groupinnerloss

        return loss


#  局部损失
class ChamferLoss(nn.Module):

    def __init__(self):

        super().__init__()
        self.n = None
        self.b = None
        self.c = None



    def forward(self, matrix1, matrix2):
        lambd = 1

        b,n,c = matrix1.shape
        b1,n1,c1 = matrix2.shape

        if b!=b1 or n!=n1 or c!=c1:
            print("维度不匹配!")
            return 0
        self.b = b
        self.n = n
        self.c = c

       #  groupdisloss = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
       #  for pcpos in matrix1[0:, :, :]:
       #     for index,tensors in enumerate(pcpos):
       #         if index < len(pcpos)-1:
       #             datatemp = torch.unsqueeze(tensors,0)
       #             inner_product = torch.mm(datatemp, pcpos[(index+1):,:].t())
       #             norms1 = torch.norm(datatemp, dim=1)
       #             norms2 = torch.norm(pcpos[(index+1):,:], dim=1)
       #             similar_matrix = inner_product / torch.mm(norms1.view(-1, 1), norms2.view(1, -1))
       #             similar_matrix[similar_matrix > 1] = 1.0
       #             similar_matrix = torch.pow(torch.abs(similar_matrix),2)
       #             #print(torch.min(similar_matrix))
       #             groupdisloss = groupdisloss + torch.sum(similar_matrix)
       #             #groupdisloss = groupdisloss + torch.min(similar_matrix)
       #
       #
       #  for pcpos in matrix2[0:, :, :]:
       #     for index,tensors in enumerate(pcpos):
       #         if index < len(pcpos)-1:
       #             datatemp = torch.unsqueeze(tensors,0)
       #             inner_product = torch.mm(datatemp, pcpos[(index+1):,:].t())
       #             norms1 = torch.norm(datatemp, dim=1)
       #             norms2 = torch.norm(pcpos[(index+1):,:], dim=1)
       #             similar_matrix = inner_product / torch.mm(norms1.view(-1, 1), norms2.view(1, -1))
       #             #similar_matrix = torch.pow(torch.abs(similar_matrix),2)
       #             similar_matrix[similar_matrix > 1] = 1.0
       #             similar_matrix = torch.pow(torch.abs(similar_matrix), 2)
       #             groupdisloss = groupdisloss + torch.sum(similar_matrix)
       #             #groupdisloss = groupdisloss + torch.min(similar_matrix)
       #
       # # groupdisloss = (groupdisloss / (n * (n-1))) / b
       #  groupdisloss = (groupdisloss / ((self.n) * ((self.n) - 1))) / self.b
       #  #groupdisloss = (groupdisloss / (n - 1)) / b










       # matrix1 = F.normalize(matrix1, dim=2)
        #matrix2 = F.normalize(matrix2, dim=2)


        loss = torch.tensor(0, dtype=torch.float32,requires_grad=True).cuda()
        ##############  归一化欧式距离
        # for index, pcpos in enumerate(matrix1[0:,:,:]):
        #
        #     plyloss = torch.tensor(0, dtype=torch.float32).cuda()
        #
        #     for index1, point in enumerate(pcpos[0:,:]):
        #
        #
        #             dis = torch.sqrt(torch.sum(torch.square(point-matrix2[index,:,:]),dim=1))  #欧氏距离
        #             #dis = torch.sum(torch.abs(point - matrix2[index, :, :]), dim=1)  #曼哈顿距离
        #             dis = 1 / (1 + dis)
        #             plyloss = torch.add(plyloss,torch.min(dis))
        #
        #     loss = torch.add(loss,plyloss / n)
        #
        #
        # for index, pcpos in enumerate(matrix2[0:, :, :]):
        #
        #     plyloss = torch.tensor(0, dtype=torch.float32).cuda()
        #
        #     for index1, point in enumerate(pcpos[0:, :]):
        #         dis = torch.sqrt(torch.sum(torch.square(point - matrix1[index, :, :]), dim=1))  #欧氏距离
        #         #dis = torch.sum(torch.abs(point - matrix1[index, :, :]), dim=1)  #曼哈顿距离
        #         dis = 1 / (1 + dis)
        #         plyloss = torch.add(plyloss, torch.min(dis))
        #
        #     loss = torch.add(loss, plyloss / n)
        ##############  归一化欧式距离



            # prod = torch.mm(pcpos, matrix2[index,:,:].t())  # 分子
            # norm1 = torch.norm(pcpos, p=2, dim=1).unsqueeze(1)  # 分母
            # norm2 = torch.norm(matrix2[index,:,:], p=2, dim=1).unsqueeze(0)  # 分母
            # #print(torch.mm(norm2.t(), norm1).shape)
            # similar_matrix = prod.div(torch.mm(norm1,norm2))
            #print(similar_matrix)
           # print(cos.shape)

            # dis = torch.dist(pcpos,matrix2[index,:,:])
            # print(dis.shape)
           # pcpos = pcpos
           # anpcpos = matrix2[index,:,:]

            # inner_product = torch.mm(pcpos,anpcpos.t())
            # norms1 = torch.norm(pcpos,dim=1)
            # norms2 = torch.norm(anpcpos, dim=1)
            # similar_matrix = inner_product / torch.mm(norms1.view(-1,1),norms2.view(1,-1))



            #similar_matrix = torch.mm((pcpos / torch.norm(pcpos, dim=-1, keepdim=True)), (matrix2[index,:,:] / torch.norm(matrix2[index,:,:], dim=-1, keepdim=True)).T)  # 矩阵乘法

            #similar_matrix = torch.cosine_similarity(pcpos.unsqueeze(1), matrix2[index,:,:].unsqueeze(0), dim=-1)


            # rs = torch.gt(similar_matrix,torch.tensor(1, dtype=torch.float32).cuda())
            # if True in rs:
            #     print("error!!!!")



            # similar_matrix = torch.sub(torch.tensor(1, dtype=torch.float32).cuda(),similar_matrix)
            # similar_matrix, _ = torch.min(similar_matrix, dim=1)
            # plyloss1 = torch.div(torch.sum(similar_matrix),torch.tensor(n, dtype=torch.float32).cuda())
            #print("print(plyloss1)",plyloss1)
           # loss = torch.add(loss,plyloss1)


        ##########余弦相似度
        for index, pcpos in enumerate(matrix1[0:,:,:]):

            inner_product = torch.mm(pcpos,matrix2[index,:,:].t())
            norms1 = torch.norm(pcpos,dim=1)
            norms2 = torch.norm(matrix2[index,:,:], dim=1)
            similar_matrix = inner_product / torch.mm(norms1.view(-1,1),norms2.view(1,-1))

            similar_matrix[similar_matrix > 1] = 1.0
            rs = torch.gt(similar_matrix,1)
            if True in rs:
                print("error!")

            similar_matrix = torch.pow(torch.sub(torch.tensor(1, dtype=torch.float32).cuda(),similar_matrix),2)
            similar_matrix, _ = torch.min(similar_matrix, dim=1)
            plyloss1 = torch.div(torch.sum(similar_matrix),torch.tensor(n, dtype=torch.float32).cuda())
            loss = loss + plyloss1






        for index, pcpos in enumerate(matrix2[0:,:,:]):

            inner_product = torch.mm(pcpos,matrix1[index,:,:].t())
            norms1 = torch.norm(pcpos,dim=1)
            norms2 = torch.norm(matrix1[index,:,:], dim=1)
            similar_matrix = inner_product / torch.mm(norms1.view(-1,1),norms2.view(1,-1))

            similar_matrix[similar_matrix > 1] = 1.0
            rs = torch.gt(similar_matrix,1)
            if True in rs:
                print("error!")


            similar_matrix = torch.pow(torch.sub(torch.tensor(1, dtype=torch.float32).cuda(),similar_matrix), 2)
            similar_matrix, _ = torch.min(similar_matrix, dim=1)
            plyloss2 = torch.div(torch.sum(similar_matrix),torch.tensor(n, dtype=torch.float32).cuda())
           # print("plyloss2",plyloss2)
            loss = loss + plyloss2

        ##########余弦相似度


        loss = loss / b
        #loss = torch.div(loss,torch.tensor(b, dtype=torch.float32).cuda()) * lambd


        #return groupdisloss, loss
        return  loss


# class Finalloss(nn.Module):
#
#     def __init__(self):
#
#         super().__init__()
#         self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
#         self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
#         self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
#
#         # initialization
#         self.w1.data.fill_(1.0).cuda()
#         self.w2.data.fill_(1.0).cuda()
#         self.w3.data.fill_(1.0).cuda()
#
#
#
#     def forward(self, globalloss, localloss1, localloss2):
#
#         loss = self.w1 * globalloss + self.w2 * localloss1 + self.w3 * localloss2
#         return loss


class MultiLossLayer(nn.Module):
  def __init__(self, loss_list_len):
      super().__init__()
      self._loss_list = None
      self._sigmas_sq = []

      self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda()
      self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda()
      self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda()
      self.w4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda()
      self.w5 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda()
     # self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda()

      # initialization
      #self.w1.data.fill_(1).cuda()
      #self.w2.data.fill_(0.1).cuda()
      #self.w3.data.fill_(0.1).cuda()

      nn.init.uniform_(self.w1,0.3,0.5)
      nn.init.uniform_(self.w2,0.9,1.0)
      nn.init.uniform_(self.w3, 0.9, 1.0)
      nn.init.uniform_(self.w4, 0.1, 0.3)
      nn.init.uniform_(self.w5, 0.8, 0.9)
     # nn.init.uniform_(self.w3,0.9,0.99)

      self._sigmas_sq.append(self.w1[0])
      self._sigmas_sq.append(self.w2[0])
      self._sigmas_sq.append(self.w3[0])
      self._sigmas_sq.append(self.w4[0])
      self._sigmas_sq.append(self.w5[0])
    #  self._sigmas_sq.append(self.w3)
      # for i in range(loss_list_len):
      #   self._sigmas_sq.append(torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).data.fill_(1.0).cuda())

  def forward(self,loss_list):

    # self._loss_list = loss_list
    # factor = 1.0 / (2.0 * self._sigmas_sq[0])
    # loss = torch.add(torch.multiply(factor, self._loss_list[0]),  (1.0 / 4.0) * torch.log(self._sigmas_sq[0])*self._loss_list[0]) #
    # for i in range(1, len(self._sigmas_sq)):
    #   factor = 1.0 / 2.0 * self._sigmas_sq[i]
    #   loss = torch.add(loss, torch.add(torch.multiply(factor, self._loss_list[i]), (1.0 / 4.0) * torch.log(self._sigmas_sq[i])*self._loss_list[i]))
    #loss = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
    # for i in range(len(loss_list)):
    #     loss = loss + loss_list[i]

    #loss = loss_list[0] + loss_list[1] + loss_list[2] + loss_list[3] * 0.1
    loss = sum(loss_list)

    return loss


# 分割损失
class SegmentLoss(nn.Module):
  def __init__(self):
      super().__init__()
      self._sigmas_sq = []

      self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda()
      self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda()


      nn.init.uniform_(self.w1,0.2,0.5)
      nn.init.uniform_(self.w2,0.9,1.0)

      self._sigmas_sq.append(self.w1)
      self._sigmas_sq.append(self.w2)



  def forward(self,loss_list):
    self._loss_list = loss_list
    factor = 1.0 / (2.0 * self._sigmas_sq[0])
    loss = torch.add(torch.multiply(factor, self._loss_list[0]),  (1.0 / 2.0) * torch.log(self._sigmas_sq[0]) * self._loss_list[0]) #
    for i in range(1, len(self._sigmas_sq)):
      factor = 1.0 / 2.0 * self._sigmas_sq[i]
      loss = torch.add(loss, torch.add(torch.multiply(factor, self._loss_list[i]), torch.log(1+self._sigmas_sq[i]) * self._loss_list[i]))



    return loss




# class My_super_loss(nn.Module):
#
#     def __init__(self):
#
#         super().__init__()
#         self.sample_fn = furthest_point_sample
#         group_args = {'NAME': 'ballquery', 'radius': 0.08,  'nsample': 32, 'return_only_idx': False}
#         self.grouper = create_grouper(group_args)
#
#         self.w = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda()
#         self.w.data.fill_(0.75).cuda()
#
#
#
#         self.b = None
#         self.n = None
#         self.c = None
#
#     def forward(self, logits, logits1,p0first,p0sec):
#         b,n,c = logits.shape
#         self.b = b
#         self.n = n
#         self.c = c
#         loss = torch.tensor(0, dtype=torch.float32,requires_grad=True).cuda()
#         for index in range(len(logits)):
#             data = logits[index]
#             data1 = logits1[index]
#            # datatemp = torch.broadcast_tensors(data[0], data1)[0]
#             coef = pearsonr(data, data1)
#             # for i in range(1, len(data)):
#             #     datatemp = torch.cat((datatemp, pearsonr(torch.broadcast_tensors(data[i], data1)[0], data1)), dim=1)
#
#            # coef = datatemp
#           #  lambd = 3.9e-3
#            # eyematrix = torch.eye(len(coef)).cuda()
#             loss = loss + torch.sum(torch.pow((1-coef),2)) / len(coef)
#             #loss = loss + (torch.sum(lossmatrix) - torch.sum(torch.diagonal(lossmatrix))) * lambd + torch.sum(torch.diagonal(lossmatrix))
#
#         loss = loss / len(logits)
#
#         #self.tcr = TotalCodingRate(self.n // 16).cuda()
#
#
#         idx = self.sample_fn(p0first, n//16).long()
#
#         new_p = torch.gather(p0first, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
#         logits = logits.permute(0,2,1)
#         fi = torch.gather(logits, -1, idx.unsqueeze(1).expand(-1, logits.shape[1], -1)) # fi b c n
#
#         dp, fj = self.grouper(new_p, p0first, logits)  #fj b c n k
#         fi = fi.permute(0, 2, 1)
#         fj = fj.permute(0, 2, 3, 1)
#
#         groupinnerloss = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
#         groupdisloss = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
#
#         for index in range(b):
#             fii = fi[index]
#             fjj = fj[index]
#
#             for pointindex,points in enumerate(fii):
#
#                 datatemp = torch.unsqueeze(points, 0)
#                 if index < len(fii) - 1:
#                     disinner_product = torch.mm(datatemp, fii[(index + 1):, :].t())
#                     disnorms1 = torch.norm(datatemp, dim=1)
#                     disnorms2 = torch.norm(fii[(index + 1):, :], dim=1)
#                     dissimilar_matrix = disinner_product / torch.mm(disnorms1.view(-1, 1), disnorms2.view(1, -1))
#                     dissimilar_matrix[dissimilar_matrix > 1] = 1.0
#                     dissimilar_matrix = torch.pow(torch.abs(dissimilar_matrix), 2)
#                     groupdisloss = groupdisloss + torch.sum(dissimilar_matrix)
#
#                 inner_product = torch.mm(datatemp, fjj[pointindex, :, :].t())
#                 norms1 = torch.norm(datatemp, dim=1)
#                 norms2 = torch.norm(fjj[pointindex, :, :], dim=1)
#                 similar_matrix = inner_product / torch.mm(norms1.view(-1, 1), norms2.view(1, -1))
#                 similar_matrix[similar_matrix > 1] = 1.0
#                 groupinnerloss = groupinnerloss + torch.sum(torch.pow((1 - similar_matrix),2)) / 32  + torch.sum(torch.abs(points)) * 0.001   #+ self.tcr(fjj[pointindex, :, :])
#                 #groupinnerloss = groupinnerloss + ((torch.sum(similar_matrix) / 32) * self.w[0] + self.tcr(fjj[pointindex, :, :]))
#                # print(self.tcr(fjj[pointindex, :, :]))
#
#
#         idx = self.sample_fn(p0sec, n // 16).long()
#         new_p = torch.gather(p0sec, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
#         logits1 = logits1.permute(0, 2, 1)
#         fi = torch.gather(logits1, -1, idx.unsqueeze(1).expand(-1, logits1.shape[1], -1))  # fi b c n
#         dp, fj = self.grouper(new_p, p0sec, logits1)  # fj b c n k
#         fi = fi.permute(0, 2, 1)
#         fj = fj.permute(0, 2, 3, 1)
#         for index in range(b):
#             fii = fi[index]
#             fjj = fj[index]
#             for pointindex, points in enumerate(fii):
#                 datatemp = torch.unsqueeze(points, 0)
#
#                 if index < len(fii) - 1:
#                     disinner_product = torch.mm(datatemp, fii[(index + 1):, :].t())
#                     disnorms1 = torch.norm(datatemp, dim=1)
#                     disnorms2 = torch.norm(fii[(index + 1):, :], dim=1)
#                     dissimilar_matrix = disinner_product / torch.mm(disnorms1.view(-1, 1), disnorms2.view(1, -1))
#                     dissimilar_matrix[dissimilar_matrix > 1] = 1.0
#                     dissimilar_matrix = torch.pow(torch.abs(dissimilar_matrix), 2)
#                     groupdisloss = groupdisloss + torch.sum(dissimilar_matrix)
#
#
#                 inner_product = torch.mm(datatemp, fjj[pointindex, :, :].t())
#                 norms1 = torch.norm(datatemp, dim=1)
#                 norms2 = torch.norm(fjj[pointindex, :, :], dim=1)
#                 similar_matrix = inner_product / torch.mm(norms1.view(-1, 1), norms2.view(1, -1))
#                 similar_matrix[similar_matrix > 1] = 1.0
#                 #groupinnerloss = groupinnerloss + ((torch.sum(similar_matrix) / 32) * self.w[0] + self.tcr(fjj[pointindex, :, :]))
#                 #groupinnerloss = groupinnerloss + torch.sum(torch.pow((1 - similar_matrix), 2)) / 32 + self.tcr(fjj[pointindex, :, :])
#                 groupinnerloss = groupinnerloss + torch.sum(torch.pow((1 - similar_matrix), 2)) / 32  + torch.sum(torch.abs(points)) * 0.001
#                 #print(self.tcr(fjj[pointindex, :, :]))
#
#         groupinnerloss = groupinnerloss / ( (self.n // 16) * self.b)
#         groupdisloss = (groupdisloss / ((self.n // 16) * ((self.n // 16)-1))) / b
#
#
#         return groupdisloss, groupinnerloss, loss


'''注释'''
# class My_super_loss(nn.Module):
#
#     def __init__(self):
#
#         super().__init__()
#         self.sample_fn = furthest_point_sample
#         self.nsample = 8
#         group_args = {'NAME': 'ballquery', 'radius': 0.08,  'nsample': self.nsample, 'return_only_idx': False}
#         self.grouper = create_grouper(group_args)
#
#         self.w = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).cuda()
#         self.w.data.fill_(0.75).cuda()
#
#         self.downsample = None
#
#         self.b = None
#         self.n = None
#         self.c = None
#
#     def forward(self, logits, logits1,p0first,p0sec,orixyz):
#         b,n,c = logits.shape
#         self.b = b
#         self.n = n
#         self.c = c
#         self.downsample = self.n // 2
#         idx = self.sample_fn(orixyz, self.downsample).long()
#
#         new_p = torch.gather(p0first, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
#         new_p1 = torch.gather(p0sec, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
#
#         logits = logits.permute(0,2,1)
#         logits1 = logits1.permute(0, 2, 1)
#         fi = torch.gather(logits, -1, idx.unsqueeze(1).expand(-1, logits.shape[1], -1)) # fi b c n
#         fi1 = torch.gather(logits1, -1, idx.unsqueeze(1).expand(-1, logits1.shape[1], -1))  # fi b c n
#         dp, fj = self.grouper(new_p, p0first, logits)  #fj b c n k
#         dp1, fj1 = self.grouper(new_p1, p0sec, logits1)  # fj b c n k
#         fi = fi.permute(0, 2, 1)
#         fj = fj.permute(0, 2, 3, 1)
#         fi1 = fi1.permute(0, 2, 1)
#         fj1 = fj1.permute(0, 2, 3, 1)
#         # print(fi.shape)
#         # print(fi1.shape)
#         logits = logits.permute(0, 2, 1)
#         logits1 = logits1.permute(0, 2, 1)
#
#         ##########global points loss #####################
#         globalpointloss = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
#         globalloss = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
#         for index in range(self.b):
#             globaldata,_ = torch.max(fi[index],dim=0)
#             globaldata1,_ = torch.max(fi1[index], dim=0)
#             cossim = nn.functional.cosine_similarity(torch.unsqueeze(globaldata, 0),torch.unsqueeze(globaldata1, 0))
#             globalloss = globalloss + torch.sum(torch.pow((1 - cossim),2))
#
#
#             data = fi[index]
#             data1 = fi1[index]
#             coef = nn.functional.cosine_similarity(data,data1,dim=1)
#             globalpointloss = globalpointloss + torch.sum(torch.pow((1 - coef),2)) / len(coef)
#
#
#
#         globalpointloss = globalpointloss / self.b
#         globalloss = globalloss / self.b
#
#         ##########global points loss #####################
#
#
#         ##################grouploss####################
#         crossgroupinnerloss = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
#         reggroupinnerloss = torch.tensor(0, dtype=torch.float32, requires_grad=True).cuda()
#
#         for index in range(self.b):
#             fii = fi[index]
#             fjj = fj[index]
#             fii1 = fi1[index]
#             fjj1 = fj1[index]
#
#
#             for pointindex,points in enumerate(fii):
#
#                 datatemp = torch.unsqueeze(points, 0)
#                 inner_product = torch.mm(datatemp, fjj1[pointindex, :, :].t())
#                 norms1 = torch.norm(datatemp, dim=1)
#                 norms2 = torch.norm(fjj1[pointindex, :, :], dim=1)
#                 similar_matrix = inner_product / torch.mm(norms1.view(-1, 1), norms2.view(1, -1))
#                 similar_matrix[similar_matrix > 1] = 1.0
#                 crossgroupinnerloss = crossgroupinnerloss + torch.sum(torch.pow((1 - similar_matrix),2)) / self.nsample
#                 reggroupinnerloss = reggroupinnerloss +  torch.sum(torch.abs(datatemp - fjj[pointindex, :, :])) / self.nsample
#
#
#
#             for pointindex, points in enumerate(fii1):
#                 datatemp = torch.unsqueeze(points, 0)
#                 inner_product = torch.mm(datatemp, fjj[pointindex, :, :].t())
#                 norms1 = torch.norm(datatemp, dim=1)
#                 norms2 = torch.norm(fjj[pointindex, :, :], dim=1)
#                 similar_matrix = inner_product / torch.mm(norms1.view(-1, 1), norms2.view(1, -1))
#                 similar_matrix[similar_matrix > 1] = 1.0
#                 crossgroupinnerloss = crossgroupinnerloss + torch.sum(torch.pow((1 - similar_matrix), 2)) / self.nsample
#                 reggroupinnerloss = reggroupinnerloss + torch.sum(torch.abs(datatemp - fjj1[pointindex, :, :])) / self.nsample
#
#
#             crossgroupinnerloss = crossgroupinnerloss / self.downsample
#             reggroupinnerloss = reggroupinnerloss / self.downsample
#
#         crossgroupinnerloss = crossgroupinnerloss / self.b
#         reggroupinnerloss = reggroupinnerloss / self.b
#
#         ##################grouploss####################
#
#
#
#         return globalloss, globalpointloss, crossgroupinnerloss, reggroupinnerloss * 0.1