import torch

# 随机初始化形状为 (256, 128) 的张量
X = torch.randn(256, 128)

# 示例centriod向量
centriod = torch.tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0],
                         [float('nan'), float('inf'), float('nan')],
                         [float('nan'), float('nan'), float('nan')],
                         [float('-inf'), float('nan'), float('nan')]])

rows_nan = torch.any(torch.isnan(centriod), dim=1) | torch.any(torch.isinf(centriod), dim=1)
rows_no = ~torch.any(torch.isnan(centriod), dim=1) & ~torch.any(torch.isinf(centriod), dim=1)
indices_nan = torch.where(rows_nan)[0]    # 取出包含这些值的第一维的索引
indices_no = torch.where(rows_no)[0]  # 取出包含这些值的第一维的索引
if len(indices_nan) != 0:
    centriod_nan = centriod[indices_nan, :]      # 从centriod向量中取出索引对应的向量
    centriod_no = centriod[indices_no, :]      # 从centriod向量中取出索引对应的向量
    farthest_points = []
    centriod_avg = torch.mean(X, dim=0)
    centriod_avg = centriod_avg.unsqueeze(0)[:, :3]
    distances = torch.cdist(X[:,:3], centriod_avg)
    distances = distances.squeeze(-1)
    # 找到距离最远的三个点的索引
    farthest_indices = torch.topk(distances, k=3).indices
    # 找到距离最远的三个点的坐标
    farthest_points = X[farthest_indices][:,:3]
    centriod[indices_nan,:] = farthest_points

