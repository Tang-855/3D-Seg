import torch

#
# def get_corr(fake_Y, Y):  # 计算两个向量person相关系数
#     for i in range()
#     fake_Y, Y = fake_Y.reshape(-1), Y.reshape(-1)
#     fake_Y_mean, Y_mean = torch.mean(fake_Y), torch.mean(Y)
#     corr = (torch.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
#             torch.sqrt(torch.sum((fake_Y - fake_Y_mean) ** 2)) * torch.sqrt(torch.sum((Y - Y_mean) ** 2)))
#     return corr
#
#

# def get_corr(fake_Y, Y):  # 计算两个向量person相关系数
#
#     fake_Y_mean, Y_mean = torch.mean(fake_Y,dim=1), torch.mean(Y,dim=1)
#     fake_Y_mean = fake_Y_mean.reshape(-1,1)
#     Y_mean = Y_mean.reshape(-1, 1)
#     print((fake_Y - fake_Y_mean) * (Y - Y_mean))
#     corr = (torch.sum((fake_Y - fake_Y_mean) * (Y - Y_mean),dim=1)) / (
#             torch.sqrt(torch.sum((fake_Y - fake_Y_mean) ** 2,dim=1)) * torch.sqrt(torch.sum((Y - Y_mean) ** 2,dim=1)))
#     return corr
# input = torch.Tensor([[1,2,3],
#                       [1,2,3],
#                       [1,2,3]])
#
# target = torch.Tensor([[1,2,3],
#                       [4,5,6],
#                       [7,8,9]])
# print(get_corr(input,target))




import torch
from audtorch.metrics.functional import pearsonr
# a = [[1,2,3],[4,5,6],[7,8,9]]
# b = [[1,2,3],[4,5,6],[7,8,9]]
#
# a = torch.tensor(a,dtype=torch.float32)
# b = torch.tensor(b,dtype=torch.float32)
#
# atemp = torch.broadcast_tensors(a[0],b)[0]
# print(get_corr(atemp,b))
# atemp = pearsonr(a,b)
# print(atemp)
# for i in range(1,len(a)):
#     atemp = torch.cat((atemp,pearsonr(torch.broadcast_tensors(a[i],b)[0],b)),dim=1)
#print(atemp)

#
# print('Pearson Correlation between input and target is {0}'.format(output[:, 0]))

# print(torch.corrcoef(temp))

# def corrcoef(x):
#     """传入一个tensor格式的矩阵x(x.shape(m,n))，输出其相关系数矩阵"""
#     f = (x.shape[0] - 1) / x.shape[0]  # 方差调整系数
#     x_reducemean = x - torch.mean(x, axis=0)
#     numerator = torch.matmul(x_reducemean.T, x_reducemean) / x.shape[0]
#     var_ = x.var(axis=0).reshape(x.shape[1], 1)
#     denominator = torch.sqrt(torch.matmul(var_, var_.T)) * f
#     corrcoef = numerator / denominator
#     return corrcoef

a = torch.Tensor([[10,15,30],
                      [40,70,90],
                      [100,30,20]])

b = torch.Tensor([[10,15,30],
                      [40,70,90],
                      [100,30,20]])
atemp = torch.broadcast_tensors(a[0],b)[0]
atemp = pearsonr(atemp,b)
for i in range(1,len(a)):
    atemp = torch.cat((atemp,pearsonr(torch.broadcast_tensors(a[i],b)[0],b)),dim=1)
print(atemp)
#print(pearsonr(input,target))
temp = torch.cat((a,b),dim=0)
print(torch.corrcoef(temp))

