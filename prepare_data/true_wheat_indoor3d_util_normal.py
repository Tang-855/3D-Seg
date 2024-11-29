import numpy as np
import glob
import os
import sys
import torch
# noinspection PyUnresolvedReferences
import math
# noinspection PyUnresolvedReferences
from torch.utils.data import DataLoader, RandomSampler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
import open3d as o3d
import pcl
import pandas as pd

np.set_printoptions(suppress=True)

def farthest_point_sampling(points, num_points):
    """
    最远点采样算法实现函数
    :param points: ndarray类型，表示点云的点集，形状为pip (N, 3)
    :param num_points: 采样后点云的点数
    :return: ndarray类型，表示采样后的点云的点集，形状为(num_points, 3)
    """
    # 从点集中随机选取一个点作为第一个采样点
    sampled_points = np.zeros((num_points, 3))
    sampled_points[0] = points[np.random.randint(0, len(points)), :]

    # 计算每个点到已选点集中的最短距离
    distances = np.sqrt(((points - sampled_points[0])**2).sum(axis=1))
    for i in range(1, num_points):
        # 选取距离已选点集最远的点
        farthest_idx = np.argmax(distances)
        sampled_points[i] = points[farthest_idx, :]
        # 更新已选点集中每个点到新采样点的最短距离
        distances = np.minimum(distances, np.sqrt(((points - sampled_points[i])**2).sum(axis=1)))

    return sampled_points


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

DATA_PATH = os.path.join(ROOT_DIR, 'data', 'true_spike')
g_classes = [x.rstrip() for x in open(os.path.join(BASE_DIR, 'meta/wheat_class_names.txt'))]
g_class2label = {cls: i for i,cls in enumerate(g_classes)}
g_class2color = {'leaf':	[0,128,0],         #0   #由此处的顺序决定赋的标签的顺序
                 'spike':	[255,215,0]   }    #1
g_easy_view_labels = [7,8,9,10,11,1]
g_label2color = {g_classes.index(cls): g_class2color[cls] for cls in g_classes}

global raw_data_index
raw_data_index = 0
    
# -----------------------------------------------------------------------------
# CONVERT ORIGINAL DATA TO OUR DATA_LABEL FILES
# -----------------------------------------------------------------------------

# 不加法向量的
def collect_point_label(anno_path, out_filename, file_format='txt'):
    """ Convert original dataset files to data_label file (each line is XYZRGBL).
        We aggregated all the points from each instance in the room.

    Args:
        anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
        out_filename: path to save collected points and labels (each line is XYZRGBL)
        file_format: txt or numpy, determines what file format to save.
    Returns:
        None
    Note:
        the points are shifted before save, the most negative point is now at origin.
    """
    points_list = []
    
    for f in glob.glob(os.path.join(anno_path, '*.txt')):  #取出anno_path中的文件
        '''查找符合特定规则的文件路径名，返回所有匹配的文件路径列表'''
        cls_1 = os.path.basename(f).split('_')[0] #用spilt分割类名和数字，得到标签文件
        cls = cls_1.split('.')[0]
        """将物品类名取出，例如：beam，board，board..."""
        """os.path.basename() 返回path最后的文件名"""
        '''被删除的部分'''
        if cls not in g_classes: # note: in some room there is 'staris' class..
            cls = 'spike'
        '''被删除的部分结束'''
        # np.loadtxt()读取txt文件，读入数据文件，要求每一行数据的格式相同，XYZRGV
        points = np.loadtxt(f)   #加载标注文件中的点云数
        # 比如one生成(点云数量,1) * 该类别的索引编号,即为该类别所有点云打上了标签
        # 按照定义颜色的顺序给标签赋值，如beam'3'
        labels = np.ones((points.shape[0],1)) * g_class2label[cls]
        points_list.append(np.concatenate([points, labels], 1)) # Nx7   XYZRGBlabel

    """
        np.concatenate()将列表变为numpy数组,列表进行拼接，xyz相对位置 + label 
    """
    data_label = np.concatenate(points_list, 0)    #根据颜色并加上标签后的数组。XYZRGBLabel
    xyz_min = np.amin(data_label, axis=0)[0:3]    # 取出所有坐标中的最小值    # np.amin(a,axis)，返回数组中的最小值
    data_label[:, 0:3] -= xyz_min   # 归一化，坐标全部减去最小坐标，全部移动至原点处
    points1 = data_label[:, 0:3]

    """
        将房间中包含的所有类别的点云数据全部写入一个npy文本中，
    """
    if file_format=='txt':   # txt
        fout = open(out_filename, 'w')
        for i in range(data_label.shape[0]):
            fout.write('%f %f %f %d \n' % \
                       (data_label[i, 0], data_label[i, 1], data_label[i, 2], data_label[i, 3]))
            # fout.write('%f %f %f %d %d %d %d\n' % \
            #               (data_label[i,0], data_label[i,1], data_label[i,2],
            #                data_label[i,3], data_label[i,4], data_label[i,5],
            #                data_label[i,6]))
        fout.close()
    elif file_format=='numpy':   # numpy
        np.save(out_filename, data_label)
    else:
        print('ERROR!! Unknown file format: %s, please use txt or numpy.' % \
            (file_format))
        exit()
        
# 加法向量的
def collect_point_label_1(anno_path, out_filename, file_format='txt'):
    """ Convert original dataset files to data_label file (each line is XYZRGBL).
        We aggregated all the points from each instance in the room.

    Args:
        anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
        out_filename: path to save collected points and labels (each line is XYZRGBL)
        file_format: txt or numpy, determines what file format to save.
    Returns:
        None
    Note:
        the points are shifted before save, the most negative point is now at origin.
    """
    points_list = []

    for f in glob.glob(os.path.join(anno_path, '*.txt')):  # 取出anno_path中的文件
        '''查找符合特定规则的文件路径名，返回所有匹配的文件路径列表'''
        cls_1 = os.path.basename(f).split('_')[0]  # 用spilt分割类名和数字，得到标签文件
        cls = cls_1.split('.')[0]
        """将物品类名取出，例如：beam，board，board..."""
        """os.path.basename() 返回path最后的文件名"""
        '''被删除的部分'''
        if cls not in g_classes:  # note: in some room there is 'staris' class..
            cls = 'spike'
        '''被删除的部分结束'''
        # np.loadtxt()读取txt文件，读入数据文件，要求每一行数据的格式相同，XYZRGV
        points = np.loadtxt(f)  # 加载标注文件中的点云数
        # 比如one生成(点云数量,1) * 该类别的索引编号,即为该类别所有点云打上了标签
        # 按照定义颜色的顺序给标签赋值，如beam'3'
        labels = np.ones((points.shape[0], 1)) * g_class2label[cls]
        points_list.append(np.concatenate([points, labels], 1))  # Nx7   XYZRGBlabel

    """
        np.concatenate()将列表变为numpy数组,列表进行拼接，xyz相对位置 + label 
    """
    data_label = np.concatenate(points_list, 0)  # 根据颜色并加上标签后的数组。XYZRGBLabel
    xyz_min = np.amin(data_label, axis=0)[0:3]  # 取出所有坐标中的最小值    # np.amin(a,axis)，返回数组中的最小值
    data_label[:, 0:3] -= xyz_min  # 归一化，坐标全部减去最小坐标，全部移动至原点处
    points1 = data_label[:, 0:3]

    '''加入法向量'''
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(data_label[:, 0:3])
    pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    # 获取点云的法向量, 并转换为 NumPy 数组
    normals = np.asarray(pointcloud.normals)
    # 将点云的归一化坐标、法向量和标签按行拼接在一起
    data_label = np.concatenate((points1, normals, data_label[:, -1:]), axis=1)
    np.set_printoptions(suppress=True)

    """
        将房间中包含的所有类别的点云数据全部写入一个npy文本中，
    """
    if file_format == 'txt':  # txt
        fout = open(out_filename, 'w')
        for i in range(data_label.shape[0]):
            # fout.write('%f %f %f %d \n' % \
            #            (data_label[i, 0], data_label[i, 1], data_label[i, 2], data_label[i, 3]))
            fout.write('%f %f %f %d %d %d %d\n' % \
                       (data_label[i, 0], data_label[i, 1], data_label[i, 2],
                        data_label[i, 3], data_label[i, 4], data_label[i, 5],
                        data_label[i, 6]))
        fout.close()
    elif file_format == 'numpy':  # numpy
        np.save(out_filename, data_label)
    else:
        print('ERROR!! Unknown file format: %s, please use txt or numpy.' % \
              (file_format))
        exit()


def point_label_to_obj(input_filename, out_filename, label_color=True, easy_view=False, no_wall=False):
    """ For visualization of a room from data_label file,
	input_filename: each line is X Y Z R G B L
	out_filename: OBJ filename,
            visualize input file by coloring point with label color
        easy_view: only visualize furnitures and floor
    """
    data_label = np.loadtxt(input_filename)
    # data = data_label[:, 0:6]   取第0~6列所有行的数据
    data = data_label[:, 0:3]
    label = data_label[:, -1].astype(int)
    fout = open(out_filename, 'w')
    for i in range(data.shape[0]):
        color = g_label2color[label[i]]
        if easy_view and (label[i] not in g_easy_view_labels):
            continue
        if no_wall and ((label[i] == 2) or (label[i]==0)):
            continue
        if label_color:
            fout.write('v %f %f %f %d %d %d\n' % \
                (data[i,0], data[i,1], data[i,2], color[0], color[1], color[2]))
        else:
            fout.write('v %f %f %f %d %d %d\n' % \
                (data[i,0], data[i,1], data[i,2], data[i,3], data[i,4], data[i,5]))
    fout.close()
 

# -----------------------------------------------------------------------------
# PREPARE BLOCK DATA FOR DEEPNETS TRAINING/TESTING
# -----------------------------------------------------------------------------


#如果当前点大于4096，将当前块的数量采样到4096个点。如果当前点小于4096个点，就补全
def sample_data(data, num_sample):
    """ data is in N x ...
        we want to keep num_sample x C of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    if (N == num_sample):  #如果当前小块的点X=num_sample，保持原始数量
        return data, range(N)
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)    #如果当前小块的点X>num_sample，随机抛弃几个点
        return data[sample, ...], sample
    else:   #else
        sample = np.random.choice(N, num_sample-N)      #如果当前小块的点X<num_sample，随机复制几个点.随机复制之后的点索引
        dup_data = data[sample, ...]
        return np.concatenate([data, dup_data], 0), list(range(N))+list(sample)


#对于上采样的点赋标签
def sample_data_label(data, label, num_sample):
    new_data, sample_indices = sample_data(data, num_sample)
    new_label = label[sample_indices]
    return new_data, new_label


#如果当前点大于4096，将当前块的数量采样到4096个点。如果当前点小于4096个点，就补全
def sample_data_spike(data, leaf, spike,  num_sample):
    """ data is in N x ...
        we want to keep num_sample x C of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    N_0 = leaf.shape[0]
    N_1 = spike.shape[0]
    num_leaf = 820
    num_spike = 3276
    if (N == num_sample):  #如果当前小块的点X=num_sample，保持原始数量
        return data, range(N)
    else:
        if (N_0 >= num_leaf) and (N_1 > num_spike) :
            sample_0 = np.random.choice(N_0, num_leaf)        # 随机抛弃叶点
            sample_1 = np.random.choice(N_1, num_spike)        # 随机抛弃穗点
            return leaf[sample_0, ...], spike[sample_1, ...], sample_0, sample_1
        elif (N_0 >= num_leaf) and (N_1 < num_spike) :
            sample_0 = np.random.choice(N_0, num_leaf)         # 随机抛弃叶点
            sample_1 = np.random.choice(N_1, num_spike-N_1)     # 随机复制穗点
            dup_data = spike[sample_1, ...]
            return leaf[sample_0, ...], np.concatenate([spike, dup_data], 0), sample_0, list(range(N_1))+list(sample_1)
        elif (N_0 >= num_leaf) and (N_1 == num_spike) :
            sample_0 = np.random.choice(N_0, num_leaf)         # 随机抛弃叶点，穗点不变
            return leaf[sample_0, ...], spike, sample_0, range(N_1)
        elif (N_0 < num_leaf) and (N_1 < num_spike) :
            sample_0 = np.random.choice(N_0, num_leaf-N_0)      # 随机复制叶点
            sample_1 = np.random.choice(N_1, num_spike-N_1)      # 随机复制穗点
            dup_data_0 = leaf[sample_0, ...]
            dup_data_1 = spike[sample_1, ...]
            return np.concatenate([leaf, dup_data_0], 0), np.concatenate([spike, dup_data_1], 0), list(range(N_0))+list(sample_0), list(range(N_1))+list(sample_1)
        elif (N_0 < num_leaf) and (N_1 > num_spike) :
            sample_0 = np.random.choice(N_0, num_leaf-N_0)      # 随机复制叶点
            sample_1 = np.random.choice(N_1, num_spike)          # 随机抛弃穗点
            dup_data_0 = leaf[sample_0, ...]
            return np.concatenate([leaf, dup_data_0], 0), spike[sample_1, ...], list(range(N_0))+list(sample_0), sample_1
        else:          # (N_0 < num_leaf) and (N_1 = num_spike) :
            sample_0 = np.random.choice(N_0, num_leaf)      # 随机复制叶点
            dup_data_0 = leaf[sample_0, ...]
            return np.concatenate([leaf, dup_data_0], 0), spike, list(range(N_0))+list(sample_0), range(N_1)
        # else:   #else
        #     sample = np.random.choice(N, num_sample-N)      #如果当前小块的点X<num_sample，随机复制几个点
        #     dup_data = data[sample, ...]
        #     return np.concatenate([data, dup_data], 0), list(range(N))+list(sample)


#对于上采样的点赋标签
def sample_data_label_spike(data, label, leaf, leaf_label, spike, spike_label, num_sample):
    new_data_0, new_data_1, sample_indices_0, sample_indices_1 = sample_data_spike(data, leaf, spike, num_sample)
    new_data = np.concatenate([new_data_0, new_data_1], 0)
    new_label_0 = leaf_label[sample_indices_0]
    new_label_1 = spike_label[sample_indices_1]
    new_label = np.concatenate([new_label_0, new_label_1], 0)
    return new_data, new_label


# 假设data是要搜索的范围
# anchor是锚点的坐标集合
# kk是要查询的最近邻的数量
def find_knn(data,anchor, kk=4096):
    # 从data中提取前三列作为坐标
    coords = data[:, :3]
    # 用KDTree类构建一个空间索引结构
    tree = KDTree(coords)

    anchor = np.squeeze(anchor)[:3]
    # 用query方法查询最近的kk个点，返回距离和索引
    distances, indices = tree.query(anchor, k=kk)
    # 用索引从data中提取对应的点和标签
    # points = data[indices]
    # labels = data[indices, -1]
    return distances, indices
    # nbrs = NearestNeighbors(n_neighbors=kk, algorithm='kd_tree').fit(data)     # 创建NearestNeighbors类的实例
    # distances, indices = nbrs.kneighbors(anchor)   #anchor[i]数组，[anchor[i]]array数组
    # return distances, indices

#DGCNN原始采样方法
def room2blocks(data_label_filename, data, label, num_point, block_size=1.0, stride=1.0, random_sample=False, sample_num=None, sample_aug=1):  #block_size=1.0
# def room2blocks(data_label_filename, data, label, num_point, block_size=1.0, stride=1.0, random_sample=True, sample_num=18, sample_aug=2):  # block_size=1.0
    """ Prepare block training data.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and aligned
            (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters 浮点数，块的物理大小（以米为单位）
        stride: float, stride for block sweeping  浮动，对小块扫描时的步长
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area][默认为房间的数量]
        sample_aug: if random sample, how much aug
    Returns:
        block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
        block_labels: K x num_point x 1 np array of uint8 labels
        
    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert(stride<=block_size)

    #将取出的最大值作为限制阈值
    limit = np.amax(data, 0)[0:3]

    # Get the corner location for our sampling blocks  获取采样块的角落位置
    xbeg_list = []  #计算(x * y)的值
    ybeg_list = []   #计算(x * y)的值
    # 如果random_sample的值为ture，就会自动采样，sample_num确定采样点数
    # 如果random_sample的值为false，就不会自动采样.
    if not random_sample:   #abs返回绝对值
        num_block_x = int(abs(np.ceil((limit[0] - block_size) / stride))) + 1  #np.ceil：向上取整。计算x划分的块数，通过三个参数调整limit[0]（最大值的第1列，即x的值） ， block_size和stride
        num_block_y = int(abs(np.ceil((limit[1] - block_size) / stride))) + 1  #计算y划分的块数。通过三个参数调整limit[1]（最大值的第1列，即y的值） ， block_size和stride
        # 对xy的值进行循环
        for i in range(num_block_x):
            for j in range(num_block_y):
                xbeg_list.append(i*stride)   #每次滑动stride的步长，将block的x和y储存在ybeg_list中
                ybeg_list.append(j*stride)
    else:
        num_block_x = int(np.ceil(limit[0] / block_size))
        num_block_y = int(np.ceil(limit[1] / block_size))
        if sample_num is None:
            sample_num = num_block_x * num_block_y * sample_aug
        for _ in range(sample_num):
            xbeg = np.random.uniform(-block_size, limit[0]) 
            ybeg = np.random.uniform(-block_size, limit[1]) 
            xbeg_list.append(xbeg)
            ybeg_list.append(ybeg)
     #data_label_filename = data_label_filename[:-4].split('//')
    data_label_filename = data_label_filename[:-4].split('\\')  #1
    data_label_filename = data_label_filename[len(data_label_filename) - 1]    # Run1_WT1_1
    test_area = data_label_filename[7]  #5
    room_name = data_label_filename[9:]   #7:

    raw_data_path_1 = "D:\\a-project-T\\Data\\true_data\\true_wheat_sem_seg_hdf5_data_test"
    raw_data_path = "D:\\a-project-T\\Data\\true_data\\true_wheat_sem_seg_hdf5_data_test\\raw_data3d"
    # raw_data_path_1 = "../data/true_wheat_sem_seg_hdf5_data_test"
    # raw_data_path = "../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d"
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)
    if not os.path.exists(str(raw_data_path) + "\\Sample_" + str(test_area)):
        os.makedirs(str(raw_data_path) + "\\Sample_" + str(test_area))
    '''原始代码'''
    # if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d"):
    #     os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d")
    # if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_"+str(test_area)):
    #     os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_"+str(test_area))
    
    # Collect blocks
    # block_datas: K * num_point * 6 np array of XYZRGB, RGB is in [0,1]
    # block_labels: K * num_point * 1 np array of uint8 labels
    block_data_list = []
    block_label_list = []
    global raw_data_index   #声明为全局数据（原始数据的索引）
    #遍历xbeg_list的值，并赋给xbeg，遍历ybeg_list的值，并赋给ybeg，
    for idx in range(len(xbeg_list)):
        xbeg = xbeg_list[idx]
        ybeg = ybeg_list[idx]
        xcond = (data[:, 0] <= xbeg + block_size) & (data[:, 0] >= xbeg)   #判断该区块中的x值有多少，将整个data中的X值与block的大小相匹配
        ycond = (data[:, 1] <= ybeg + block_size) & (data[:, 1] >= ybeg)   #判断该区块中的x值有多少，将整个data中的y值与block的大小相匹配
        cond = xcond & ycond   #求xcond与ycond的交集，查看是否在
        ###原始的数据集加载
        if np.sum(cond) <= 1:  # discard block if there are less than 100 pts.  如果少于 100 点，则丢弃块。 #100
            print("number is not enough discard")
            continue
        block_data = data[cond, :3]  ##丢弃了小于100个点的块之后剩下的，在wheat数据集中每个块均大于100个点，所以没有丢弃的块，在S3DIS中由于点云的缺失，有丢弃的块，此处的阈值可以调高
        block_label = label[cond]

        # randomly subsample data   随机重采样数据
        block_data_sampled, block_label_sampled = \
            sample_data_label(block_data, block_label, num_point)   #num_point
        block_data_list.append(np.expand_dims(block_data_sampled, 0))    #block_data_sampled
        block_label_list.append(np.expand_dims(block_label_sampled, 0))    #block_data_sampled

        '''制作shapenet数据集时使用'''
        # x_tensor = torch.from_numpy(block_data_sampled)        # 将ndarray转换为PyTorch的Tensor
        # y_tensor = torch.from_numpy(block_label_sampled)
        # y_tensor = y_tensor.unsqueeze(1).int()        # 将y_tensor的维度从 (2,) 转换为 (2, 1)
        # z = torch.cat((x_tensor, y_tensor), dim=1)        # 在维度1上拼接x_tensor和y_tensor
        # f = open('../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_' + str(test_area) + '/' + str(room_name) + '(' + str(raw_data_index) + ').txt', "w+")
        # np.savetxt(f, z, fmt='%s', delimiter=' ')  # block_data_sampled
        '''加入结束'''

        f = open('../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_' + str(test_area) + '/' + str(room_name) + '(' + str(raw_data_index) + ').txt', "w+")
        np.savetxt(f, block_data_sampled[:, 0:3], fmt='%s', delimiter=' ')
        raw_data_index = raw_data_index + 1
    return np.concatenate(block_data_list, 0), \
           np.concatenate(block_label_list, 0)


# 自定义每个区块的大小，且每个小块的点在4000个左右，每个小块的y值不一样
def room2blocks_2(data_label_filename, data, label, num_point, data_label, block_size=1.0, stride=1.0, random_sample=False, sample_num=None, sample_aug=1):  # block_size=1.0
    # def room2blocks(data_label_filename, data, label, num_point, block_size=1.0, stride=1.0, random_sample=True, sample_num=18, sample_aug=2):  # block_size=1.0
    """ Prepare block training data.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and aligned
            (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters 浮点数，块的物理大小（以米为单位）
        stride: float, stride for block sweeping  浮动，对小块扫描时的步长
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area][默认为房间的数量]
        sample_aug: if random sample, how much aug
    Returns:
        block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
        block_labels: K x num_point x 1 np array of uint8 labels

    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert (stride <= block_size)

    # 将取出的最大值作为限制阈值
    limit = np.amax(data, 0)[0:3]

    data_label_filename = data_label_filename[:-4].split('\\')  # 1
    data_label_filename = data_label_filename[len(data_label_filename) - 1]  # Run1_WT1_1
    test_area = data_label_filename[7]  # 5
    room_name = data_label_filename[9:]  # 7:
    
    raw_data_path_1 = "D:\\a-project-T\\Data\\true_data\\true_wheat_sem_seg_hdf5_data_test"
    raw_data_path = "D:\\a-project-T\\Data\\true_data\\true_wheat_sem_seg_hdf5_data_test\\raw_data3d"
    # raw_data_path_1 = "../data/true_wheat_sem_seg_hdf5_data_test"
    # raw_data_path = "../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d"
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)
    if not os.path.exists(str(raw_data_path) + "\\Sample_" + str(test_area)):
        os.makedirs(str(raw_data_path) + "\\Sample_" + str(test_area))

    # Get the corner location for our sampling blocks  获取采样块的角落位置
    xbeg_list = []  # 计算(x * y)的值
    metrics_1 = {room_name: [], 'sampled': []}
    # 如果random_sample的值为ture，就会自动采样，sample_num确定采样点数
    # 如果random_sample的值为false，就不会自动采样.
    global raw_data_index                    # 声明为全局数据（原始数据的索引）
    block_data_list = []
    block_label_list = []
    if not random_sample:                   # abs返回绝对值   np.floor向下取整
        num_block_x = int(abs(np.floor(limit[0] / 2.49)))  # np.ceil：向上取整。计算x划分的块数，通过三个参数调整limit[0]（最大值的第1列，即x的值） ， block_size和stride
        for i in range(num_block_x):        # 对xy的值进行循环
            xbeg_list.append(i * 2.49)  # stride 每次滑动stride的步长，将block的x和y储存在ybeg_list中
        for idx in range(len(xbeg_list)):
            xbeg = xbeg_list[idx]
            j, m = 0, 0
            while j <= limit[1] and m == 0:
                ybeg = float(j) + 0.3              # stride
                if xbeg == xbeg_list[-1]:
                    xcond = ((xbeg <= data[:, 0]) & (data[:, 0] <= limit[0]))  # 判断该区块中的x值有多少，将整个data中的X值与block的大小相匹配
                    ycond = ((j <= data[:, 1]) & (data[:, 1] <= ybeg))  # 判断该区块中的x值有多少，将整个data中的y值与block的大小相匹配
                else:
                    xcond = ((xbeg <= data[:, 0]) & (data[:, 0] <= xbeg_list[idx + 1]))  # 判断该区块中的x值有多少，将整个data中的X值与block的大小相匹配
                    ycond = ((j <= data[:, 1]) & (data[:, 1] <= ybeg))  # 判断该区块中的x值有多少，将整个data中的y值与block的大小相匹配
                cond = xcond & ycond  # 求xcond与ycond的交集，查看是否在
                ###原始的数据集加载     设置阈值，抛弃小于某个阈值的点
                if np.sum(cond) <= 0:  # discard block if there are less than 100 pts.  如果少于 100 点，则丢弃块。 #100
                    print("number is not enough discard")
                    j = j + 0.1
                    continue
                block_data = data[cond, :]  ##丢弃了小于100个点的块之后剩下的，在wheat数据集中每个块均大于100个点，所以没有丢弃的块，在S3DIS中由于点云的缺失，有丢弃的块，此处的阈值可以调高
                block_label = label[cond]
                plot_number = data[cond, :].shape[0]

                while 3500 > plot_number and ybeg <= limit[1]:
                    ybeg = ybeg + 0.1  # stride
                    if xbeg == xbeg_list[-1]:
                        xcond = ((xbeg <= data[:, 0]) & (data[:, 0] <= limit[0]))  # 判断该区块中的x值有多少，将整个data中的X值与block的大小相匹配
                        ycond = ((j <= data[:, 1]) & (data[:, 1] <= ybeg))  # 判断该区块中的x值有多少，将整个data中的y值与block的大小相匹配
                    else:
                        xcond = ((xbeg <= data[:, 0]) & (data[:, 0] <= xbeg_list[idx + 1]))  # 判断该区块中的x值有多少，将整个data中的X值与block的大小相匹配
                        ycond = ((j <= data[:, 1]) & (data[:, 1] <= ybeg))  # 判断该区块中的x值有多少，将整个data中的y值与block的大小相匹配
                    cond = xcond & ycond  # 求xcond与ycond的交集，查看是否在
                    ###原始的数据集加载     设置阈值，抛弃小于某个阈值的点
                    if np.sum(cond) <= 0:  # discard block if there are less than 100 pts.  如果少于 100 点，则丢弃块。 #100
                        print("number is not enough discard")
                        continue
                    block_data = data[cond, :]  ##丢弃了小于100个点的块之后剩下的，在wheat数据集中每个块均大于100个点，所以没有丢弃的块，在S3DIS中由于点云的缺失，有丢弃的块，此处的阈值可以调高
                    block_label = label[cond]
                    plot_number = data[cond, :].shape[0]
                    print("小区的数值小于3500，重新采样。当前的小区数值是：" + str(plot_number))

                while  plot_number > 5500 and j< ybeg < limit[1]:
                    print("小区的数值大于5000，重新采样")
                    ybeg = ybeg - 0.05  # stride
                    if xbeg == xbeg_list[-1]:
                        xcond = ((xbeg <= data[:, 0]) & (data[:, 0] <= limit[0]))  # 判断该区块中的x值有多少，将整个data中的X值与block的大小相匹配
                        ycond = ((j <= data[:, 1]) & (data[:, 1] <= ybeg))  # 判断该区块中的x值有多少，将整个data中的y值与block的大小相匹配
                    else:
                        xcond = ((xbeg <= data[:, 0]) & (data[:, 0] <= xbeg_list[idx + 1]))  # 判断该区块中的x值有多少，将整个data中的X值与block的大小相匹配
                        ycond = ((j <= data[:, 1]) & (data[:, 1] <= ybeg))  # 判断该区块中的x值有多少，将整个data中的y值与block的大小相匹配
                    cond = xcond & ycond  # 求xcond与ycond的交集，查看是否在
                    ###原始的数据集加载     设置阈值，抛弃小于某个阈值的点
                    if np.sum(cond) <= 0:  # discard block if there are less than 100 pts.  如果少于 100 点，则丢弃块。 #100
                        print("number is not enough discard")
                        continue
                    block_data = data[cond, :]  ##丢弃了小于100个点的块之后剩下的，在wheat数据集中每个块均大于100个点，所以没有丢弃的块，在S3DIS中由于点云的缺失，有丢弃的块，此处的阈值可以调高
                    block_label = label[cond]
                    plot_number = data[cond, :].shape[0]
                    print("小区的数值大于5000，重新采样。当前的小区数值是：" + str(plot_number))
                if 3500 <= plot_number <= 5000:
                    print("小区的数值在350~5000之间，不用重新采样。当前的小区数值是：" + str(plot_number))
                # randomly subsample data   随机重采样数据
                block_data_sampled, block_label_sampled = \
                    sample_data_label(block_data, block_label, num_point)  # num_point
                block_data_list.append(np.expand_dims(block_data_sampled, 0))  # block_data_sampled
                block_label_list.append(np.expand_dims(block_label_sampled, 0))  # block_label_sampled
                j = ybeg
                if j <= limit[1]:
                    j = j
                else:
                    j = limit[1]
                    m = limit[1]

                '''制作shapenet有标签的数据集时使用'''
                #   # 将ndarray转换为PyTorch的Tensor
                # x_tensor = torch.from_numpy(block_data_sampled)
                # y_tensor = torch.from_numpy(block_label_sampled)
                # y_tensor = y_tensor.unsqueeze(1).int()  # # 将y_tensor的维度从 (2,) 转换为 (2, 1)
                # z = torch.cat((x_tensor, y_tensor), dim=1)  # # 在维度1上拼接x_tensor和y_tensor
                # path = 'D:/a-project-T/Pointnext\data/ShapeNetPart/useddatasetxyz/Sample_' + str(test_area)
                # if not os.path.exists(path):
                #     os.makedirs(path)
                # with open(str(path) + '/' + str(room_name) + '(' + str(raw_data_index) + ').txt', "w+") as f:
                #     np.savetxt(f, z, fmt='%.6f',
                #                delimiter=' ')  # np.savetxt(f, block_data_sampled[:,0:3], fmt='%s', delimiter=' ')
                '''加入结束'''

                '''制作shapenet无监督的数据集时使用'''
                #   # 将ndarray转换为PyTorch的Tensor
                # path = 'D:/a-project-T/Pointnext\data/ShapeNetPart/useddatasetxyz/Sample_' + str(test_area)
                # if not os.path.exists(path):
                #     os.makedirs(path)
                # with open(str(path) + '/' + str(room_name) + '(' + str(raw_data_index) + ').txt', "w+") as f:
                #     np.savetxt(f, block_data_sampled, fmt='%.6f', delimiter=' ')  # np.savetxt(f, block_data_sampled[:,0:3], fmt='%s', delimiter=' ')
                '''加入结束'''

                f = open(str(raw_data_path)+ '/Sample_' + str(test_area) + '/' + str(room_name) + '(' + str(raw_data_index) + ').txt', "w+")
                with open(str(raw_data_path)+ '/Sample_' + str(test_area) + '/' + str(room_name) + '(' + str(raw_data_index) + ').txt', "w+") as f:
                    np.savetxt(f, block_data_sampled[:,0:3], fmt='%s', delimiter=' ')  # np.savetxt(f, block_data_sampled[:,0:3], fmt='%s', delimiter=' ')

                raw_data_index = raw_data_index + 1
                metrics_1[room_name].append(block_data.shape[0])
                metrics_1['sampled'].append(block_data_sampled[:, 0:3].shape[0])

    else:
        num_block_x = int(np.ceil(limit[0] / block_size))
        num_block_y = int(np.ceil(limit[1] / block_size))
        if sample_num is None:
            sample_num = num_block_x * num_block_y * sample_aug
        for _ in range(sample_num):
            xbeg = np.random.uniform(-block_size, limit[0])
            ybeg = np.random.uniform(-block_size, limit[1])
            xbeg_list.append(xbeg)
            # ybeg_list.append(ybeg)

    # 保存为Excel文件
    df = pd.DataFrame(metrics_1)
    root = raw_data_path_1 + '\\statistic\\'
    if not os.path.exists(root):
        os.makedirs(root)
    save_path_1 = root + room_name + '.xlsx'  # 指定文件夹路径
    df.to_excel(save_path_1, index=False)
    return np.concatenate(block_data_list, 0), \
           np.concatenate(block_label_list, 0)


# 自定义每个区块的大小，每个小块的y值一样,叶片和穗分开采样，叶的2/1以上和穗部占80%，叶的2/1以下20%
def room2blocks_2(data_label_filename, data, label, num_point, data_label, block_size=1.0, stride=1.0, random_sample=False, sample_num=None, sample_aug=1):  # block_size=1.0
    # def room2blocks(data_label_filename, data, label, num_point, block_size=1.0, stride=1.0, random_sample=True, sample_num=18, sample_aug=2):  # block_size=1.0
    """ Prepare block training data.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and aligned
            (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters 浮点数，块的物理大小（以米为单位）
        stride: float, stride for block sweeping  浮动，对小块扫描时的步长
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area][默认为房间的数量]
        sample_aug: if random sample, how much aug
    Returns:
        block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
        block_labels: K x num_point x 1 np array of uint8 labels

    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert (stride <= block_size)

    # 将取出的最大值作为限制阈值
    limit = np.amax(data, 0)[0:3]

    '''计算每类的数量和锚点数量'''
    values_s = data_label[np.where(data_label[:, 3] == 1)[0]]  # 取出标签为 1（spike） 的点的第三列的值
    values_l = data_label[np.where(data_label[:, 3] == 0)[0]]  # 取出标签为 0（leaf） 的点的第三列的值 # 取出所有标签点的第三列的值

    # Get the corner location for our sampling blocks  获取采样块的角落位置
    xbeg_list = []  # 计算(x * y)的值
    ybeg_list = []  # 计算(x * y)的值
    # 如果random_sample的值为ture，就会自动采样，sample_num确定采样点数
    # 如果random_sample的值为false，就不会自动采样.
    if not random_sample:  # abs返回绝对值   np.floor向下取整
        num_block_x = int(abs(np.floor(limit[0] / 2.49)))  # np.ceil：向上取整。计算x划分的块数，通过三个参数调整limit[0]（最大值的第1列，即x的值） ， block_size和stride
        num_block_y = int(abs(np.floor(limit[1] / 0.4)))  # 计算y划分的块数。通过三个参数调整limit[1]（最大值的第1列，即y的值） ， block_size和stride
        for i in range(num_block_x):        # 对xy的值进行循环
            for j in range(num_block_y):
                xbeg_list.append(i * 2.49)  # stride 每次滑动stride的步长，将block的x和y储存在ybeg_list中
                ybeg_list.append(j * 0.4)  # stride
    else:
        num_block_x = int(np.ceil(limit[0] / block_size))
        num_block_y = int(np.ceil(limit[1] / block_size))
        if sample_num is None:
            sample_num = num_block_x * num_block_y * sample_aug
        for _ in range(sample_num):
            xbeg = np.random.uniform(-block_size, limit[0])
            ybeg = np.random.uniform(-block_size, limit[1])
            xbeg_list.append(xbeg)
            ybeg_list.append(ybeg)
    data_label_filename = data_label_filename[:-4].split('\\')  # 1
    data_label_filename = data_label_filename[len(data_label_filename) - 1]  # Run1_WT1_1
    test_area = data_label_filename[7]  # 5
    room_name = data_label_filename[9:]  # 7:

    raw_data_path_1 = "D:\\a-project-T\\Data\\true_wheat_data\\true_wheat_sem_seg_hdf5_data_test"
    raw_data_path = "D:\\a-project-T\\Data\\true_wheat_data\\true_wheat_sem_seg_hdf5_data_test\\raw_data3d"
    # raw_data_path_1 = "../data/true_wheat_sem_seg_hdf5_data_test"
    # raw_data_path = "../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d"
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)
    if not os.path.exists(str(raw_data_path) + "\\Sample_" + str(test_area)):
        os.makedirs(str(raw_data_path) + "\\Sample_" + str(test_area))
    '''原始代码'''
    # if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d"):
    #     os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d")
    # if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_" + str(test_area)):
    #     os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_" + str(test_area))

    # Collect blocks
    # block_datas: K * num_point * 6 np array of XYZRGB, RGB is in [0,1]
    # block_labels: K * num_point * 1 np array of uint8 labels

    metrics_1 = {room_name: [], 'sampled': []}
    block_data_list = []
    block_label_list = []
    global raw_data_index  # 声明为全局数据（原始数据的索引）
    # 遍历xbeg_list的值，并赋给xbeg，遍历ybeg_list的值，并赋给ybeg
    for idx in range(len(xbeg_list)):
        xbeg = xbeg_list[idx]
        ybeg = ybeg_list[idx]
        x_list = []
        y_list = []
        for x in range(num_block_x - 1):
            w = (x + 1)*num_block_y - 1
            x_list.append(w)
        y_list = [i for i in range((num_block_y * (num_block_x-1)) , (num_block_y * num_block_x - 1))]

        if idx in x_list:
            xcond = ((xbeg <= data[:, 0]) & (data[:, 0] <= xbeg + 2.49))
            ycond = (data[:, 1] >= ybeg)
        elif idx in y_list:
            xcond = (data[:, 0] >= xbeg)
            ycond = ((ybeg <= data[:, 1]) & (data[:, 1] <= ybeg + 0.4))
        elif idx == len(xbeg_list)-1:
            xcond = (data[:, 0] >= xbeg)  # 判断该区块中的x值有多少，将整个data中的X值与block的大小相匹配
            ycond = (data[:, 1] >= ybeg)  # 判断该区块中的x值有多少，将整个data中的y值与block的大小相匹配
        else:
            xcond = ((xbeg <= data[:, 0]) & (data[:, 0] <= xbeg + 2.49))  # 判断该区块中的x值有多少，将整个data中的X值与block的大小相匹配
            ycond = ((ybeg <= data[:, 1]) & (data[:, 1] <= ybeg + 0.4))  # 判断该区块中的x值有多少，将整个data中的y值与block的大小相匹配

        cond = xcond & ycond  # 求xcond与ycond的交集，查看是否在
        ###原始的数据集加载     设置阈值，抛弃小于某个阈值的点
        if np.sum(cond) <= 0:  # discard block if there are less than 100 pts.  如果少于 100 点，则丢弃块。 #100
            print("number is not enough discard")
            continue
        block_data = data[cond, :]  ##丢弃了小于100个点的块之后剩下的，在wheat数据集中每个块均大于100个点，所以没有丢弃的块，在S3DIS中由于点云的缺失，有丢弃的块，此处的阈值可以调高
        block_label = label[cond]

        '''计算每类的数量和锚点数量'''
        # 找出每个类别的数量
        spike_num = (block_label == 1).sum().item()
        leaf_num = (block_label == 0).sum().item()
        # 找出第四列值为0和1的索引
        index_zeros = (block_label == 0)      # leaf
        index_ones = (block_label == 1)       # spike
        leaf = block_data[block_label == 0]   # 取出所有叶点
        spike = block_data[block_label == 1]  # 取出所有穗点
        leaf_label = block_label[block_label == 0]   # 取出所有叶点
        spike_label = block_label[block_label == 1]  # 取出所有穗点
        if spike_num == 0:
            leaf_max = np.amax(leaf, 0)
        elif leaf_num== 0:
            spike_min = np.amin(spike, 0)
            spike_max = np.amax(spike, 0)
        else:
            leaf_max = np.amax(leaf, 0)
            spike_min = np.amin(spike, 0)
            spike_max = np.amax(spike, 0)

        '''在每个block中间进行采样'''
        # if spike_num == 0 or leaf_num == 0 :
        #     # randomly subsample data   随机重采样数据
        #     block_data_sampled, block_label_sampled = \
        #         sample_data_label(block_data, block_label, num_point)  # num_point
        #     block_data_list.append(np.expand_dims(block_data_sampled, 0))  # block_data_sampled
        #     block_label_list.append(np.expand_dims(block_label_sampled, 0))  # block_label_sampled
        # else:
        #     # randomly subsample data   随机重采样数据
        #     block_data_sampled, block_label_sampled = \
        #         sample_data_label_spike(block_data, block_label, leaf, leaf_label, spike, spike_label, num_point)  # num_point
        #     block_data_list.append(np.expand_dims(block_data_sampled, 0))  # block_data_sampled
        #     block_label_list.append(np.expand_dims(block_label_sampled, 0))  # block_label_sampled
        '''**********  结束  **********'''

        '''原始随机采样'''
          # randomly subsample data   随机重采样数据
        block_data_sampled, block_label_sampled = \
            sample_data_label(block_data, block_label, num_point)  # num_point
        block_data_list.append(np.expand_dims(block_data_sampled, 0))  # block_data_sampled
        block_label_list.append(np.expand_dims(block_label_sampled, 0))  # block_label_sampled
        '''**********  结束  **********'''

        '''制作shapenet数据集时使用'''
          # 将ndarray转换为PyTorch的Tensor
        # x_tensor = torch.from_numpy(block_data_sampled)
        # y_tensor = torch.from_numpy(block_label_sampled)
        # #   # 将y_tensor的维度从 (2,) 转换为 (2, 1)
        # y_tensor = y_tensor.unsqueeze(1).int()
        # #   # 在维度1上拼接x_tensor和y_tensor
        # z = torch.cat((x_tensor, y_tensor), dim=1)
        # f = open(str(raw_data_path) + '/Sample_' + str(test_area) + '/' + str(room_name) + '(' + str(raw_data_index) + ').txt', "w+")
        # with open(str(raw_data_path) + '/Sample_' + str(test_area) + '/' + str(room_name) + '(' + str(raw_data_index) + ').txt', "w+") as f:
        #     np.savetxt(f, z, fmt='%s', delimiter=' ')
        '''加入结束'''

        # f = open('D:\\a-project-T\\Data\\true_wheat_data\\true_wheat_sem_seg_hdf5_data_test\\raw_data3d/Sample_' + str(test_area) + '/' + str( room_name) + '(' + str(raw_data_index) + ').txt', "w+")
        # np.savetxt(f, block_data_sampled, fmt='%s', delimiter=' ')   # block_data_sampled

        f = open(str(raw_data_path) + '/Sample_' + str(test_area) + '/' + str(room_name) + '(' + str(raw_data_index) + ').txt', "w+")
        with open(str(raw_data_path) + '/Sample_' + str(test_area) + '/' + str(room_name) + '(' + str(raw_data_index) + ').txt', "w+") as f:
            np.savetxt(f, block_data_sampled[:, 0:3], fmt='%s', delimiter=' ')

        raw_data_index = raw_data_index + 1
        metrics_1[room_name].append(block_data.shape[0])
        metrics_1['sampled'].append(block_data_sampled[:, 0:3].shape[0])

    # 保存为Excel文件
    df = pd.DataFrame(metrics_1)
    root = 'D:\\a-project-T\\Data\\true_wheat_data\\true_wheat_sem_seg_hdf5_data_test\\statistic\\'
    if not os.path.exists(root):
        os.makedirs(root)
    save_path_1 = root + room_name + '.xlsx'  # 指定文件夹路径
    df.to_excel(save_path_1, index=False)
    return np.concatenate(block_data_list, 0), \
           np.concatenate(block_label_list, 0)


# 自定义每个区块的大小，每个小块的y值一样,叶片和穗分开采样，叶的2/1以上和穗部占80%，叶的2/1以下20%
def room2blocks_2(data_label_filename, data, label, num_point, data_label, block_size=1.0, stride=1.0, random_sample=False, sample_num=None, sample_aug=1):  # block_size=1.0
    # def room2blocks(data_label_filename, data, label, num_point, block_size=1.0, stride=1.0, random_sample=True, sample_num=18, sample_aug=2):  # block_size=1.0
    """ Prepare block training data.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and aligned
            (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters 浮点数，块的物理大小（以米为单位）
        stride: float, stride for block sweeping  浮动，对小块扫描时的步长
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area][默认为房间的数量]
        sample_aug: if random sample, how much aug
    Returns:
        block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
        block_labels: K x num_point x 1 np array of uint8 labels

    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert (stride <= block_size)

    # 将取出的最大值作为限制阈值
    limit = np.amax(data, 0)[0:3]

    '''计算每类的数量和锚点数量'''
    values_s = data_label[np.where(data_label[:, 5] == 1)[0]]  # 取出标签为 1（spike） 的点的第三列的值
    values_l = data_label[np.where(data_label[:, 5] == 0)[0]]  # 取出标签为 0（leaf） 的点的第三列的值 # 取出所有标签点的第三列的值

    # Get the corner location for our sampling blocks  获取采样块的角落位置
    xbeg_list = []  # 计算(x * y)的值
    ybeg_list = []  # 计算(x * y)的值
    # 如果random_sample的值为ture，就会自动采样，sample_num确定采样点数
    # 如果random_sample的值为false，就不会自动采样.
    if not random_sample:  # abs返回绝对值   np.floor向下取整
        num_block_x = int(abs(np.floor(limit[0] / 2.49)))  # np.ceil：向上取整。计算x划分的块数，通过三个参数调整limit[0]（最大值的第1列，即x的值） ， block_size和stride
        num_block_y = int(abs(np.floor(limit[1] / 0.4)))  # 计算y划分的块数。通过三个参数调整limit[1]（最大值的第1列，即y的值） ， block_size和stride
        for i in range(num_block_x):        # 对xy的值进行循环
            for j in range(num_block_y):
                xbeg_list.append(i * 2.49)  # stride 每次滑动stride的步长，将block的x和y储存在ybeg_list中
                ybeg_list.append(j * 0.4)  # stride
    else:
        num_block_x = int(np.ceil(limit[0] / block_size))
        num_block_y = int(np.ceil(limit[1] / block_size))
        if sample_num is None:
            sample_num = num_block_x * num_block_y * sample_aug
        for _ in range(sample_num):
            xbeg = np.random.uniform(-block_size, limit[0])
            ybeg = np.random.uniform(-block_size, limit[1])
            xbeg_list.append(xbeg)
            ybeg_list.append(ybeg)
    data_label_filename = data_label_filename[:-4].split('\\')  # 1
    data_label_filename = data_label_filename[len(data_label_filename) - 1]  # Run1_WT1_1
    test_area = data_label_filename[7]  # 5
    room_name = data_label_filename[9:]  # 7:

    raw_data_path_1 = "D:\\a-project-T\\Data\\true_wheat_data\\true_wheat_sem_seg_hdf5_data_test"
    raw_data_path = "D:\\a-project-T\\Data\\true_wheat_data\\true_wheat_sem_seg_hdf5_data_test\\raw_data3d"
    # raw_data_path_1 = "../data/true_wheat_sem_seg_hdf5_data_test"
    # raw_data_path = "../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d"
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)
    if not os.path.exists(str(raw_data_path) + "\\Sample_" + str(test_area)):
        os.makedirs(str(raw_data_path) + "\\Sample_" + str(test_area))
    '''原始代码'''
    # if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d"):
    #     os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d")
    # if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_" + str(test_area)):
    #     os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_" + str(test_area))

    # Collect blocks
    # block_datas: K * num_point * 6 np array of XYZRGB, RGB is in [0,1]
    # block_labels: K * num_point * 1 np array of uint8 labels

    metrics_1 = {room_name: [], 'sampled': []}
    block_data_list = []
    block_label_list = []
    global raw_data_index  # 声明为全局数据（原始数据的索引）
    # 遍历xbeg_list的值，并赋给xbeg，遍历ybeg_list的值，并赋给ybeg
    for idx in range(len(xbeg_list)):
        xbeg = xbeg_list[idx]
        ybeg = ybeg_list[idx]
        x_list = []
        y_list = []
        for x in range(num_block_x - 1):
            w = (x + 1)*num_block_y - 1
            x_list.append(w)
        y_list = [i for i in range((num_block_y * (num_block_x-1)) , (num_block_y * num_block_x - 1))]

        if idx in x_list:
            xcond = ((xbeg <= data[:, 0]) & (data[:, 0] <= xbeg + 2.49))
            ycond = (data[:, 1] >= ybeg)
        elif idx in y_list:
            xcond = (data[:, 0] >= xbeg)
            ycond = ((ybeg <= data[:, 1]) & (data[:, 1] <= ybeg + 0.4))
        elif idx == len(xbeg_list)-1:
            xcond = (data[:, 0] >= xbeg)  # 判断该区块中的x值有多少，将整个data中的X值与block的大小相匹配
            ycond = (data[:, 1] >= ybeg)  # 判断该区块中的x值有多少，将整个data中的y值与block的大小相匹配
        else:
            xcond = ((xbeg <= data[:, 0]) & (data[:, 0] <= xbeg + 2.49))  # 判断该区块中的x值有多少，将整个data中的X值与block的大小相匹配
            ycond = ((ybeg <= data[:, 1]) & (data[:, 1] <= ybeg + 0.4))  # 判断该区块中的x值有多少，将整个data中的y值与block的大小相匹配

        cond = xcond & ycond  # 求xcond与ycond的交集，查看是否在
        ###原始的数据集加载     设置阈值，抛弃小于某个阈值的点
        if np.sum(cond) <= 0:  # discard block if there are less than 100 pts.  如果少于 100 点，则丢弃块。 #100
            print("number is not enough discard")
            continue
        block_data = data[cond, :3]  ##丢弃了小于100个点的块之后剩下的，在wheat数据集中每个块均大于100个点，所以没有丢弃的块，在S3DIS中由于点云的缺失，有丢弃的块，此处的阈值可以调高
        block_label = label[cond]

        '''计算每类的数量和锚点数量'''
        # 找出每个类别的数量
        spike_num = (block_label == 1).sum().item()
        leaf_num = (block_label == 0).sum().item()
        # 找出第四列值为0和1的索引
        index_zeros = (block_label == 0)
        index_ones = (block_label == 1)
        leaf = block_data[block_label == 0]  # 取出所有叶点
        spike = block_data[block_label == 1]  # 取出所有穗点
        leaf_max = torch.max(leaf,dim=2)
        spike_min = torch.min(spike,dim=2)

        if spike_num == 0:
            # randomly subsample data   随机重采样数据
            block_data_sampled, block_label_sampled = \
                sample_data_label(block_data, block_label, num_point)  # num_point
            block_data_list.append(np.expand_dims(block_data_sampled, 0))  # block_data_sampled
            block_label_list.append(np.expand_dims(block_label_sampled, 0))  # block_label_sampled
        else:
            # randomly subsample data   随机重采样数据
            block_data_sampled, block_label_sampled = \
                sample_data_label_spike(block_data, block_label, num_point)  # num_point
            block_data_list.append(np.expand_dims(block_data_sampled, 0))  # block_data_sampled
            block_label_list.append(np.expand_dims(block_label_sampled, 0))  # block_label_sampled


        '''原始采样方法'''
        # # randomly subsample data   随机重采样数据
        # block_data_sampled, block_label_sampled = \
        #     sample_data_label(block_data, block_label, num_point)  # num_point
        # block_data_list.append(np.expand_dims(block_data_sampled, 0))  # block_data_sampled
        # block_label_list.append(np.expand_dims(block_label_sampled, 0))  # block_label_sampled
        '''原始采样方法结束'''
        '''制作shapenet数据集时使用'''
        #   # 将ndarray转换为PyTorch的Tensor
        # x_tensor = torch.from_numpy(block_data_sampled)
        # y_tensor = torch.from_numpy(block_label_sampled)
        # #   # 将y_tensor的维度从 (2,) 转换为 (2, 1)
        # y_tensor = y_tensor.unsqueeze(1).int()
        # #   # 在维度1上拼接x_tensor和y_tensor
        # z = torch.cat((x_tensor, y_tensor), dim=1)
        # f = open('../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_' + str(test_area) + '/' + str(room_name) + '(' + str(raw_data_index) + ').txt', "w+")
        # np.savetxt(f, z, fmt='%s', delimiter=' ')  # block_data_sampled
        '''加入结束'''

        # f = open('D:\\a-project-T\\Data\\true_wheat_data\\true_wheat_sem_seg_hdf5_data_test\\raw_data3d/Sample_' + str(test_area) + '/' + str( room_name) + '(' + str(raw_data_index) + ').txt', "w+")
        # np.savetxt(f, block_data_sampled, fmt='%s', delimiter=' ')   # block_data_sampled
        f = open(str(raw_data_path) + '/Sample_' + str(test_area) + '/' + str(room_name) + '(' + str(
            raw_data_index) + ').txt', "w+")
        with open(str(raw_data_path) + '/Sample_' + str(test_area) + '/' + str(room_name) + '(' + str(raw_data_index) + ').txt', "w+") as f:
            np.savetxt(f, block_data_sampled[:, 0:3], fmt='%s', delimiter=' ')

        raw_data_index = raw_data_index + 1
        metrics_1[room_name].append(block_data.shape[0])
        metrics_1['sampled'].append(block_data_sampled[:, 0:3].shape[0])

    # 保存为Excel文件
    df = pd.DataFrame(metrics_1)
    root = 'D:\\a-project-T\\Data\\true_wheat_data\\true_wheat_sem_seg_hdf5_data_test\\statistic\\'
    if not os.path.exists(root):
        os.makedirs(root)
    save_path_1 = root + room_name + '.xlsx'  # 指定文件夹路径
    df.to_excel(save_path_1, index=False)
    return np.concatenate(block_data_list, 0), \
           np.concatenate(block_label_list, 0)


# 自定义每个区块的大小，且每个区块的大小相同
def room2blocks_1(data_label_filename, data, label, num_point, data_label, block_size=1.0, stride=1.0, random_sample=False, sample_num=None, sample_aug=1):  # block_size=1.0
    # def room2blocks(data_label_filename, data, label, num_point, block_size=1.0, stride=1.0, random_sample=True, sample_num=18, sample_aug=2):  # block_size=1.0
    """ Prepare block training data.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and aligned
            (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters 浮点数，块的物理大小（以米为单位）
        stride: float, stride for block sweeping  浮动，对小块扫描时的步长
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area][默认为房间的数量]
        sample_aug: if random sample, how much aug
    Returns:
        block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
        block_labels: K x num_point x 1 np array of uint8 labels

    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert (stride <= block_size)

    # 将取出的最大值作为限制阈值
    limit = np.amax(data, 0)[0:3]

    # Get the corner location for our sampling blocks  获取采样块的角落位置
    xbeg_list = []  # 计算(x * y)的值
    ybeg_list = []  # 计算(x * y)的值
    # 如果random_sample的值为ture，就会自动采样，sample_num确定采样点数
    # 如果random_sample的值为false，就不会自动采样.
    if not random_sample:  # abs返回绝对值   # 2.49   0.4
        num_block_x = int(abs(np.ceil(limit[0] / 100))-1)  # np.ceil：向上取整。计算x划分的块数，通过三个参数调整limit[0]（最大值的第1列，即x的值） ， block_size和stride
        num_block_y = int(abs(np.ceil(limit[1] / 50))-1)  # 计算y划分的块数。通过三个参数调整limit[1]（最大值的第1列，即y的值） ， block_size和stride
        # 对xy的值进行循环
        for i in range(num_block_x):
            for j in range(num_block_y):
                xbeg_list.append(i * 100)  # stride 每次滑动stride的步长，将block的x和y储存在ybeg_list中
                ybeg_list.append(j * 50)  # stride
    else:
        num_block_x = int(np.ceil(limit[0] / block_size))
        num_block_y = int(np.ceil(limit[1] / block_size))
        if sample_num is None:
            sample_num = num_block_x * num_block_y * sample_aug
        for _ in range(sample_num):
            xbeg = np.random.uniform(-block_size, limit[0])
            ybeg = np.random.uniform(-block_size, limit[1])
            xbeg_list.append(xbeg)
            ybeg_list.append(ybeg)
    # data_label_filename = data_label_filename[:-4].split('//')
    data_label_filename = data_label_filename[:-4].split('\\')  # 1
    data_label_filename = data_label_filename[len(data_label_filename) - 1]  # Run1_WT1_1
    # test_area = data_label_filename[7]  # 5
    # room_name = data_label_filename[9:]  # 7:
    '''水稻文件时使用'''
    test_area = data_label_filename.split('_')[0]  # 5
    room_name = str(data_label_filename.split('_')[2])+"_"+str(data_label_filename.split('_')[3])+"_"+str(data_label_filename.split('_')[4])  # 7:

    raw_data_path_1 = "C:/Users/tang/Desktop/true_wheat_sem_seg_hdf5_data_test"
    raw_data_path = "C:/Users/tang/Desktop/true_wheat_sem_seg_hdf5_data_test/raw_data3d"
    # raw_data_path_1 = "../data/true_wheat_sem_seg_hdf5_data_test"
    # raw_data_path = "../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d"
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)
    if not os.path.exists(str(raw_data_path) + "\\Sample_" + str(test_area)):
        os.makedirs(str(raw_data_path) + "\\Sample_" + str(test_area))
    '''原始代码'''
    # if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d"):
    #     os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d")
    # if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_" + str(test_area)):
    #     os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_" + str(test_area))

    # Collect blocks
    # block_datas: K * num_point * 6 np array of XYZRGB, RGB is in [0,1]
    # block_labels: K * num_point * 1 np array of uint8 labels
    metrics_1 = {room_name: [], 'sampled': []}
    block_data_list = []
    block_label_list = []
    global raw_data_index  # 声明为全局数据（原始数据的索引）
    # 遍历xbeg_list的值，并赋给xbeg，遍历ybeg_list的值，并赋给ybeg，
    for idx in range(len(xbeg_list)):
        xbeg = xbeg_list[idx]
        ybeg = ybeg_list[idx]
        if idx == len(xbeg_list)-1:
            xcond = (data[:, 0] >= xbeg)  # 判断该区块中的x值有多少，将整个data中的X值与block的大小相匹配
            ycond = (data[:, 1] >= ybeg)  # 判断该区块中的x值有多少，将整个data中的y值与block的大小相匹配
        else:   # 100 50
            xcond = ((data[:, 0] <= xbeg + 100) & (data[:, 0] >= xbeg))  # 判断该区块中的x值有多少，将整个data中的X值与block的大小相匹配
            ycond = ((data[:, 1] <= ybeg + 50) & (data[:, 1] >= ybeg))  # 判断该区块中的x值有多少，将整个data中的y值与block的大小相匹配
        cond = xcond & ycond  # 求xcond与ycond的交集，查看是否在
        ###原始的数据集加载
        if np.sum(cond) <= 10:  # discard block if there are less than 100 pts.  如果少于 100 点，则丢弃块。 #100
            print("number is not enough discard")
            continue
        block_data = data[cond, :]  ##丢弃了小于100个点的块之后剩下的，在wheat数据集中每个块均大于100个点，所以没有丢弃的块，在S3DIS中由于点云的缺失，有丢弃的块，此处的阈值可以调高
        block_label = label[cond]

        # randomly subsample data   随机重采样数据
        block_data_sampled, block_label_sampled = \
            sample_data_label(block_data, block_label, num_point)  # num_point
        block_data_list.append(np.expand_dims(block_data_sampled, 0))  # block_data_sampled
        block_label_list.append(np.expand_dims(block_label_sampled, 0))  # block_label_sampled
        '''制作shapenet全监督学习的数据集时使用'''
        #  # 将ndarray转换为PyTorch的Tensor
        # x_tensor = torch.from_numpy(block_data_sampled)
        # y_tensor = torch.from_numpy(block_label_sampled)
        # #   # 将y_tensor的维度从 (2,) 转换为 (2, 1)
        # y_tensor = y_tensor.unsqueeze(1).int()
        # #   # 在维度1上拼接x_tensor和y_tensor
        # z = torch.cat((x_tensor, y_tensor), dim=1)
        # f = open(str(raw_data_path) + '/Sample_' + str(test_area) + '/' + str(room_name) + '(' + str(raw_data_index) + ').txt', "w+")
        # with open(str(raw_data_path) + '/Sample_' + str(test_area) + '/' + str(room_name) + '(' + str(raw_data_index) + ').txt', "w+") as f:
        #     np.savetxt(f, z, fmt='%.6f', delimiter=' ')

        '''制作shapenet无监督学习的数据集时使用'''
        # f = open('../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_' + str(test_area) + '/' + str( room_name) + '(' + str(raw_data_index) + ').txt', "w+")
        # np.savetxt(f, block_data_sampled, fmt='%s', delimiter=' ')   # block_data_sampled

        f = open(str(raw_data_path) + '/Sample_' + str(test_area) + '/' + str(room_name) + '(' + str(
            raw_data_index) + ').txt', "w+")
        with open(str(raw_data_path) + '/Sample_' + str(test_area) + '/' + str(room_name) + '(' + str(
                raw_data_index) + ').txt', "w+") as f:
            np.savetxt(f, block_data_sampled[:, 0:], fmt='%.6f', delimiter=' ')

        raw_data_index = raw_data_index + 1
        metrics_1[room_name].append(block_data.shape[0])
        metrics_1['sampled'].append(block_data_sampled[:, 0:3].shape[0])

    # 保存为Excel文件
    df = pd.DataFrame(metrics_1)
    root = 'C:/Users/tang/Desktop/true_wheat_sem_seg_hdf5_data_test\\statistic\\'
    if not os.path.exists(root):
        os.makedirs(root)
    save_path_1 = root + room_name + '.xlsx'  # 指定文件夹路径
    df.to_excel(save_path_1, index=False)
    return np.concatenate(block_data_list, 0), \
           np.concatenate(block_label_list, 0)



# 加入法向量和采样
def room2blocks_1_sample(data_label_filename, data, label, num_point, data_label, block_size=1.0, stride=1.0, random_sample=False, sample_num=None, sample_aug=1):  # block_size=1.0
    # def room2blocks(data_label_filename, data, label, num_point, block_size=1.0, stride=1.0, random_sample=True, sample_num=18, sample_aug=2):  # block_size=1.0
    """ Prepare block training data.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and aligned
            (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters 浮点数，块的物理大小（以米为单位）
        stride: float, stride for block sweeping  浮动，对小块扫描时的步长
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area][默认为房间的数量]
        sample_aug: if random sample, how much aug
    Returns:
        block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
        block_labels: K x num_point x 1 np array of uint8 labels

    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert (stride <= block_size)

    # 将取出的最大值作为限制阈值
    limit = np.amax(data, 0)[0:3]

    # Get the corner location for our sampling blocks  获取采样块的角落位置
    xbeg_list = []  # 计算(x * y)的值
    ybeg_list = []  # 计算(x * y)的值
    # 如果random_sample的值为ture，就会自动采样，sample_num确定采样点数
    # 如果random_sample的值为false，就不会自动采样.
    if not random_sample:  # abs返回绝对值
        num_block_x = int(abs(np.ceil(limit[0] / 2.49))-1)  # np.ceil：向上取整。计算x划分的块数，通过三个参数调整limit[0]（最大值的第1列，即x的值） ， block_size和stride
        num_block_y = int(abs(np.ceil(limit[1] / 0.8))-1)  # 计算y划分的块数。通过三个参数调整limit[1]（最大值的第1列，即y的值） ， block_size和stride
        # 对xy的值进行循环
        for i in range(num_block_x):
            for j in range(num_block_y):
                xbeg_list.append(i * 2.49)  # stride 每次滑动stride的步长，将block的x和y储存在ybeg_list中
                ybeg_list.append(j * 0.8)  # stride
    else:
        num_block_x = int(np.ceil(limit[0] / block_size))
        num_block_y = int(np.ceil(limit[1] / block_size))
        if sample_num is None:
            sample_num = num_block_x * num_block_y * sample_aug
        for _ in range(sample_num):
            xbeg = np.random.uniform(-block_size, limit[0])
            ybeg = np.random.uniform(-block_size, limit[1])
            xbeg_list.append(xbeg)
            ybeg_list.append(ybeg)
    # data_label_filename = data_label_filename[:-4].split('//')
    data_label_filename = data_label_filename[:-4].split('\\')  # 1
    data_label_filename = data_label_filename[len(data_label_filename) - 1]  # Run1_WT1_1
    test_area = data_label_filename[7]  # 5
    room_name = data_label_filename[9:]  # 7:

    raw_data_path_1 = "D:\\a-project-T\\Data\\true_data\\true_wheat_sem_seg_hdf5_data_test"
    raw_data_path = "D:\\a-project-T\\Data\\true_data\\true_wheat_sem_seg_hdf5_data_test\\raw_data3d"
    # raw_data_path_1 = "../data/true_wheat_sem_seg_hdf5_data_test"
    # raw_data_path = "../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d"
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)
    if not os.path.exists(str(raw_data_path) + "\\Sample_" + str(test_area)):
        os.makedirs(str(raw_data_path) + "\\Sample_" + str(test_area))
    '''原始代码'''
    # if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d"):
    #     os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d")
    # if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_" + str(test_area)):
    #     os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_" + str(test_area))

    # Collect blocks
    # block_datas: K * num_point * 6 np array of XYZRGB, RGB is in [0,1]
    # block_labels: K * num_point * 1 np array of uint8 labels

    '''选择锚点的公式'''
    data_all = data_label[:, :]        # data_all = data_label[:, :4]
    a = 6000
    l = len(data[:, 0])
    '''计算每类的数量和锚点数量'''
    np.set_printoptions(suppress=True)
    indices_s = np.where(data_all[:, 6] == 1)[0]  # 取出标签为 1（spike）的点的索引                   [0,1,2,5,...]
    indices_l = np.where(data_all[:, 6] == 0)[0]  # 取出标签为 0（leaf） 的点的索引                   [6,7,8,9,...]
    indices_ = np.where((data_all[:, 6] == 1) | (data_all[:, 6] == 0))[0]  # 取出所有标签点的索引    [0,1,2,3,...]

    values_s = data_all[indices_s, 2]  # 取出标签为 1（spike） 的点的第三列的值
    values_l = data_all[indices_l, 2]  # 取出标签为 0（leaf） 的点的第三列的值
    values_ = data_all[indices_, 2]    # 取出所有标签点的第三列的值

    max_s = torch.argmax(torch.tensor(values_s))  # 求出spike在z轴上最大值的索引
    min_s = torch.argmin(torch.tensor(values_s))  # 求出spike在z轴上最小值的索引
    max_l = torch.argmax(torch.tensor(values_l))  # 求出leaf在z轴上最大值的索引
    min_l = torch.argmin(torch.tensor(values_l))  # 求出leaf在z轴上最小值的索引

    points_ = data_all[indices_, :]  # 取出标签为所有点的第一、二、三、四列作为点的坐标   data_all[indices_, :4]
    points_s = data_all[indices_s, :]  # 取出标签为 1（spike） 的点的第一、二、三、四列作为点的坐标    data_all[indices_s, :4]
    points_l = data_all[indices_l, :]  # 取出标签为 0（leaf） 的点的第一、二、三、四列作为点的坐标     data_all[indices_l, :4]

    max_s_point = points_s[max_s]  # 取出spike在第三列上最大值的点的坐标
    min_s_point = points_s[min_s]  # 取出spike在第三列上最小值的点的坐标
    max_l_point = points_l[max_l]  # 取出leaf在第三列上最大值的点的坐标
    min_l_point = points_l[min_l]  # 取出leaf在第三列上最小值的点的坐标

    '''将整个点云划分为2个部分'''
    cal = max_l_point[2] / 2
    if min_s_point[2] <= cal:
        cal = min_s_point[2] - min_s_point[2] / 2

    sam_indices_s1 = np.where((values_ <= cal))[0]  # 在所有标签中，取出在二分之一以下之间的点的索引
    sam_indices_s2 = np.where((values_ >= cal))[0]  # 在所有标签中，取出在二分之一以上最大值和最小值之间的点的索引

    sam_points_s1 = data_all[indices_[sam_indices_s1], :]  # 取出这些点的坐标    data_all[indices_[sam_indices_s1], :4]
    sam_points_s2 = data_all[indices_[sam_indices_s2], :]  # 取出这些点的坐标    data_all[indices_[sam_indices_s2], :4]
    # if zcond = ((min_spike-1 < data_all[index_ones][:, 2]) & (data_all[index_ones][:, 2] <= max_spike)):

    '''定义二分之一以下的锚点采样'''  # 定义随机采样器，在spike类中每次采样 4096 个点。replacement=False每个点只会被采样一次
    sampler_spike_1 = RandomSampler(sam_points_s1, replacement=False, num_samples=int(0.3 * a))  # data_spike #RandomSampler(采样数据集, replacement=False采样是否放回, num_samples=采样点数)
    # 定义 DataLoader，batch_size 设置为 None，表示每个 batch 的大小等于采样器的 num_samples
    dataloader_spike_1 = DataLoader(sam_points_s1, batch_size=1, sampler=sampler_spike_1)  # None   #data_spike
    block_spike = []  # 定义空列表
    for batch in dataloader_spike_1:  # 遍历dataloader，将随机采样的点保存在block中
        #squeeze()函数将其从形状为(1, 3)的张量压缩为形状为(3,)的张量，tolist()函数将其转换为 Python 列表
        block_spike.append(batch.squeeze().tolist())  # numpy()方法将其转换为 NumPy 数组   .astype('float32')浮点数类型的矩阵
    block_spike = np.array(block_spike).reshape(len(block_spike), 7)  # 将列表转换为NumPy数组arrarr，并使用reshapereshape()方法将其形状修改为 (77, 3)

    block_data_list = []
    block_label_list = []
    global raw_data_index  # 声明为全局数据（原始数据的索引）
    data_block = np.concatenate((block_spike, sam_points_s2), axis=0)  # 连接两个数组
    # 遍历xbeg_list的值，并赋给xbeg，遍历ybeg_list的值，并赋给ybeg，
    for idx in range(len(xbeg_list)):
        xbeg = xbeg_list[idx]
        ybeg = ybeg_list[idx]
        if idx == len(xbeg_list)-1:
            xcond = (data_block[:, 0] >= xbeg)  # 判断该区块中的x值有多少，将整个data中的X值与block的大小相匹配
            ycond = (data_block[:, 1] >= ybeg)  # 判断该区块中的x值有多少，将整个data中的y值与block的大小相匹配
        else:
            xcond = ((data_block[:, 0] <= xbeg + 2.49) & (data_block[:, 0] >= xbeg))  # 判断该区块中的x值有多少，将整个data中的X值与block的大小相匹配
            ycond = ((data_block[:, 1] <= ybeg + 0.8) & (data_block[:, 1] >= ybeg))  # 判断该区块中的x值有多少，将整个data中的y值与block的大小相匹配
        cond = xcond & ycond  # 求xcond与ycond的交集，查看是否在
        ###原始的数据集加载
        if np.sum(cond) <= 0:  # discard block if there are less than 100 pts.  如果少于 100 点，则丢弃块。 #100
            print("number is not enough discard")
            continue
        block_data = data_block[cond, :6]  ##丢弃了小于100个点的块之后剩下的，在wheat数据集中每个块均大于100个点，所以没有丢弃的块，在S3DIS中由于点云的缺失，有丢弃的块，此处的阈值可以调高
        block_label = data_block[cond, 6]

        # randomly subsample data   随机重采样数据
        block_data_sampled, block_label_sampled = \
            sample_data_label(block_data, block_label, num_point)  # num_point
        block_data_list.append(np.expand_dims(block_data_sampled, 0))  # block_data_sampled
        block_label_list.append(np.expand_dims(block_label_sampled, 0))  # block_label_sampled
        '''制作shapenet数据集时使用'''
        # 将ndarray转换为PyTorch的Tensor
        x_tensor = torch.from_numpy(block_data_sampled)
        y_tensor = torch.from_numpy(block_label_sampled)
        # 将y_tensor的维度从 (2,) 转换为 (2, 1)
        y_tensor = y_tensor.unsqueeze(1).int()
        # 在维度1上拼接x_tensor和y_tensor
        z = torch.cat((x_tensor, y_tensor), dim=1)
        f = open('../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_' + str(test_area) + '/' + str(room_name) + '(' + str(raw_data_index) + ').txt', "w+")
        np.savetxt(f, z, fmt='%s', delimiter=' ')  # block_data_sampled
        '''加入结束'''

        # f = open('../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_' + str(test_area) + '/' + str( room_name) + '(' + str(raw_data_index) + ').txt', "w+")
        # np.savetxt(f, block_data_sampled, fmt='%s', delimiter=' ')   # block_data_sampled

        raw_data_index = raw_data_index + 1
    return np.concatenate(block_data_list, 0), \
           np.concatenate(block_label_list, 0)


# 加入法向量，并将叶穗按位置分开采样,选择锚点时用最远点采样，对锚点
def room2blocks_method1(data_label_filename, data, label, num_point, data_label, block_size=1.0, stride=14, random_sample=False, sample_num=None, sample_aug=1):  # block_size=1.0
    # def room2blocks(data_label_filename, data, label, num_point, block_size=1.0, stride=1.0, random_sample=True, sample_num=18, sample_aug=2):  # block_size=1.0
    """ Prepare block training data.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and aligned
            (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters 浮点数，块的物理大小（以米为单位）
        stride: float, stride for block sweeping  浮动，对小块扫描时的步长
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area][默认为房间的数量]
        sample_aug: if random sample, how much aug
    Returns:
        block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
        block_labels: K x num_point x 1 np array of uint8 labels

    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert (stride <= block_size)

    # 将取出的最大值作为限制阈值
    limit = np.amax(data, 0)[0:3]

    # Get the corner location for our sampling blocks  获取采样块的角落位置
    xbeg_list = []  # 计算(x * y)的值
    ybeg_list = []  # 计算(x * y)的值
    # 如果random_sample的值为ture，就会自动采样，sample_num确定采样点数
    # 如果random_sample的值为false，就不会自动采样.
    if not random_sample:  # abs返回绝对值
        num_block_x = int(abs(np.ceil((limit[0] - block_size) / stride))) + 1  # np.ceil：向上取整。计算x划分的块数，通过三个参数调整limit[0]（最大值的第1列，即x的值） ， block_size和stride
        num_block_y = int(abs(np.ceil((limit[1] - block_size) / stride))) + 1  # 计算y划分的块数。通过三个参数调整limit[1]（最大值的第1列，即y的值） ， block_size和stride
        # 对xy的值进行循环
        for i in range(num_block_x):
            for j in range(num_block_y):
                xbeg_list.append(i * stride)  # 每次滑动stride的步长，将block的x和y储存在ybeg_list中
                ybeg_list.append(j * stride)
    else:
        num_block_x = int(np.ceil(limit[0] / block_size))
        num_block_y = int(np.ceil(limit[1] / block_size))
        if sample_num is None:
            sample_num = num_block_x * num_block_y * sample_aug
        for _ in range(sample_num):
            xbeg = np.random.uniform(-block_size, limit[0])
            ybeg = np.random.uniform(-block_size, limit[1])
            xbeg_list.append(xbeg)
            ybeg_list.append(ybeg)
    # data_label_filename = data_label_filename[:-4].split('//')
    data_label_filename = data_label_filename[:-4].split('\\')
    data_label_filename = data_label_filename[len(data_label_filename) - 1]
    test_area = data_label_filename[7]  # 5
    room_name = data_label_filename[9:]  # 7:

    raw_data_path_1 = "D:\\a-project-T\\Data\\true_wheat_data\\true_wheat_sem_seg_hdf5_data_test"
    raw_data_path = "D:\\a-project-T\\Data\\true_wheat_data\\true_wheat_sem_seg_hdf5_data_test\\raw_data3d"
    # raw_data_path_1 = "../data/true_wheat_sem_seg_hdf5_data_test"
    # raw_data_path = "../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d"
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)
    if not os.path.exists(str(raw_data_path) + "\\Sample_" + str(test_area)):
        os.makedirs(str(raw_data_path) + "\\Sample_" + str(test_area))
    '''原始代码'''
    # if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d"):
    #     os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d")
    # if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_" + str(test_area)):
    #     os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_" + str(test_area))

    # Collect blocks
    # block_datas: K * num_point * 6 np array of XYZRGB, RGB is in [0,1]
    # block_labels: K * num_point * 1 np array of uint8 labels

    '''选择锚点的公式'''
    data_all = data_label[:, :4]
    a = 25000
    n = 0
    m = 0
    l = len(data[:,0])
    '''计算每类的数量和锚点数量'''
    # 找出每个类别的数量
    spike = (data_all[:, 3] == 1).sum().item()
    leaf = (data_all[:, 3] == 0).sum().item()
    # 找出第四列值为0和1的索引
    index_zeros = (data_all[:, 3] == 0)
    index_ones = (data_all[:, 3] == 1)

    num_spike = int(np.ceil(((1-spike/l)*a) / ((1-spike/l)+(1-leaf/l))))   #计算穗的锚点数量int(np.ceil())
    num_leaf = int(np.ceil(((1-leaf/l)*a) / ((1-spike/l) + (1-leaf/l)) ))   #计算叶的锚点数量
    hd = -1 * ((spike/l) * math.log2(spike/l) + (leaf/l) * math.log2(leaf/l))    #香农熵公式量化类不平衡的水平
    print("香农熵的计算结果为：",hd)

    # 定义随机采样器，在spike类中每次采样 4096 个点。replacement=False每个点只会被采样一次
    sampler_spike = RandomSampler(data_all[index_ones], replacement=False, num_samples=num_spike)    #data_spike #RandomSampler(采样数据集, replacement=False采样是否放回, num_samples=采样点数)
    # 定义 DataLoader，batch_size 设置为 None，表示每个 batch 的大小等于采样器的 num_samples
    dataloader_spike = DataLoader(data_all[index_ones], batch_size=1, sampler=sampler_spike)  #None   #data_spike
    block_spike = []   #定义空列表
    for batch in dataloader_spike:    # 遍历dataloader，将随机采样的点保存在block中
        # print(batch.squeeze().tolist())   #squeeze()函数将其从形状为(1, 3)的张量压缩为形状为(3,)的张量，tolist()函数将其转换为 Python 列表
        block_spike.append(batch.squeeze().tolist())  #numpy()方法将其转换为 NumPy 数组   .astype('float32')浮点数类型的矩阵
    block_spike = np.array(block_spike).reshape(len(block_spike), 4)   #将列表转换为NumPy数组arrarr，并使用reshapereshape()方法将其形状修改为 (77, 3)
    # block_spike = torch.stack(block_spike)   # 将block中的点堆叠成一个张量

    # 定义随机采样器，在leaf类中每次采样 4096 个点。replacement=False每个点只会被采样一次
    sampler_leaf = RandomSampler(data_all[index_zeros], replacement=False, num_samples=num_leaf)    #data_leaf
    # 定义 DataLoader，batch_size 设置为 None，表示每个 batch 的大小等于采样器的 num_samples
    dataloader_leaf = DataLoader(data_all[index_zeros], batch_size=1, sampler=sampler_leaf)  #加载采样的锚点  #None   #data_leaf
    block_leaf = []
    for batch in dataloader_leaf:    # 遍历dataloader，将随机采样的点保存在block中
        block_leaf.append(batch.squeeze().tolist())   # #squeeze()函数将其从形状为(1, 3)的张量压缩为形状为(3,)的张量，tolist()函数将其转换为 Python 列表
    # block_leaf = torch.stack(block_leaf)  # 将block中的点堆叠成一个张量
    block_leaf = np.array(block_leaf).reshape(len(block_leaf), 4)   #将采样的锚点从列表转为数组

    block_data_list = []
    block_label_list = []   #定义空列表
    global raw_data_index  # 声明为全局数据（原始数据的索引）
    # 遍历xbeg_list的值，并赋给xbeg，遍历ybeg_list的值，并赋给ybeg，
    for idx in range(len(xbeg_list)):
        xbeg = xbeg_list[idx]   #读取划分块的x轴列表
        ybeg = ybeg_list[idx]   #读取划分块的y轴列表
        xcond_spike = (block_spike[:, 0] <= xbeg + block_size) & (block_spike[:, 0] >= xbeg)  # 判断该区块中的x值有多少，将整个data中的X值与block的大小相匹配
        ycond_spike  = (block_spike[:, 1] <= ybeg + block_size) & (block_spike[:, 1] >= ybeg)  # 判断该区块中的x值有多少，将整个data中的y值与block的大小相匹配
        cond_spike  = xcond_spike  & ycond_spike   # 求xcond与ycond的交集，查看是否在包含锚点

        xcond_leaf = (block_leaf[:,0] <= xbeg + block_size) & (block_leaf[:,0] >= xbeg)  # 判断该区块中的x值有多少，将整个data中的X值与block的大小相匹配
        ycond_leaf = (block_leaf[:,1] <= ybeg + block_size) & (block_leaf[:,1] >= ybeg)  # 判断该区块中的x值有多少，将整个data中的y值与block的大小相匹配
        cond_leaf = xcond_leaf & ycond_leaf  # 求xcond与ycond的交集，查看是否包含锚点
        ###原始的数据集加载
        if (np.sum(cond_spike) < 40) | (np.sum(cond_leaf) <= 40) :  #求两个集合的并集 抛弃不包含锚点的块
            print("spike or leaf anchor are not in this block")
            continue
        # cond = cond_spike & cond_leaf
        cond = np.concatenate([cond_spike,cond_leaf])   #连接两个数组
        data_block = np.concatenate((block_spike,block_leaf),axis=0)   #连接两个数组
        '''Over'''

        block_data = data_block[cond, :3]  ##丢弃了小于100个点的块之后剩下的，在wheat数据集中每个块均大于100个点，所以没有丢弃的块，在S3DIS中由于点云的缺失，有丢弃的块，此处的阈值可以调高
        block_label = data_block[cond,3]

        # randomly subsample data   随机重采样数据
        block_data_sampled, block_label_sampled = \
            sample_data_label(block_data, block_label, num_point)  # num_point
        block_data_list.append(np.expand_dims(block_data_sampled, 0))  # block_data_sampled
        block_label_list.append(np.expand_dims(block_label_sampled, 0))  # block_data_sampled
        f = open('../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_' + str(test_area) + '/' + str(room_name) + '(' + str(raw_data_index) + ').txt', "w+")
        np.savetxt(f, block_data_sampled[:, 0:3], fmt='%s', delimiter=' ')
        raw_data_index = raw_data_index + 1
    return np.concatenate(block_data_list, 0), \
           np.concatenate(block_label_list, 0)


# 完全复制eff论文中的方法，计算锚点，对块进行KNN上采样
def room2blocks_method2(data_label_filename, data, label, num_point, data_label, block_size=1.0, stride=14, random_sample=False, sample_num=None, sample_aug=1):  # block_size=1.0
    # def room2blocks(data_label_filename, data, label, num_point, block_size=1.0, stride=1.0, random_sample=True, sample_num=18, sample_aug=2):  # block_size=1.0
    """ Prepare block training data.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and aligned
            (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters 浮点数，块的物理大小（以米为单位）
        stride: float, stride for block sweeping  浮动，对小块扫描时的步长
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area][默认为房间的数量]
        sample_aug: if random sample, how much aug
    Returns:
        block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
        block_labels: K x num_point x 1 np array of uint8 labels

    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert (stride <= block_size)

    # 将取出的最大值作为限制阈值

    # data_label_filename = data_label_filename[:-4].split('//')
    data_label_filename = data_label_filename[:-4].split('\\')
    data_label_filename = data_label_filename[len(data_label_filename) - 1]
    test_area = data_label_filename[7]  # 5
    room_name = data_label_filename[9:]  # 7:
    if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d"):
        os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d")
    if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_" + str(test_area)):
        os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_" + str(test_area))
    # Collect blocks
    # block_datas: K * num_point * 6 np array of XYZRGB, RGB is in [0,1]
    # block_labels: K * num_point * 1 np array of uint8 labels

    '''选择锚点的公式'''
    data_all = data_label[:, :4]
    a = 300
    l = len(data[:,0])
    '''计算每类的数量和锚点数量'''
    # 找出每个类别的数量
    spike = (data_all[:, 3] == 1).sum().item()
    leaf = (data_all[:, 3] == 0).sum().item()
    # 找出第四列值为0和1的索引
    index_zeros = (data_all[:, 3] == 0)
    index_ones = (data_all[:, 3] == 1)

    num_spike = int(np.ceil(((1-spike/l)*a) / ((1-spike/l)+(1-leaf/l))))   #计算穗的锚点数量int(np.ceil())
    num_leaf = int(np.ceil(((1-leaf/l)*a) / ((1-spike/l) + (1-leaf/l)) ))   #计算叶的锚点数量
    hd = -1 * ((spike/l) * math.log2(spike/l) + (leaf/l) * math.log2(leaf/l))    #香农熵公式量化类不平衡的水平
    print("原始数据集的香农熵的计算结果为：",hd)

    # 定义随机采样器，在spike类中每次采样 4096 个点。replacement=False每个点只会被采样一次
    sampler_spike = RandomSampler(data_all[index_ones], replacement=False, num_samples=num_spike)    #data_spike #RandomSampler(采样数据集, replacement=False采样是否放回, num_samples=采样点数)
    # 定义 DataLoader，batch_size 设置为 None，表示每个 batch 的大小等于采样器的 num_samples
    dataloader_spike = DataLoader(data_all[index_ones], batch_size=1, sampler=sampler_spike)  #None   #data_spike
    block_spike = []   #定义空列表
    for batch in dataloader_spike:    # 遍历dataloader，将随机采样的点保存在block中
        # print(batch.squeeze().tolist())   #squeeze()函数将其从形状为(1, 3)的张量压缩为形状为(3,)的张量，tolist()函数将其转换为 Python 列表
        block_spike.append(batch.squeeze().tolist())  #numpy()方法将其转换为 NumPy 数组   .astype('float32')浮点数类型的矩阵
    block_spike = np.array(block_spike).reshape(len(block_spike), 4)   #将列表转换为NumPy数组arrarr，并使用reshapereshape()方法将其形状修改为 (77, 3)
    # block_spike = torch.stack(block_spike)   # 将block中的点堆叠成一个张量

    # 定义随机采样器，在leaf类中每次采样 4096 个点。replacement=False每个点只会被采样一次
    sampler_leaf = RandomSampler(data_all[index_zeros], replacement=False, num_samples=num_leaf)    #data_leaf
    # 定义 DataLoader，batch_size 设置为 None，表示每个 batch 的大小等于采样器的 num_samples
    dataloader_leaf = DataLoader(data_all[index_zeros], batch_size=1, sampler=sampler_leaf)  #加载采样的锚点  #None   #data_leaf
    block_leaf = []
    for batch in dataloader_leaf:    # 遍历dataloader，将随机采样的点保存在block中
        block_leaf.append(batch.squeeze().tolist())   # #squeeze()函数将其从形状为(1, 3)的张量压缩为形状为(3,)的张量，tolist()函数将其转换为 Python 列表
    # block_leaf = torch.stack(block_leaf)  # 将block中的点堆叠成一个张量
    block_leaf = np.array(block_leaf).reshape(len(block_leaf), 4)   #将采样的锚点从列表转为数组

    block_data_list = []
    block_label_list = []   #定义空列表
    global raw_data_index  # 声明为全局数据（原始数据的索引）

    # 直接对锚点进行KNN（k-nearest-neighbour）上采样
    data_block = np.concatenate((block_spike, block_leaf), axis=0)  # 连接两个数组
    # knn_indices = []
    # knn_distances = []
    for id in range(data_block.shape[0]):
        distances, indices = find_knn(data_all, [data_block[id]], kk=4096)
        indices_1 = indices[0]
        block_data_sampled = data_all[indices_1, 0:3]
        block_label_sampled = data_all[indices_1, 3]
        block_data_list.append(np.expand_dims(block_data_sampled, 0))  # block_data_sampled
        block_label_list.append(np.expand_dims(block_label_sampled, 0))  # block_data_sampled
        f = open('../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_' + str(test_area) + '/' + str(
            room_name) + '(' + str(raw_data_index) + ').txt', "w+")
        np.savetxt(f, block_data_sampled[:, 0:3], fmt='%s', delimiter=' ')
        raw_data_index = raw_data_index + 1


    return np.concatenate(block_data_list, 0), \
           np.concatenate(block_label_list, 0)


# 按类别：将点云分成两个部分，计算锚点，将原本的KNN上采样替换为行随机下采样
def room2blocks_method3(data_label_filename, data, label, num_point, data_label, block_size=1.0, stride=14, random_sample=False, sample_num=None, sample_aug=1):  # block_size=1.0
    # def room2blocks(data_label_filename, data, label, num_point, block_size=1.0, stride=1.0, random_sample=True, sample_num=18, sample_aug=2):  # block_size=1.0
    """ Prepare block training data.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and aligned
            (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters 浮点数，块的物理大小（以米为单位）
        stride: float, stride for block sweeping  浮动，对小块扫描时的步长
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area][默认为房间的数量]
        sample_aug: if random sample, how much aug
    Returns:
        block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
        block_labels: K x num_point x 1 np array of uint8 labels

    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert (stride <= block_size)

    # 将取出的最大值作为限制阈值

    # data_label_filename = data_label_filename[:-4].split('//')
    data_label_filename = data_label_filename[:-4].split('\\')
    data_label_filename = data_label_filename[len(data_label_filename) - 1]
    test_area = data_label_filename[7]  # 5
    room_name = data_label_filename[9:]  # 7:
    if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d"):
        os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d")
    if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_" + str(test_area)):
        os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_" + str(test_area))
    # Collect blocks
    # block_datas: K * num_point * 6 np array of XYZRGB, RGB is in [0,1]
    # block_labels: K * num_point * 1 np array of uint8 labels

    '''选择锚点的公式'''
    data_all = data_label[:, :4]
    a = 300
    l = len(data[:,0])
    '''计算每类的数量和锚点数量'''
    # 找出每个类别的数量
    spike = (data_all[:, 3] == 1).sum().item()
    leaf = (data_all[:, 3] == 0).sum().item()
    # 找出第四列值为0和1的索引
    index_zeros = (data_all[:, 3] == 0)
    index_ones = (data_all[:, 3] == 1)

    num_spike = int(np.ceil(((1-spike/l)*a) / ((1-spike/l)+(1-leaf/l))))   #计算穗的锚点数量int(np.ceil())
    num_leaf = int(np.ceil(((1-leaf/l)*a) / ((1-spike/l) + (1-leaf/l)) ))   #计算叶的锚点数量
    hd = -1 * ((spike/l) * math.log2(spike/l) + (leaf/l) * math.log2(leaf/l))    #香农熵公式量化类不平衡的水平
    print("原始数据集的香农熵的计算结果为：",hd)

    # 定义随机采样器，在spike类中每次采样 4096 个点。replacement=False每个点只会被采样一次
    sampler_spike = RandomSampler(data_all[index_ones], replacement=False, num_samples=num_spike)    #data_spike #RandomSampler(采样数据集, replacement=False采样是否放回, num_samples=采样点数)
    # 定义 DataLoader，batch_size 设置为 None，表示每个 batch 的大小等于采样器的 num_samples
    dataloader_spike = DataLoader(data_all[index_ones], batch_size=1, sampler=sampler_spike)  #None   #data_spike
    block_spike = []   #定义空列表
    for batch in dataloader_spike:    # 遍历dataloader，将随机采样的点保存在block中
        # print(batch.squeeze().tolist())   #squeeze()函数将其从形状为(1, 3)的张量压缩为形状为(3,)的张量，tolist()函数将其转换为 Python 列表
        block_spike.append(batch.squeeze().tolist())  #numpy()方法将其转换为 NumPy 数组   .astype('float32')浮点数类型的矩阵
    block_spike = np.array(block_spike).reshape(len(block_spike), 4)   #将列表转换为NumPy数组arrarr，并使用reshapereshape()方法将其形状修改为 (77, 3)
    # block_spike = torch.stack(block_spike)   # 将block中的点堆叠成一个张量

    # 定义随机采样器，在leaf类中每次采样 4096 个点。replacement=False每个点只会被采样一次
    sampler_leaf = RandomSampler(data_all[index_zeros], replacement=False, num_samples=num_leaf)    #data_leaf
    # 定义 DataLoader，batch_size 设置为 None，表示每个 batch 的大小等于采样器的 num_samples
    dataloader_leaf = DataLoader(data_all[index_zeros], batch_size=1, sampler=sampler_leaf)  #加载采样的锚点  #None   #data_leaf
    block_leaf = []
    for batch in dataloader_leaf:    # 遍历dataloader，将随机采样的点保存在block中
        block_leaf.append(batch.squeeze().tolist())   # #squeeze()函数将其从形状为(1, 3)的张量压缩为形状为(3,)的张量，tolist()函数将其转换为 Python 列表
    # block_leaf = torch.stack(block_leaf)  # 将block中的点堆叠成一个张量
    block_leaf = np.array(block_leaf).reshape(len(block_leaf), 4)   #将采样的锚点从列表转为数组

    block_data_list = []
    block_label_list = []   #定义空列表
    global raw_data_index  # 声明为全局数据（原始数据的索引）

    # 直接对锚点进行KNN（k-nearest-neighbour）上采样
    data_block = np.concatenate((block_spike, block_leaf), axis=0)  # 连接两个数组
    # knn_indices = []
    data_1 = np.empty((0,3))
    label_1 = np.empty((0, 1))
    for id in range(data_block.shape[0]):
        data_1 = data_block[id, 0:3].reshape((1, 3))
        label_1 = data_block[id, 3].ravel()   #.ravel将数组变成一维数组
        # randomly subsample data   随机重采样数据
        block_data_sampled, block_label_sampled = \
            sample_data_label(data_1, label_1, num_point)  # num_point

        block_data_list.append(np.expand_dims(block_data_sampled, 0))  # block_data_sampled
        block_label_list.append(np.expand_dims(block_label_sampled, 0))  # block_data_sampled
        f = open('../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_' + str(test_area) + '/' + str(
            room_name) + '(' + str(raw_data_index) + ').txt', "w+")
        np.savetxt(f, block_data_sampled[:, 0:3], fmt='%s', delimiter=' ')
        raw_data_index = raw_data_index + 1


    return np.concatenate(block_data_list, 0), \
           np.concatenate(block_label_list, 0)


# 按位置：将点云分成两个部分，穗和叶两个部分，对两个位置分别进行随机下采样
def room2blocks_method1(data_label_filename, data, label, num_point, data_label, block_size=1.0, stride=14, random_sample=False, sample_num=None, sample_aug=1):  # block_size=1.0
    # def room2blocks(data_label_filename, data, label, num_point, block_size=1.0, stride=1.0, random_sample=True, sample_num=18, sample_aug=2):  # block_size=1.0
    """ Prepare block training data.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and aligned
            (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters 浮点数，块的物理大小（以米为单位）
        stride: float, stride for block sweeping  浮动，对小块扫描时的步长
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area][默认为房间的数量]
        sample_aug: if random sample, how much aug
    Returns:
        block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
        block_labels: K x num_point x 1 np array of uint8 labels

    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert (stride <= block_size)

    # 将取出的最大值作为限制阈值
    limit = np.amax(data, 0)[0:3]

    # Get the corner location for our sampling blocks  获取采样块的角落位置
    xbeg_list = []  # 计算(x * y)的值
    ybeg_list = []  # 计算(x * y)的值
    # 如果random_sample的值为ture，就会自动采样，sample_num确定采样点数
    # 如果random_sample的值为false，就不会自动采样.
    if not random_sample:  # abs返回绝对值
        num_block_x = int(abs(np.floor(limit[0] / 2.49)))  # np.ceil：向上取整。计算x划分的块数，通过三个参数调整limit[0]（最大值的第1列，即x的值） ， block_size和stride
        num_block_y = int(abs(np.floor(limit[1] / 0.4)))  # 计算y划分的块数。通过三个参数调整limit[1]（最大值的第1列，即y的值） ， block_size和stride
        # num_block_x = int(abs(np.ceil((limit[0] - block_size) / stride))) + 1  # np.ceil：向上取整。计算x划分的块数，通过三个参数调整limit[0]（最大值的第1列，即x的值） ， block_size和stride
        # num_block_y = int(abs(np.ceil((limit[1] - block_size) / stride))) + 1  # 计算y划分的块数。通过三个参数调整limit[1]（最大值的第1列，即y的值） ， block_size和stride
        # 对xy的值进行循环
        for i in range(num_block_x):
            for j in range(num_block_y):
                xbeg_list.append(i * 2.49)  # stride 每次滑动stride的步长，将block的x和y储存在ybeg_list中
                ybeg_list.append(j * 0.4)  # stride
                # xbeg_list.append(i * stride)  # 每次滑动stride的步长，将block的x和y储存在ybeg_list中
                # ybeg_list.append(j * stride)

    # data_label_filename = data_label_filename[:-4].split('//')
    data_label_filename = data_label_filename[:-4].split('\\')
    data_label_filename = data_label_filename[len(data_label_filename) - 1]
    test_area = data_label_filename[7]  # 5
    room_name = data_label_filename[9:]  # 7:

    raw_data_path_1 = "D:\\a-project-T\\Data\\true_wheat_data\\true_wheat_sem_seg_hdf5_data_test"
    raw_data_path = "D:\\a-project-T\\Data\\true_wheat_data\\true_wheat_sem_seg_hdf5_data_test\\raw_data3d"
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)
    if not os.path.exists(str(raw_data_path) + "\\Sample_" + str(test_area)):
        os.makedirs(str(raw_data_path) + "\\Sample_" + str(test_area))
    # if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d"):
    #     os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d")
    # if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_" + str(test_area)):
    #     os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_" + str(test_area))
    # Collect blocks
    # block_datas: K * num_point * 6 np array of XYZRGB, RGB is in [0,1]
    # block_labels: K * num_point * 1 np array of uint8 labels

    '''选择锚点的公式'''
    data_all = data_label[:, :4]
    a = 60000
    l = len(data[:,0])
    '''计算每类的数量和锚点数量'''
    # 找出每个类别的数量
    spike = (data_all[:, 3] == 1).sum().item()    # 找出spike的数量
    leaf = (data_all[:, 3] == 0).sum().item()    # 找出leaf的数量
    # 找出第四列值为0和1的索引
    indices_s = np.where(data_all[:, 5] == 1)[0]    # 取出标签为 1（spike）的点的索引
    indices_l = np.where(data_all[:, 5] == 0)[0]  # 取出标签为 0（leaf） 的点的索引
    indices_ = np.where((data_all[:, 5] == 1)|(data_all[:, 3] == 0))[0]  # 取出标签为 0（leaf） 的点的索引

    values_s = data_all[indices_s, 0]    # 取出标签为 1（spike） 的点的第三列的值
    values_l = data_all[indices_l, 0]    # 取出标签为 0（leaf） 的点的第三列的值
    values_ = data_all[indices_, 0]  # 取出标签为 0（leaf） 的点的第三列的值

    max_s = torch.argmax(torch.tensor(values_s))    # 求出spike最大值的索引
    min_s = torch.argmin(torch.tensor(values_s))    # 求出spike最小值的索引
    max_l = torch.argmax(torch.tensor(values_l))    # 求出leaf最大值的索引
    min_l = torch.argmin(torch.tensor(values_l))    # 求出leaf最小值的索引

    points_s = data_all[indices_s, :4]    # 取出标签为 1（spike） 的点的第一、二、三列作为点的坐标
    points_l = data_all[indices_l, :4]    # 取出标签为 0（leaf） 的点的第一、二、三列作为点的坐标

    max_s_point = points_s[max_s]    # 取出spike在第三列上最大值的点的坐标
    min_s_point = points_s[min_s]     # 取出spike在第三列上最小值的点的坐标
    max_l_point = points_l[max_l]    # 取出leaf在第三列上最大值的点的坐标
    min_l_point = points_l[min_l]  # 取出leaf在第三列上最小值的点的坐标

    sam_indices_s1 = np.where((values_s >= min_s_point[2]) & (values_s <= max_s_point[2]))[0]  # 在所有标签中，取出在三分之一最大值和最小值之间的点的索引
    # sam_indices_s1 = np.where((values_ >= min_s_point[2]) & (values_ <= max_s_point[2]))[0]  # 在所有标签中，取出在三分之一最大值和最小值之间的点的索引
    sam_indices_l = np.where((values_l >= min_l_point[2]) & (values_l <= max_l_point[2]))[0]  # 取出在三分之一最大值和最小值之间的点的索引
    # sam_indices_l = np.where((values_l >= min_l_point[2]) & (values_l <= min_s_point[2]))[0]  # 取出在三分之一最大值和最小值之间的点的索引

    sam_points_s1 = data_all[indices_s[sam_indices_s1], :4]    # 取出这些点的坐标
    # sam_points_s1 = data_all[indices_[sam_indices_s1], :4]  # 取出这些点的坐标
    sam_points_l = data_all[indices_l[sam_indices_l], :4]  # 取出这些点的坐标
    # if zcond = ((min_spike-1 < data_all[index_ones][:, 2]) & (data_all[index_ones][:, 2] <= max_spike)):

    '''定义叶穗交接处的锚点采样'''# 定义随机采样器，在spike类中每次采样 4096 个点。replacement=False每个点只会被采样一次
    sampler_spike_1 = RandomSampler(sam_points_s1, replacement=False, num_samples=int(0.9*a))    #data_spike #RandomSampler(采样数据集, replacement=False采样是否放回, num_samples=采样点数)
    # 定义 DataLoader，batch_size 设置为 None，表示每个 batch 的大小等于采样器的 num_samples
    dataloader_spike_1 = DataLoader(sam_points_s1, batch_size=1, sampler=sampler_spike_1)  #None   #data_spike
    block_spike = []   #定义空列表
    for batch in dataloader_spike_1:    # 遍历dataloader，将随机采样的点保存在block中
        # print(batch.squeeze().tolist())   #squeeze()函数将其从形状为(1, 3)的张量压缩为形状为(3,)的张量，tolist()函数将其转换为 Python 列表
        block_spike.append(batch.squeeze().tolist())  #numpy()方法将其转换为 NumPy 数组   .astype('float32')浮点数类型的矩阵
    block_spike = np.array(block_spike).reshape(len(block_spike), 4)   #将列表转换为NumPy数组arrarr，并使用reshapereshape()方法将其形状修改为 (77, 3)
    # block_spike = torch.stack(block_spike)   # 将block中的点堆叠成一个张量

    '''定义只有叶的锚点采样'''# 定义随机采样器，在leaf类中每次采样 4096 个点。replacement=False每个点只会被采样一次
    sampler_leaf = RandomSampler(sam_points_l, replacement=False, num_samples=int(0.1*a))    #data_leaf
    # 定义 DataLoader，batch_size 设置为 None，表示每个 batch 的大小等于采样器的 num_samples
    dataloader_leaf = DataLoader(sam_points_l, batch_size=1, sampler=sampler_leaf)  #加载采样的锚点  #None   #data_leaf
    block_leaf = []
    for batch in dataloader_leaf:    # 遍历dataloader，将随机采样的点保存在block中
        block_leaf.append(batch.squeeze().tolist())   # #squeeze()函数将其从形状为(1, 3)的张量压缩为形状为(3,)的张量，tolist()函数将其转换为 Python 列表
    # block_leaf = torch.stack(block_leaf)  # 将block中的点堆叠成一个张量
    block_leaf = np.array(block_leaf).reshape(len(block_leaf), 4)   #将采样的锚点从列表转为数组

    block_data_list = []
    block_label_list = []   #定义空列表
    global raw_data_index  # 声明为全局数据（原始数据的索引）

    data_block = np.concatenate((block_spike, block_leaf), axis=0)  # 连接两个数组
    data_1 = np.empty((0,3))
    label_1 = np.empty((0, 1))

    for idx in range(len(xbeg_list)):
        xbeg = xbeg_list[idx]   #读取划分块的x轴列表
        ybeg = ybeg_list[idx]   #读取划分块的y轴列表
        xcond_spike = ( data_block[:, 0] <= xbeg + block_size) & ( data_block[:, 0] >= xbeg)  # 判断该区块中的x值有多少，将整个data中的X值与block的大小相匹配
        ycond_spike  = ( data_block[:, 1] <= ybeg + block_size) & ( data_block[:, 1] >= ybeg)  # 判断该区块中的x值有多少，将整个data中的y值与block的大小相匹配
        cond  = xcond_spike & ycond_spike   # 求xcond与ycond的交集，查看是否在包含锚点

        ###原始的数据集加载
        if (np.sum(cond) <= 0):  #求两个集合的并集 抛弃不包含锚点的块
            print("spike or leaf anchor are not in this block")
            continue
        '''Over'''

        block_data = data_block[cond, :3]  ##丢弃了小于100个点的块之后剩下的，在wheat数据集中每个块均大于100个点，所以没有丢弃的块，在S3DIS中由于点云的缺失，有丢弃的块，此处的阈值可以调高
        block_label = data_block[cond,3]

        # randomly subsample data   随机重采样数据
        block_data_sampled, block_label_sampled = \
            sample_data_label(block_data, block_label, num_point)  # num_point
        block_data_list.append(np.expand_dims(block_data_sampled, 0))  # block_data_sampled
        block_label_list.append(np.expand_dims(block_label_sampled, 0))  # block_data_sampled
        f = open('../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_' + str(test_area) + '/' + str(room_name) + '(' + str(raw_data_index) + ').txt', "w+")
        np.savetxt(f, block_data_sampled[:, 0:3], fmt='%s', delimiter=' ')
        raw_data_index = raw_data_index + 1

    return np.concatenate(block_data_list, 0), \
           np.concatenate(block_label_list, 0)


# 按位置：直接对anchor进行KNN上采样——耗时
def room2blocks_method5(data_label_filename, data, label, num_point, data_label, block_size=1.0, stride=14, random_sample=False, sample_num=None, sample_aug=1):  # block_size=1.0
    # def room2blocks(data_label_filename, data, label, num_point, block_size=1.0, stride=1.0, random_sample=True, sample_num=18, sample_aug=2):  # block_size=1.0
    """ Prepare block training data.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and aligned
            (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters 浮点数，块的物理大小（以米为单位）
        stride: float, stride for block sweeping  浮动，对小块扫描时的步长
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area][默认为房间的数量]
        sample_aug: if random sample, how much aug
    Returns:
        block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
        block_labels: K x num_point x 1 np array of uint8 labels

    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert (stride <= block_size)

    # 将取出的最大值作为限制阈值

    # data_label_filename = data_label_filename[:-4].split('//')
    data_label_filename = data_label_filename[:-4].split('\\')
    data_label_filename = data_label_filename[len(data_label_filename) - 1]
    test_area = data_label_filename[7]  # 5
    room_name = data_label_filename[9:]  # 7:
    if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d"):
        os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d")
    if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_" + str(test_area)):
        os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_" + str(test_area))
    # Collect blocks
    # block_datas: K * num_point * 6 np array of XYZRGB, RGB is in [0,1]
    # block_labels: K * num_point * 1 np array of uint8 labels

    '''选择锚点的公式'''
    data_all = data_label[:, :4]
    a = 300
    l = len(data[:,0])
    '''计算每类的数量和锚点数量'''
    # 找出每个类别的数量
    spike = (data_all[:, 3] == 1).sum().item()    # 找出spike的数量
    leaf = (data_all[:, 3] == 0).sum().item()    # 找出leaf的数量
    # 找出第四列值为0和1的索引
    index_zeros = (data_all[:, 3] == 0)       #spike的标签为1
    index_ones = (data_all[:, 3] == 1)        #leaf的标签为0

    indices_s = np.where(data_all[:, 3] == 1)[0]    # 取出标签为 1（spike）的点的索引
    indices_l = np.where(data_all[:, 3] == 0)[0]  # 取出标签为 0（leaf） 的点的索引
    indices_ = np.where((data_all[:, 3] == 1)|(data_all[:, 3] == 0))[0]  # 取出标签为 0（leaf） 的点的索引

    values_s = data_all[indices_s, 2]    # 取出标签为 1（spike） 的点的第三列的值
    values_l = data_all[indices_l, 2]    # 取出标签为 0（leaf） 的点的第三列的值
    values_ = data_all[indices_, 2]  # 取出标签为 0（leaf） 的点的第三列的值

    max_s = torch.argmax(torch.tensor(values_s))    # 求出spike最大值的索引
    min_s = torch.argmin(torch.tensor(values_s))    # 求出spike最小值的索引
    max_l = torch.argmax(torch.tensor(values_l))    # 求出leaf最大值的索引
    min_l = torch.argmin(torch.tensor(values_l))    # 求出leaf最小值的索引

    points_s = data_all[indices_s, :4]    # 取出标签为 1（spike） 的点的第一、二、三列作为点的坐标
    points_l = data_all[indices_l, :4]    # 取出标签为 0（leaf） 的点的第一、二、三列作为点的坐标

    max_s_point = points_s[max_s]    # 取出spike在第三列上最大值的点的坐标
    min_s_point = points_s[min_s]     # 取出spike在第三列上最大值的点的坐标
    max_l_point = points_l[max_l]    # 取出leaf在第三列上最大值的点的坐标
    min_l_point = points_l[min_l]  # 取出leaf在第三列上最小值的点的坐标

    # sam_indices_s1 = np.where((values_s >= min_s_point[2]) & (values_s <= max_s_point[2]))[0]  # 在所有标签中，取出在三分之一最大值和最小值之间的点的索引
    # sam_indices_l = np.where((values_l >= min_l_point[2]) & (values_l <= max_l_point[2]))[0]  # 取出在三分之一最大值和最小值之间的点的索引
    sam_indices_s1 = np.where((values_ >= min_s_point[2]) & (values_ <= max_s_point[2]))[0]  # 在所有标签中，取出在三分之一最大值和最小值之间的点的索引
    sam_indices_l = np.where((values_l >= min_l_point[2]) & (values_l <= max_l_point[2]))[0]  # 取出在三分之一最大值和最小值之间的点的索引

    # sam_points_s1 = data_all[indices_s[sam_indices_s1], :4]  # 在spike标签中，取出这些点的坐标
    # sam_points_s2 = data_all[indices_s[sam_indices_s2], :4]  # 在spike标签中，取出这些点的坐标
    sam_points_s1 = data_all[indices_[sam_indices_s1], :3]    # 取出这些点的坐标
    sam_points_l = data_all[indices_l[sam_indices_l], :3]  # 取出这些点的坐标
    # if zcond = ((min_spike-1 < data_all[index_ones][:, 2]) & (data_all[index_ones][:, 2] <= max_spike)):

    '''随机采样——定义叶穗交接处的锚点采样'''# 定义随机采样器，在spike类中每次采样 4096 个点。replacement=False每个点只会被采样一次
    # sampler_spike_1 = RandomSampler(sam_points_s1, replacement=False, num_samples=int(0.9*a))    #data_spike #RandomSampler(采样数据集, replacement=False采样是否放回, num_samples=采样点数)
    # # 定义 DataLoader，batch_size 设置为 None，表示每个 batch 的大小等于采样器的 num_samples
    # dataloader_spike_1 = DataLoader(sam_points_s1, batch_size=1, sampler=sampler_spike_1)  #None   #data_spike
    # block_spike = []   #定义空列表
    # for batch in dataloader_spike_1:    # 遍历dataloader，将随机采样的点保存在block中
    #     # print(batch.squeeze().tolist())   #squeeze()函数将其从形状为(1, 3)的张量压缩为形状为(3,)的张量，tolist()函数将其转换为 Python 列表
    #     block_spike.append(batch.squeeze().tolist())  #numpy()方法将其转换为 NumPy 数组   .astype('float32')浮点数类型的矩阵
    # block_spike = np.array(block_spike).reshape(len(block_spike), 4)   #将列表转换为NumPy数组arrarr，并使用reshapereshape()方法将其形状修改为 (77, 3)
    # # block_spike = torch.stack(block_spike)   # 将block中的点堆叠成一个张量

    '''随机采样——定义只有叶的锚点采样'''# 定义随机采样器，在leaf类中每次采样 4096 个点。replacement=False每个点只会被采样一次
    # sampler_leaf = RandomSampler(sam_points_l, replacement=False, num_samples=int(0.1*a))    #data_leaf
    # # 定义 DataLoader，batch_size 设置为 None，表示每个 batch 的大小等于采样器的 num_samples
    # dataloader_leaf = DataLoader(sam_points_l, batch_size=1, sampler=sampler_leaf)  #加载采样的锚点  #None   #data_leaf
    # block_leaf = []
    # for batch in dataloader_leaf:    # 遍历dataloader，将随机采样的点保存在block中
    #     block_leaf.append(batch.squeeze().tolist())   # #squeeze()函数将其从形状为(1, 3)的张量压缩为形状为(3,)的张量，tolist()函数将其转换为 Python 列表
    # # block_leaf = torch.stack(block_leaf)  # 将block中的点堆叠成一个张量
    # block_leaf = np.array(block_leaf).reshape(len(block_leaf), 4)   #将采样的锚点从列表转为数组

    '''最远点采样'''
    block_leaf = farthest_point_sampling(sam_points_l, int(0.1 * a))  # data_leaf
    block_spike = farthest_point_sampling(sam_points_s1, int(0.9 * a))

    block_data_list = []
    block_label_list = []   #定义空列表
    global raw_data_index  # 声明为全局数据（原始数据的索引）

    # 直接对锚点进行KNN（k-nearest-neighbour）上采样
    data_block = np.concatenate((block_spike, block_leaf), axis=0)  # 连接两个数组
    data_1 = np.empty((0,3))
    label_1 = np.empty((0, 1))

    for id in range(data_block.shape[0]):
        distances, indices = find_knn(data_all,[data_block[id]], kk=4096)
        block_data_sampled = data_all[indices, 0:3]
        block_label_sampled = data_all[indices, 3]
        block_data_list.append(np.expand_dims(block_data_sampled, 0))  # block_data_sampled
        block_label_list.append(np.expand_dims(block_label_sampled, 0))  # block_data_sampled
        f = open('../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_' + str(test_area) + '/' + str(room_name) + '(' + str(raw_data_index) + ').txt', "w+")
        np.savetxt(f, block_data_sampled[:, 0:3], fmt='%s', delimiter=' ')
        raw_data_index = raw_data_index + 1

    return np.concatenate(block_data_list, 0), \
           np.concatenate(block_label_list, 0)

# 按位置：将点云分成三个部分，穗、叶和叶穗交接处三个部分，对三个位置分别进行随机下采样。其中在穗存在的位置采样分为两种情况，一种是只对穗，一种是穗和叶都采样
def room2blocks_method6(data_label_filename, data, label, num_point, data_label, block_size=1.0, stride=14, random_sample=False, sample_num=None, sample_aug=1):  # block_size=1.0
    # def room2blocks(data_label_filename, data, label, num_point, block_size=1.0, stride=1.0, random_sample=True, sample_num=18, sample_aug=2):  # block_size=1.0
    """ Prepare block training data.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and aligned
            (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters 浮点数，块的物理大小（以米为单位）
        stride: float, stride for block sweeping  浮动，对小块扫描时的步长
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area][默认为房间的数量]
        sample_aug: if random sample, how much aug
    Returns:
        block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
        block_labels: K x num_point x 1 np array of uint8 labels

    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert (stride <= block_size)

    # 将取出的最大值作为限制阈值
    limit = np.amax(data, 0)[0:3]

    # Get the corner location for our sampling blocks  获取采样块的角落位置
    xbeg_list = []  # 计算(x * y)的值
    ybeg_list = []  # 计算(x * y)的值
    # 如果random_sample的值为ture，就会自动采样，sample_num确定采样点数
    # 如果random_sample的值为false，就不会自动采样.
    if not random_sample:  # abs返回绝对值
        num_block_x = int(abs(np.ceil((limit[
                                           0] - block_size) / stride))) + 1  # np.ceil：向上取整。计算x划分的块数，通过三个参数调整limit[0]（最大值的第1列，即x的值） ， block_size和stride
        num_block_y = int(abs(np.ceil(
            (limit[1] - block_size) / stride))) + 1  # 计算y划分的块数。通过三个参数调整limit[1]（最大值的第1列，即y的值） ， block_size和stride
        # 对xy的值进行循环
        for i in range(num_block_x):
            for j in range(num_block_y):
                xbeg_list.append(i * stride)  # 每次滑动stride的步长，将block的x和y储存在ybeg_list中
                ybeg_list.append(j * stride)

    # data_label_filename = data_label_filename[:-4].split('//')
    data_label_filename = data_label_filename[:-4].split('\\')
    data_label_filename = data_label_filename[len(data_label_filename) - 1]
    test_area = data_label_filename[7]  # 5
    room_name = data_label_filename[9:]  # 7:
    if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d"):
        os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d")
    if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_" + str(test_area)):
        os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_" + str(test_area))
    # Collect blocks
    # block_datas: K * num_point * 6 np array of XYZRGB, RGB is in [0,1]
    # block_labels: K * num_point * 1 np array of uint8 labels

    '''选择锚点的公式'''
    data_all = data_label[:, :4]
    a = 60000
    l = len(data[:, 0])
    '''计算每类的数量和锚点数量'''
    # 找出每个类别的数量
    spike = (data_all[:, 3] == 1).sum().item()  # 找出spike的数量
    leaf = (data_all[:, 3] == 0).sum().item()  # 找出leaf的数量
    # 找出第四列值为0和1的索引
    index_zeros = (data_all[:, 3] == 0)  # spike的标签为1
    index_ones = (data_all[:, 3] == 1)  # leaf的标签为0

    indices_s = np.where(data_all[:, 3] == 1)[0]  # 取出标签为 1（spike）的点的索引
    indices_l = np.where(data_all[:, 3] == 0)[0]  # 取出标签为 0（leaf） 的点的索引
    indices_ = np.where((data_all[:, 3] == 1) | (data_all[:, 3] == 0))[0]  # 取出标签为 0（leaf） 的点的索引

    values_s = data_all[indices_s, 2]  # 取出标签为 1（spike） 的点的第三列的值
    values_l = data_all[indices_l, 2]  # 取出标签为 0（leaf） 的点的第三列的值
    values_ = data_all[indices_, 2]  # 取出标签为 0（leaf） 的点的第三列的值

    # max_ = torch.argmax(torch.tensor(values_))  # 求出所有的最大值的索引
    max_s = torch.argmax(torch.tensor(values_s))  # 求出spike最大值的索引
    min_s = torch.argmin(torch.tensor(values_s))  # 求出spike最小值的索引
    max_l = torch.argmax(torch.tensor(values_l))  # 求出leaf最大值的索引
    min_l = torch.argmin(torch.tensor(values_l))  # 求出leaf最小值的索引

    points_ = data_all[indices_, :4]  # 取出标签为所有点的第一、二、三列作为点的坐标
    points_s = data_all[indices_s, :4]  # 取出标签为 1（spike） 的点的第一、二、三列作为点的坐标
    points_l = data_all[indices_l, :4]  # 取出标签为 0（leaf） 的点的第一、二、三列作为点的坐标

    # max_point = points_[max_]  # 取出spike在第三列上最大值的点的坐标
    max_s_point = points_s[max_s]  # 取出spike在第三列上最大值的点的坐标
    min_s_point = points_s[min_s]  # 取出spike在第三列上最大值的点的坐标
    max_l_point = points_l[max_l]  # 取出leaf在第三列上最大值的点的坐标
    min_l_point = points_l[min_l]  # 取出leaf在第三列上最小值的点的坐标

    cal = min_s_point[2] + (max_s_point[2] - min_s_point[2]) / 3
    # sam_indices_s1 = np.where((values_s >= min_s_point[2]) & (values_s <= cal))[0]  # 在spike标签中，取出在三分之一最大值和最小值之间的点的索引
    # sam_indices_s2 = np.where((values_s >= cal) & (values_s <= max_s_point[2]))[0]  #在spike标签中， 取出在三分之一最大值和最小值之间的点的索引
    sam_indices_s1 = np.where((values_ >= min_s_point[2]) & (values_ <= cal))[0]  # 在所有标签中，取出在三分之一最大值和最小值之间的点的索引
    sam_indices_s2 = np.where((values_ >= cal) & (values_ <= max_s_point[2]))[0]  # 在所有标签中，取出在三分之一最大值和最小值之间的点的索引
    sam_indices_l = np.where((values_l >= min_l_point[2]) & (values_l <= min_s_point[2]))[0]  # 取出在三分之一最大值和最小值之间的点的索引
    # sam_indices_l = np.where((values_l >= min_l_point[2]) & (values_l <= min_s_point[2]))[0]  # 取出在三分之一最大值和最小值之间的点的索引

    # sam_points_s1 = data_all[indices_s[sam_indices_s1], :4]  # 在spike标签中，取出这些点的坐标
    # sam_points_s2 = data_all[indices_s[sam_indices_s2], :4]  # 在spike标签中，取出这些点的坐标
    sam_points_s1 = data_all[indices_[sam_indices_s1], :4]  # 取出这些点的坐标
    sam_points_s2 = data_all[indices_[sam_indices_s2], :4]  # 取出这些点的坐标
    sam_points_l = data_all[indices_l[sam_indices_l], :4]  # 取出这些点的坐标
    # if zcond = ((min_spike-1 < data_all[index_ones][:, 2]) & (data_all[index_ones][:, 2] <= max_spike)):

    '''定义叶穗交接处的锚点采样'''  # 定义随机采样器，在spike类中每次采样 4096 个点。replacement=False每个点只会被采样一次
    sampler_spike_1 = RandomSampler(sam_points_s1, replacement=False, num_samples=int(0.5 * a))  # data_spike #RandomSampler(采样数据集, replacement=False采样是否放回, num_samples=采样点数)
    # 定义 DataLoader，batch_size 设置为 None，表示每个 batch 的大小等于采样器的 num_samples
    dataloader_spike_1 = DataLoader(sam_points_s1, batch_size=1, sampler=sampler_spike_1)  # None   #data_spike
    block_spike = []  # 定义空列表
    for batch in dataloader_spike_1:  # 遍历dataloader，将随机采样的点保存在block中
        # print(batch.squeeze().tolist())   #squeeze()函数将其从形状为(1, 3)的张量压缩为形状为(3,)的张量，tolist()函数将其转换为 Python 列表
        block_spike.append(batch.squeeze().tolist())  # numpy()方法将其转换为 NumPy 数组   .astype('float32')浮点数类型的矩阵
    block_spike = np.array(block_spike).reshape(len(block_spike), 4)  # 将列表转换为NumPy数组arrarr，并使用reshapereshape()方法将其形状修改为 (77, 3)
    # block_spike = torch.stack(block_spike)   # 将block中的点堆叠成一个张量

    '''定义穗叶共存处的锚点采样'''  # 定义随机采样器，在spike类中每次采样 4096 个点。replacement=False每个点只会被采样一次
    sampler_spike_2 = RandomSampler(sam_points_s2, replacement=False, num_samples=int( 0.4 * a))  # data_spike #RandomSampler(采样数据集, replacement=False采样是否放回, num_samples=采样点数)
    # 定义 DataLoader，batch_size 设置为 None，表示每个 batch 的大小等于采样器的 num_samples
    dataloader_spike_2 = DataLoader(sam_points_s2, batch_size=1, sampler=sampler_spike_2)  # None   #data_spike
    block_spike_2 = []  # 定义空列表
    for batch in dataloader_spike_2:  # 遍历dataloader，将随机采样的点保存在block中
        # print(batch.squeeze().tolist())   #squeeze()函数将其从形状为(1, 3)的张量压缩为形状为(3,)的张量，tolist()函数将其转换为 Python 列表
        block_spike_2.append(batch.squeeze().tolist())  # numpy()方法将其转换为 NumPy 数组   .astype('float32')浮点数类型的矩阵
    block_spike_2 = np.array(block_spike_2).reshape(len(block_spike_2), 4)  # 将列表转换为NumPy数组arrarr，并使用reshapereshape()方法将其形状修改为 (77, 3)

    # block_spike = torch.stack(block_spike)   # 将block中的点堆叠成一个张量

    '''定义只有叶的锚点采样'''  # 定义随机采样器，在leaf类中每次采样 4096 个点。replacement=False每个点只会被采样一次
    sampler_leaf = RandomSampler(sam_points_l, replacement=False, num_samples=int(0.1 * a))  # data_leaf
    # 定义 DataLoader，batch_size 设置为 None，表示每个 batch 的大小等于采样器的 num_samples
    dataloader_leaf = DataLoader(sam_points_l, batch_size=1, sampler=sampler_leaf)  # 加载采样的锚点  #None   #data_leaf
    block_leaf = []
    for batch in dataloader_leaf:  # 遍历dataloader，将随机采样的点保存在block中
        block_leaf.append(batch.squeeze().tolist())  # #squeeze()函数将其从形状为(1, 3)的张量压缩为形状为(3,)的张量，tolist()函数将其转换为 Python 列表
    # block_leaf = torch.stack(block_leaf)  # 将block中的点堆叠成一个张量
    block_leaf = np.array(block_leaf).reshape(len(block_leaf), 4)  # 将采样的锚点从列表转为数组

    block_data_list = []
    block_label_list = []  # 定义空列表
    global raw_data_index  # 声明为全局数据（原始数据的索引）

    data_block = np.concatenate((block_spike, block_spike_2, block_leaf), axis=0)  # 连接两个数组
    # knn_indices = []
    data_1 = np.empty((0, 3))
    label_1 = np.empty((0, 1))

    for idx in range(len(xbeg_list)):
        xbeg = xbeg_list[idx]  # 读取划分块的x轴列表
        ybeg = ybeg_list[idx]  # 读取划分块的y轴列表
        xcond_spike = (data_block[:, 0] <= xbeg + block_size) & (data_block[:, 0] >= xbeg)  # 判断该区块中的x值有多少，将整个data中的X值与block的大小相匹配
        ycond_spike = (data_block[:, 1] <= ybeg + block_size) & (data_block[:, 1] >= ybeg)  # 判断该区块中的x值有多少，将整个data中的y值与block的大小相匹配
        cond = xcond_spike & ycond_spike  # 求xcond与ycond的交集，查看是否在包含锚点

        ###原始的数据集加载
        if (np.sum(cond) <= 0):  # 求两个集合的并集 抛弃不包含锚点的块
            print("spike or leaf anchor are not in this block")
            continue
        # cond = cond_spike & cond_leaf
        # cond = np.concatenate([cond_spike,cond_leaf])   #连接两个数组
        # data_block = np.concatenate((block_spike,block_leaf),axis=0)   #连接两个数组
        '''Over'''

        block_data = data_block[cond, :3]  ##丢弃了小于100个点的块之后剩下的，在wheat数据集中每个块均大于100个点，所以没有丢弃的块，在S3DIS中由于点云的缺失，有丢弃的块，此处的阈值可以调高
        block_label = data_block[cond, 3]

        # randomly subsample data   随机重采样数据
        block_data_sampled, block_label_sampled = \
            sample_data_label(block_data, block_label, num_point)  # num_point
        block_data_list.append(np.expand_dims(block_data_sampled, 0))  # block_data_sampled
        block_label_list.append(np.expand_dims(block_label_sampled, 0))  # block_data_sampled
        f = open('../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_' + str(test_area) + '/' + str(room_name) + '(' + str(raw_data_index) + ').txt', "w+")
        np.savetxt(f, block_data_sampled[:, 0:3], fmt='%s', delimiter=' ')
        raw_data_index = raw_data_index + 1

    # # 直接对锚点进行KNN（k-nearest-neighbour）上采样
    # data_block = np.concatenate((block_spike, block_spike_2, block_leaf), axis=0)  # 连接两个数组
    # # knn_indices = []
    # data_1 = np.empty((0,3))
    # label_1 = np.empty((0, 1))
    #
    # for id in range(data_block.shape[0]):
    #     # print(data_block.shape[0])
    #     # print(data_block[id])
    #     # print([data_block[id]])
    #     distances, indices = find_knn(data_all, [data_block[id]], kk=4096)
    #     # print(indices[0])
    #     indices_1 = indices[0]
    #     block_data_sampled = data_all[indices_1, 0:3]
    #     block_label_sampled = data_all[indices_1, 3]
    #     block_data_list.append(np.expand_dims(block_data_sampled, 0))  # block_data_sampled
    #     block_label_list.append(np.expand_dims(block_label_sampled, 0))  # block_data_sampled
    #     f = open('../data/wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_' + str(test_area) + '/' + str(room_name) + '(' + str(raw_data_index) + ').txt', "w+")
    #     np.savetxt(f, block_data_sampled[:, 0:3], fmt='%s', delimiter=' ')
    #     raw_data_index = raw_data_index + 1

    return np.concatenate(block_data_list, 0), \
           np.concatenate(block_label_list, 0)

# 真实小麦数据的采样，按位置叶的二分之一以上，和叶的二分一以上
def room2blocks_method7(data_label_filename, data, label, num_point, data_label, block_size=1.0, stride=14, random_sample=False, sample_num=None, sample_aug=1):  # block_size=1.0
    # def room2blocks(data_label_filename, data, label, num_point, block_size=1.0, stride=1.0, random_sample=True, sample_num=18, sample_aug=2):  # block_size=1.0
    """ Prepare block training data.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and aligned
            (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters 浮点数，块的物理大小（以米为单位）
        stride: float, stride for block sweeping  浮动，对小块扫描时的步长
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area][默认为房间的数量]
        sample_aug: if random sample, how much aug
    Returns:
        block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
        block_labels: K x num_point x 1 np array of uint8 labels

    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    assert (stride <= block_size)

    # 将取出x轴的最大值作为限制阈值
    limit = np.amax(data, 0)[0:3]

    # Get the corner location for our sampling blocks  获取采样块的角落位置
    xbeg_list = []  # 计算(x * y)的值
    ybeg_list = []  # 计算(x * y)的值
    # 如果random_sample的值为ture，就会自动采样，sample_num确定采样点数
    # 如果random_sample的值为false，就不会自动采样.
    if not random_sample:  # abs返回绝对值
        num_block_x = int(abs(np.ceil((limit[0] - block_size) / stride))) + 1  # np.ceil：向上取整。计算x划分的块数，通过三个参数调整limit[0]（最大值的第1列，即x的值） ， block_size和stride
        num_block_y = int(abs(np.ceil((limit[1] - block_size) / stride))) + 1  # 计算y划分的块数。通过三个参数调整limit[1]（最大值的第1列，即y的值） ， block_size和stride
        # 对xy的值进行循环
        for i in range(num_block_x):
            for j in range(num_block_y):
                xbeg_list.append(i * stride)  # 每次滑动stride的步长，将block的x和y储存在ybeg_list中
                ybeg_list.append(j * stride)

    # data_label_filename = data_label_filename[:-4].split('//')
    data_label_filename = data_label_filename[:-4].split('\\')
    data_label_filename = data_label_filename[len(data_label_filename) - 1]
    test_area = data_label_filename[7]  # 5
    room_name = data_label_filename[9:]  # 7:
    if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d"):
        os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d")
    if not os.path.exists("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_" + str(test_area)):
        os.makedirs("../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_" + str(test_area))
    # Collect blocks
    # block_datas: K * num_point * 6 np array of XYZRGB, RGB is in [0,1]
    # block_labels: K * num_point * 1 np array of uint8 labels

    '''选择锚点的公式'''
    data_all = data_label[:, :4]
    a = 6000
    l = len(data[:, 0])
    '''计算每类的数量和锚点数量'''
    # 找出每个类别的数量
    spike = (data_all[:, 3] == 1).sum().item()  # 找出spike的数量  spike = 155150
    leaf = (data_all[:, 3] == 0).sum().item()  # 找出leaf的数量    leaf = 630683
    # 找出第四列值为0和1的索引
    index_zeros = (data_all[:, 3] == 1)  # spike的标签为1   [ture,false,...]
    index_ones = (data_all[:, 3] == 0)  # leaf的标签为0     [ture,false,...]

    indices_s = np.where(data_all[:, 3] == 1)[0]  # 取出标签为 1（spike）的点的索引                   [0,1,2,5,...]
    indices_l = np.where(data_all[:, 3] == 0)[0]  # 取出标签为 0（leaf） 的点的索引                   [6,7,8,9,...]
    indices_ = np.where((data_all[:, 3] == 1) | (data_all[:, 3] == 0))[0]  # 取出所有标签点的索引    [0,1,2,3,...]


    values_s = data_all[indices_s, 2]  # 取出标签为 1（spike） 的点的第三列的值
    values_l = data_all[indices_l, 2]  # 取出标签为 0（leaf） 的点的第三列的值
    values_ = data_all[indices_, 2]    # 取出所有标签点的第三列的值

    # max_ = torch.argmax(torch.tensor(values_))  # 求出所有的最大值的索引
    max_s = torch.argmax(torch.tensor(values_s))  # 求出spike在z轴上最大值的索引
    min_s = torch.argmin(torch.tensor(values_s))  # 求出spike在z轴上最小值的索引
    max_l = torch.argmax(torch.tensor(values_l))  # 求出leaf在z轴上最大值的索引
    min_l = torch.argmin(torch.tensor(values_l))  # 求出leaf在z轴上最小值的索引

    points_ = data_all[indices_, :4]  # 取出标签为所有点的第一、二、三、四列作为点的坐标
    points_s = data_all[indices_s, :4]  # 取出标签为 1（spike） 的点的第一、二、三、四列作为点的坐标
    points_l = data_all[indices_l, :4]  # 取出标签为 0（leaf） 的点的第一、二、三、四列作为点的坐标

    # max_point = points_[max_]  # 取出spike在第三列上最大值的点的坐标
    max_s_point = points_s[max_s]  # 取出spike在第三列上最大值的点的坐标
    min_s_point = points_s[min_s]  # 取出spike在第三列上最小值的点的坐标
    max_l_point = points_l[max_l]  # 取出leaf在第三列上最大值的点的坐标
    min_l_point = points_l[min_l]  # 取出leaf在第三列上最小值的点的坐标

    '''将整个点云划分为2个部分'''
    cal = max_l_point[2] / 2
    if min_s_point[2] <= cal:
        cal = min_s_point[2] - min_s_point[2] / 2
    # sam_indices_s1 = np.where((values_s >= min_s_point[2]) & (values_s <= cal))[0]  # 在spike标签中，取出在三分之一最大值和最小值之间的点的索引
    # sam_indices_s2 = np.where((values_s >= cal) & (values_s <= max_s_point[2]))[0]  #在spike标签中， 取出在三分之一最大值和最小值之间的点的索引
    sam_indices_s1 = np.where((values_ <= cal))[0]  # 在所有标签中，取出在二分之一以下之间的点的索引
    sam_indices_s2 = np.where((values_ >= cal))[0]  # 在所有标签中，取出在二分之一以上最大值和最小值之间的点的索引

    # sam_points_s1 = data_all[indices_s[sam_indices_s1], :4]  # 在spike标签中，取出这些点的坐标
    # sam_points_s2 = data_all[indices_s[sam_indices_s2], :4]  # 在spike标签中，取出这些点的坐标
    sam_points_s1 = data_all[indices_[sam_indices_s1], :4]  # 取出这些点的坐标
    sam_points_s2 = data_all[indices_[sam_indices_s2], :4]  # 取出这些点的坐标
    # if zcond = ((min_spike-1 < data_all[index_ones][:, 2]) & (data_all[index_ones][:, 2] <= max_spike)):

    '''定义二分之一以下的锚点采样'''  # 定义随机采样器，在spike类中每次采样 4096 个点。replacement=False每个点只会被采样一次
    sampler_spike_1 = RandomSampler(sam_points_s1, replacement=False, num_samples=int(0.7 * a))  # data_spike #RandomSampler(采样数据集, replacement=False采样是否放回, num_samples=采样点数)
    # 定义 DataLoader，batch_size 设置为 None，表示每个 batch 的大小等于采样器的 num_samples
    dataloader_spike_1 = DataLoader(sam_points_s1, batch_size=1, sampler=sampler_spike_1)  # None   #data_spike
    block_spike = []  # 定义空列表
    for batch in dataloader_spike_1:  # 遍历dataloader，将随机采样的点保存在block中
        # print(batch.squeeze().tolist())   #squeeze()函数将其从形状为(1, 3)的张量压缩为形状为(3,)的张量，tolist()函数将其转换为 Python 列表
        block_spike.append(batch.squeeze().tolist())  # numpy()方法将其转换为 NumPy 数组   .astype('float32')浮点数类型的矩阵
    block_spike = np.array(block_spike).reshape(len(block_spike), 4)  # 将列表转换为NumPy数组arrarr，并使用reshapereshape()方法将其形状修改为 (77, 3)

    '''定义二分之一以上的锚点采样'''  # 定义随机采样器，在spike类中每次采样 4096 个点。replacement=False每个点只会被采样一次
    # sampler_spike_2 = RandomSampler(sam_points_s2, replacement=False, num_samples=int( 0.3 * a))  # data_spike #RandomSampler(采样数据集, replacement=False采样是否放回, num_samples=采样点数)
    # # 定义 DataLoader，batch_size 设置为 None，表示每个 batch 的大小等于采样器的 num_samples
    # dataloader_spike_2 = DataLoader(sam_points_s2, batch_size=1, sampler=sampler_spike_2)  # None   #data_spike
    # block_spike_2 = []  # 定义空列表
    # for batch in dataloader_spike_2:  # 遍历dataloader，将随机采样的点保存在block中
    #     # print(batch.squeeze().tolist())   #squeeze()函数将其从形状为(1, 3)的张量压缩为形状为(3,)的张量，tolist()函数将其转换为 Python 列表
    #     block_spike_2.append(batch.squeeze().tolist())  # numpy()方法将其转换为 NumPy 数组   .astype('float32')浮点数类型的矩阵
    # block_spike_2 = np.array(block_spike_2).reshape(len(block_spike_2), 4)  # 将列表转换为NumPy数组arrarr，并使用reshapereshape()方法将其形状修改为 (77, 3)

    block_data_list = []
    block_label_list = []  # 定义空列表
    global raw_data_index  # 声明为全局数据（原始数据的索引）

    data_block = np.concatenate((block_spike, sam_points_s2), axis=0)  # 连接两个数组
    for idx in range(len(xbeg_list)):
        xbeg = xbeg_list[idx]  # 读取划分块的x轴列表
        ybeg = ybeg_list[idx]  # 读取划分块的y轴列表
        xcond_spike = (data_block[:, 0] <= xbeg + block_size) & (data_block[:, 0] >= xbeg)  # 判断该区块中的x值有多少，将整个data中的X值与block的大小相匹配
        ycond_spike = (data_block[:, 1] <= ybeg + block_size) & (data_block[:, 1] >= ybeg)  # 判断该区块中的x值有多少，将整个data中的y值与block的大小相匹配
        cond = xcond_spike & ycond_spike  # 求xcond与ycond的交集，查看是否在包含锚点

        ###原始的数据集加载
        if (np.sum(cond) <= 0):  # 求两个集合的并集 抛弃不包含锚点的块
            print("spike or leaf anchor are not in this block")
            continue

        block_data = data_block[cond, :3]  ##丢弃了小于100个点的块之后剩下的，在wheat数据集中每个块均大于100个点，所以没有丢弃的块，在S3DIS中由于点云的缺失，有丢弃的块，此处的阈值可以调高
        block_label = data_block[cond, 3]

        # randomly subsample data   随机重采样数据
        block_data_sampled, block_label_sampled = \
            sample_data_label(block_data, block_label, num_point)  # num_point
        block_data_list.append(np.expand_dims(block_data_sampled, 0))  # block_data_sampled
        block_label_list.append(np.expand_dims(block_label_sampled, 0))  # block_data_sampled
        f = open('../data/true_wheat_sem_seg_hdf5_data_test/raw_data3d/Sample_' + str(test_area) + '/' + str(room_name) + '(' + str(raw_data_index) + ').txt', "w+")
        np.savetxt(f, block_data_sampled[:, 0:3], fmt='%s', delimiter=' ')
        raw_data_index = raw_data_index + 1

    return np.concatenate(block_data_list, 0), \
           np.concatenate(block_label_list, 0)


def room2blocks_plus(data_label, num_point, block_size, stride, random_sample, sample_num, sample_aug):
    """ room2block with input filename and RGB preprocessing.
        数据读取，获得前3列数据，并将RGB通道归一化
    """
    #data = data_label[:,0:6]
    data = data_label[:,0:3]
    # data[:,3:6] /= 255.0
    label = data_label[:,-1].astype(np.uint8)
    
    return room2blocks(data, label, num_point, block_size, stride, random_sample, sample_num, sample_aug)
   
def room2blocks_wrapper(data_label_filename, num_point, block_size=1.0, stride=1.0, random_sample=False, sample_num=None, sample_aug=1):
    if data_label_filename[-3:] == 'txt':
        data_label = np.loadtxt(data_label_filename)
    elif data_label_filename[-3:] == 'npy':
        data_label = np.load(data_label_filename)
    else:
        print('Unknown file type! exiting.')
        exit()
    return room2blocks_plus(data_label, num_point, block_size, stride, random_sample, sample_num, sample_aug)

#对room2blocks_wrapper_normalized加载的数据进行归一化，
def room2blocks_plus_normalized(data_label_filename, data_label, num_point, block_size, stride, random_sample, sample_num, sample_aug):
    """ room2block, with input filename and RGB preprocessing.
        for each block centralize XYZ, add normalized XYZ as 678 channels
    """
    """ 
    S3DIS带rgb通道，加载的数据就是第1-6列的数据就是xyzrgb，后面的7-8列再加归一化后的数据；
    wheat数据就是不带RGB通道，加载数据的第1-3列是xyz，后面的第3-6列是归一化后的数据
    """
    data = data_label[:,0:6]
    # data = data_label[:, 0:3]  #读取数据所有行的1-3列
    # data[:,3:6] /= 255.0   归一化RGB通道
    label = data_label[:,-1].astype(np.uint8)  #将标签加在最后一列
    max_room_x = max(data[:,0])   #分别取xyz的最大值
    max_room_y = max(data[:,1])
    max_room_z = max(data[:,2])
    # data_batch, label_batch = room2blocks(data_label_filename, data, label, num_point, block_size, stride,random_sample, sample_num, sample_aug)  #原始
    data_batch, label_batch = room2blocks_1(data_label_filename, data, label, num_point, data_label, block_size, stride, random_sample, sample_num, sample_aug)   #加了采样
    new_data_batch = np.zeros((data_batch.shape[0], num_point, 9))   #num_point, 9
    #原始
    for b in range(data_batch.shape[0]):
        new_data_batch[b, :, 6] = data_batch[b, :, 0]/max_room_x
        new_data_batch[b, :, 7] = data_batch[b, :, 1]/max_room_y
        new_data_batch[b, :, 8] = data_batch[b, :, 2]/max_room_z
        minx = min(data_batch[b, :, 0])
        miny = min(data_batch[b, :, 1])
        data_batch[b, :, 0] -= (minx+block_size/2)
        data_batch[b, :, 1] -= (miny+block_size/2)
    new_data_batch[:, :, 0:6] = data_batch

    # for b in range(data_batch.shape[0]):
    #     new_data_batch[b, :, 3] = data_batch[b, :, 0]/max_room_x
    #     new_data_batch[b, :, 4] = data_batch[b, :, 1]/max_room_y
    #     new_data_batch[b, :, 5] = data_batch[b, :, 2]/max_room_z
    #     minx = min(data_batch[b, :, 0])
    #     miny = min(data_batch[b, :, 1])
    #     data_batch[b, :, 0] -= (minx+block_size/2)
    #     data_batch[b, :, 1] -= (miny+block_size/2)
    # new_data_batch[:, :, 0:3] = data_batch
    return new_data_batch, label_batch

#取出.npy文件的数据和标签，并返回room2blocks_plus_normalized进行归一化
def room2blocks_wrapper_normalized(data_label_filename, num_point, block_size=1.0, stride=1.0, random_sample=False, sample_num=None, sample_aug=1):
    #加载.npy文件中的数据和标签，获取从索引-3开始的之后的所有数据
    # wheat数据集加载的是xyz和label
    # S3DIS加载的是xyzrgb和label
    if data_label_filename[-3:] == 'txt':
        data_label = np.loadtxt(data_label_filename)
    elif data_label_filename[-3:] == 'npy':
        data_label = np.load(data_label_filename)
    else:
        print('Unknown file type! exiting.')
        exit()
    return room2blocks_plus_normalized(data_label_filename, data_label, num_point, block_size, stride, random_sample, sample_num, sample_aug)

def room2samples(data, label, sample_num_point):
    """ Prepare whole room samples.

    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and
            aligned (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        sample_num_point: int, how many points to sample in each sample
    Returns:
        sample_datas: K x sample_num_point x 9
                     numpy array of XYZRGBX'Y'Z', RGB is in [0,1]
        sample_labels: K x sample_num_point x 1 np array of uint8 labels
    """
    N = data.shape[0]   #data=[xyz]
    order = np.arange(N)
    np.random.shuffle(order) 
    data = data[order, :]
    label = label[order]

    batch_num = int(np.ceil(N / float(sample_num_point)))
    sample_datas = np.zeros((batch_num, sample_num_point, 6))  #sample_datas:6[x,Y,Z,Nx,Ny,Nz]
    sample_labels = np.zeros((batch_num, sample_num_point, 1))

    for i in range(batch_num):
        beg_idx = i*sample_num_point
        end_idx = min((i+1)*sample_num_point, N)
        num = end_idx - beg_idx
        sample_datas[i,0:num,:] = data[beg_idx:end_idx, :]
        sample_labels[i,0:num,0] = label[beg_idx:end_idx]
        if num < sample_num_point:   #如果当前的点小于采样的num_points(=4096)
            makeup_indices = np.random.choice(N, sample_num_point - num)  #就随机采样 num_points(=4096) - num（当前点）
            sample_datas[i,num:,:] = data[makeup_indices, :]
            sample_labels[i,num:,0] = label[makeup_indices]
    return sample_datas, sample_labels

def room2samples_plus_normalized(data_label, num_point):
    """ room2sample, with input filename and RGB preprocessing.
        for each block centralize XYZ, add normalized XYZ as 678 channels
    """
    data = data_label[:,0:6]
    # data[:,3:6] /= 255.0
    # data = data_label[:, 0:3]
    label = data_label[:,-1].astype(np.uint8)
    max_room_x = max(data[:,0])
    max_room_y = max(data[:,1])
    max_room_z = max(data[:,2])
    
    data_batch, label_batch = room2samples(data, label, num_point)
    new_data_batch = np.zeros((data_batch.shape[0], num_point, 9))   #num_point, 9
    for b in range(data_batch.shape[0]):
        # 原始
        new_data_batch[b, :, 6] = data_batch[b, :, 0] / max_room_x
        new_data_batch[b, :, 7] = data_batch[b, :, 1] / max_room_y
        new_data_batch[b, :, 8] = data_batch[b, :, 2] / max_room_z
        # new_data_batch[b, :, 3] = data_batch[b, :, 0]/max_room_x
        # new_data_batch[b, :, 4] = data_batch[b, :, 1]/max_room_y
        # new_data_batch[b, :, 5] = data_batch[b, :, 2]/max_room_z
        #minx = min(data_batch[b, :, 0])
        #miny = min(data_batch[b, :, 1])
        #data_batch[b, :, 0] -= (minx+block_size/2)
        #data_batch[b, :, 1] -= (miny+block_size/2)
    # new_data_batch[:, :, 0:3] = data_batch
    new_data_batch[:, :, 0:6] = data_batch
    return new_data_batch, label_batch

def room2samples_wrapper_normalized(data_label_filename, num_point):
    if data_label_filename[-3:] == 'txt':
        data_label = np.loadtxt(data_label_filename)
    elif data_label_filename[-3:] == 'npy':
        data_label = np.load(data_label_filename)
    else:
        print('Unknown file type! exiting.')
        exit()
    return room2samples_plus_normalized(data_label, num_point)


# -----------------------------------------------------------------------------
# EXTRACT INSTANCE BBOX FROM ORIGINAL DATA (for detection evaluation)
# -----------------------------------------------------------------------------

def collect_bounding_box(anno_path, out_filename):
    """ Compute bounding boxes from each instance in original dataset files on
        one room. **We assume the bbox is aligned with XYZ coordinate.**
    
    Args:
        anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
        out_filename: path to save instance bounding boxes for that room.
            each line is x1 y1 z1 x2 y2 z2 label,
            where (x1,y1,z1) is the point on the diagonal closer to origin
    Returns:
        None
    Note:
        room points are shifted, the most negative point is now at origin.
    """
    bbox_label_list = []

    for f in glob.glob(os.path.join(anno_path, '*.txt')):
        cls = os.path.basename(f).split('_')[0]
        if cls not in g_classes: # note: in some room there is 'staris' class..
            cls = 'clutter'
        points = np.loadtxt(f)
        label = g_class2label[cls]
        # Compute tightest axis aligned bounding box
        xyz_min = np.amin(points[:, 0:3], axis=0)
        xyz_max = np.amax(points[:, 0:3], axis=0)
        ins_bbox_label = np.expand_dims(
            np.concatenate([xyz_min, xyz_max, np.array([label])], 0), 0)
        bbox_label_list.append(ins_bbox_label)

    bbox_label = np.concatenate(bbox_label_list, 0)
    room_xyz_min = np.amin(bbox_label[:, 0:3], axis=0)
    bbox_label[:, 0:3] -= room_xyz_min 
    bbox_label[:, 3:6] -= room_xyz_min 

    fout = open(out_filename, 'w')
    for i in range(bbox_label.shape[0]):
        fout.write('%f %f %f %f %f %f %d\n' % \
                      (bbox_label[i,0], bbox_label[i,1], bbox_label[i,2],
                       bbox_label[i,3], bbox_label[i,4], bbox_label[i,5],
                       bbox_label[i,6]))
    fout.close()

def bbox_label_to_obj(input_filename, out_filename_prefix, easy_view=False):
    """ Visualization of bounding boxes.
    
    Args:
        input_filename: each line is x1 y1 z1 x2 y2 z2 label
        out_filename_prefix: OBJ filename prefix,
            visualize object by g_label2color
        easy_view: if True, only visualize furniture and floor
    Returns:
        output a list of OBJ file and MTL files with the same prefix
    """
    bbox_label = np.loadtxt(input_filename)
    bbox = bbox_label[:, 0:6]
    label = bbox_label[:, -1].astype(int)
    v_cnt = 0 # count vertex
    ins_cnt = 0 # count instance
    for i in range(bbox.shape[0]):
        if easy_view and (label[i] not in g_easy_view_labels):
            continue
        obj_filename = out_filename_prefix+'_'+g_classes[label[i]]+'_'+str(ins_cnt)+'.obj'
        mtl_filename = out_filename_prefix+'_'+g_classes[label[i]]+'_'+str(ins_cnt)+'.mtl'
        fout_obj = open(obj_filename, 'w')
        fout_mtl = open(mtl_filename, 'w')
        fout_obj.write('mtllib %s\n' % (os.path.basename(mtl_filename)))

        length = bbox[i, 3:6] - bbox[i, 0:3]
        a = length[0]
        b = length[1]
        c = length[2]
        x = bbox[i, 0]
        y = bbox[i, 1]
        z = bbox[i, 2]
        color = np.array(g_label2color[label[i]], dtype=float) / 255.0

        material = 'material%d' % (ins_cnt)
        fout_obj.write('usemtl %s\n' % (material))
        fout_obj.write('v %f %f %f\n' % (x,y,z+c))
        fout_obj.write('v %f %f %f\n' % (x,y+b,z+c))
        fout_obj.write('v %f %f %f\n' % (x+a,y+b,z+c))
        fout_obj.write('v %f %f %f\n' % (x+a,y,z+c))
        fout_obj.write('v %f %f %f\n' % (x,y,z))
        fout_obj.write('v %f %f %f\n' % (x,y+b,z))
        fout_obj.write('v %f %f %f\n' % (x+a,y+b,z))
        fout_obj.write('v %f %f %f\n' % (x+a,y,z))
        fout_obj.write('g default\n')
        v_cnt = 0 # for individual box
        fout_obj.write('f %d %d %d %d\n' % (4+v_cnt, 3+v_cnt, 2+v_cnt, 1+v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (1+v_cnt, 2+v_cnt, 6+v_cnt, 5+v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (7+v_cnt, 6+v_cnt, 2+v_cnt, 3+v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (4+v_cnt, 8+v_cnt, 7+v_cnt, 3+v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (5+v_cnt, 8+v_cnt, 4+v_cnt, 1+v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (5+v_cnt, 6+v_cnt, 7+v_cnt, 8+v_cnt))
        fout_obj.write('\n')

        fout_mtl.write('newmtl %s\n' % (material))
        fout_mtl.write('Kd %f %f %f\n' % (color[0], color[1], color[2]))
        fout_mtl.write('\n')
        fout_obj.close()
        fout_mtl.close() 

        v_cnt += 8
        ins_cnt += 1

def bbox_label_to_obj_room(input_filename, out_filename_prefix, easy_view=False, permute=None, center=False, exclude_table=False):
    """ Visualization of bounding boxes.
    
    Args:
        input_filename: each line is x1 y1 z1 x2 y2 z2 label
        out_filename_prefix: OBJ filename prefix,
            visualize object by g_label2color
        easy_view: if True, only visualize furniture and floor
        permute: if not None, permute XYZ for rendering, e.g. [0 2 1]
        center: if True, move obj to have zero origin
    Returns:
        output a list of OBJ file and MTL files with the same prefix
    """
    bbox_label = np.loadtxt(input_filename)
    bbox = bbox_label[:, 0:6]
    if permute is not None:
        assert(len(permute)==3)
        permute = np.array(permute)
        bbox[:,0:3] = bbox[:,permute]
        bbox[:,3:6] = bbox[:,permute+3]
    if center:
        xyz_max = np.amax(bbox[:,3:6], 0)
        bbox[:,0:3] -= (xyz_max/2.0)
        bbox[:,3:6] -= (xyz_max/2.0)
        bbox /= np.max(xyz_max/2.0)
    label = bbox_label[:, -1].astype(int)
    obj_filename = out_filename_prefix+'.obj' 
    mtl_filename = out_filename_prefix+'.mtl'

    fout_obj = open(obj_filename, 'w')
    fout_mtl = open(mtl_filename, 'w')
    fout_obj.write('mtllib %s\n' % (os.path.basename(mtl_filename)))
    v_cnt = 0 # count vertex
    ins_cnt = 0 # count instance
    for i in range(bbox.shape[0]):
        if easy_view and (label[i] not in g_easy_view_labels):
            continue
        if exclude_table and label[i] == g_classes.index('table'):
            continue

        length = bbox[i, 3:6] - bbox[i, 0:3]
        a = length[0]
        b = length[1]
        c = length[2]
        x = bbox[i, 0]
        y = bbox[i, 1]
        z = bbox[i, 2]
        color = np.array(g_label2color[label[i]], dtype=float) / 255.0

        material = 'material%d' % (ins_cnt)
        fout_obj.write('usemtl %s\n' % (material))
        fout_obj.write('v %f %f %f\n' % (x,y,z+c))
        fout_obj.write('v %f %f %f\n' % (x,y+b,z+c))
        fout_obj.write('v %f %f %f\n' % (x+a,y+b,z+c))
        fout_obj.write('v %f %f %f\n' % (x+a,y,z+c))
        fout_obj.write('v %f %f %f\n' % (x,y,z))
        fout_obj.write('v %f %f %f\n' % (x,y+b,z))
        fout_obj.write('v %f %f %f\n' % (x+a,y+b,z))
        fout_obj.write('v %f %f %f\n' % (x+a,y,z))
        fout_obj.write('g default\n')
        fout_obj.write('f %d %d %d %d\n' % (4+v_cnt, 3+v_cnt, 2+v_cnt, 1+v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (1+v_cnt, 2+v_cnt, 6+v_cnt, 5+v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (7+v_cnt, 6+v_cnt, 2+v_cnt, 3+v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (4+v_cnt, 8+v_cnt, 7+v_cnt, 3+v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (5+v_cnt, 8+v_cnt, 4+v_cnt, 1+v_cnt))
        fout_obj.write('f %d %d %d %d\n' % (5+v_cnt, 6+v_cnt, 7+v_cnt, 8+v_cnt))
        fout_obj.write('\n')

        fout_mtl.write('newmtl %s\n' % (material))
        fout_mtl.write('Kd %f %f %f\n' % (color[0], color[1], color[2]))
        fout_mtl.write('\n')

        v_cnt += 8
        ins_cnt += 1

    fout_obj.close()
    fout_mtl.close() 

def collect_point_bounding_box(anno_path, out_filename, file_format):
    """ Compute bounding boxes from each instance in original dataset files on
        one room. **We assume the bbox is aligned with XYZ coordinate.**
        Save both the point XYZRGB and the bounding box for the point's
        parent element.
 
    Args:
        anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
        out_filename: path to save instance bounding boxes for each point,
            plus the point's XYZRGBL
            each line is XYZRGBL offsetX offsetY offsetZ a b c,
            where cx = X+offsetX, cy=X+offsetY, cz=Z+offsetZ
            where (cx,cy,cz) is center of the box, a,b,c are distances from center
            to the surfaces of the box, i.e. x1 = cx-a, x2 = cx+a, y1=cy-b etc.
        file_format: output file format, txt or numpy
    Returns:
        None

    Note:
        room points are shifted, the most negative point is now at origin.
    """
    point_bbox_list = []

    for f in glob.glob(os.path.join(anno_path, '*.txt')):
        cls = os.path.basename(f).split('_')[0]
        if cls not in g_classes: # note: in some room there is 'staris' class..
            cls = 'clutter'
        points = np.loadtxt(f) # Nx6
        label = g_class2label[cls] # N,
        # Compute tightest axis aligned bounding box
        xyz_min = np.amin(points[:, 0:3], axis=0) # 3,
        xyz_max = np.amax(points[:, 0:3], axis=0) # 3,
        xyz_center = (xyz_min + xyz_max) / 2
        dimension = (xyz_max - xyz_min) / 2

        xyz_offsets = xyz_center - points[:,0:3] # Nx3
        dimensions = np.ones((points.shape[0],3)) * dimension # Nx3
        labels = np.ones((points.shape[0],1)) * label # N
        point_bbox_list.append(np.concatenate([points, labels,
                                           xyz_offsets, dimensions], 1)) # Nx13

    point_bbox = np.concatenate(point_bbox_list, 0) # KxNx13
    room_xyz_min = np.amin(point_bbox[:, 0:3], axis=0)
    point_bbox[:, 0:3] -= room_xyz_min 

    if file_format == 'txt':
        fout = open(out_filename, 'w')
        for i in range(point_bbox.shape[0]):
            fout.write('%f %f %f %d %d %d %d %f %f %f %f %f %f\n' % \
                          (point_bbox[i,0], point_bbox[i,1], point_bbox[i,2],
                           point_bbox[i,3], point_bbox[i,4], point_bbox[i,5],
                           point_bbox[i,6],
                           point_bbox[i,7], point_bbox[i,8], point_bbox[i,9],
                           point_bbox[i,10], point_bbox[i,11], point_bbox[i,12]))
        
        fout.close()
    elif file_format == 'numpy':
        np.save(out_filename, point_bbox)
    else:
        print('ERROR!! Unknown file format: %s, please use txt or numpy.' % \
            (file_format))
        exit()


