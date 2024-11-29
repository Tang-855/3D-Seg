import os
import numpy as np
import sys
# noinspection PyUnresolvedReferences
import json
# noinspection PyUnresolvedReferences
# from prepare_data import wheat_data_prep_util
from wheat_data_prep_util import *
# noinspection PyUnresolvedReferences
# from .prepare_data import true_wheat_indoor3d_util_normal
from true_wheat_indoor3d_util_normal import *
# noinspection PyUnresolvedReferences
import open3d as o3d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

# Constants
data_dir = os.path.join(ROOT_DIR, 'data')
# indoor3d_data_dir = os.path.join(data_dir, 'true_wheat_npy')
indoor3d_data_dir = "D:/a-project-T/Data/rice_data/rice_npy"
# indoor3d_data_dir = "D:/a-project-T/Data/true_wheat_data/true_wheat_npy"

NUM_POINT = 4096   #把每个点云小块有4096个点，并储存在./Wheat_hdf5_test/raw_data3d中
H5_BATCH_SIZE = 1000   #总共有75678个小块，每个1000个储存.h5文件，共储存了75个.h5文件
data_dim = [NUM_POINT, 9]   #每个数据维度[每一个块的点数4096，特征维度]，9：xyzrgb和归一化的xyz，6：xyz和归一化的xyz
label_dim = [NUM_POINT]   #数据标签，给每一个点分配标签
data_dtype = 'float32'   #定义数据类型
label_dtype = 'uint8'    #定义标签的类型

# Set paths
filelist = os.path.join(BASE_DIR, 'meta/wheat_all_data_label.txt')  #储存所有房间名npy格式的文件
data_label_files = [os.path.join(indoor3d_data_dir, line.rstrip()) for line in open(filelist)]  #遍历.npy文件的所有的文件并储存出来

# output_dir = os.path.join(data_dir, 'true_wheat_sem_seg_hdf5_data_test')   #输出文件夹
output_dir = "C:/Users/tang/Desktop/true_wheat_sem_seg_hdf5_data_test"   #输出文件夹

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    # os.mkdir(output_dir)   #如果文件夹存在，利用这个创建文件夹就会报错
output_filename_prefix = os.path.join(output_dir, 'ply_data_all')   #每个1000个NUM_POINT储存一个.h5文件
output_room_filelist = os.path.join(output_dir, 'room_filelist.txt')    #储存每一个NUM_POINT块的文件名
output_all_file = os.path.join(output_dir, 'all_files.txt')   #储存生成的.h5文件的所有路径
fout_room = open(output_room_filelist, 'w')
all_file = open(output_all_file, 'w')

# --------------------------------------
# ----- BATCH WRITE TO HDF5 -----
# --------------------------------------
batch_data_dim = [H5_BATCH_SIZE] + data_dim   #[1000：h5文件的间隔数，4096：每一块的点数，6：数据维度（xyz和归一化的xyz）]
batch_label_dim = [H5_BATCH_SIZE] + label_dim  #[1000：h5文件的间隔数，4096：每一块的点的标签]
h5_batch_data = np.zeros(batch_data_dim, dtype = np.float32)  #数组（浮点数）[1000：h5文件的间隔数，4096：每一块的点数，6：数据维度（xyz和归一化的xyz）]
h5_batch_label = np.zeros(batch_label_dim, dtype = np.uint8)  #数组（整数）[1000：h5文件的间隔数，4096：每一块的点的标签]
buffer_size = 0  # state: record how many samples are currently in buffer   #记录当前的缓冲区有多少样本量
h5_index = 0 # state: the next h5 file to save   #记录保存的是第几个h5文件

def insert_batch(data, label, last_batch=False):
    global h5_batch_data, h5_batch_label   #声明为全局变量
    global buffer_size, h5_index     #声名为全局变量
    data_size = data.shape[0]
    # If there is enough space, just insert   如果空间数量足够，就不插入
    if buffer_size + data_size <= h5_batch_data.shape[0]:
        h5_batch_data[buffer_size:buffer_size+data_size, ...] = data
        h5_batch_label[buffer_size:buffer_size+data_size] = label
        buffer_size += data_size
    else: # not enough space   如果不够就插入
        capacity = h5_batch_data.shape[0] - buffer_size
        assert(capacity>=0)
        if capacity > 0:
           h5_batch_data[buffer_size:buffer_size+capacity, ...] = data[0:capacity, ...] 
           h5_batch_label[buffer_size:buffer_size+capacity, ...] = label[0:capacity, ...]
        # Save batch data and label to h5 file, reset buffer_size
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        # wheat_data_prep_util.save_h5(h5_filename, h5_batch_data, h5_batch_label, data_dtype, label_dtype)
        save_h5(h5_filename, h5_batch_data, h5_batch_label, data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, h5_batch_data.shape[0]))
        h5_index += 1
        buffer_size = 0
        # recursive call  递归调用
        insert_batch(data[capacity:, ...], label[capacity:, ...], last_batch)
    if last_batch and buffer_size > 0:
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        # wheat_data_prep_util.save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], data_dtype, label_dtype)
        save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
        h5_index += 1
        buffer_size = 0
    return


sample_cnt = 0
for i, data_label_filename in enumerate(data_label_files):  #遍历.npy文件
    #block_size的大小决定

    #调用indoor3d_util.room2blocks_wrapper_normalized函数对.npy文件进行归一化
    # data, label = true_wheat_indoor3d_util_normal.room2blocks_wrapper_normalized(data_label_filename, NUM_POINT, block_size=1, stride=1, random_sample=False, sample_num=None)    #block_size=1.0   random_sample=False, sample_num=None
    data, label = room2blocks_wrapper_normalized(data_label_filename, NUM_POINT, block_size=1, stride=1, random_sample=False, sample_num=None)  # block_size=1.0   random_sample=False, sample_num=None
    print('{0}, {1}'.format(data.shape, label.shape))  #stride=np.floor(block_size/10.0)
    for _ in range(data.shape[0]):
        fout_room.write(os.path.basename(data_label_filename)[0:-4]+'\n')

    sample_cnt += data.shape[0]
    insert_batch(data, label, i == len(data_label_files)-1)
    print('完成' + str(data_label_filename))
fout_room.close()
print("Total samples: {0}".format(sample_cnt))

for i in range(h5_index):   #生成ply_data_all_数据
    all_file.write(os.path.join('true_wheat_sem_seg_hdf5_data_test', 'ply_data_all_') + str(i) +'.h5' +'\n')
all_file.close()
