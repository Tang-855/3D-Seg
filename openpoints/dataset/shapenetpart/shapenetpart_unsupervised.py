import os
import glob
import h5py
import json
import pickle
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from ..build import DATASETS
from openpoints.models.layers import fps, furthest_point_sample
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import open3d as o3d
import copy
import random
import math
from ..data_augmentation.PointWOLF import PointWOLF
from ..data_augmentation.PartAwareAugmentation import PartAwareAugmentation

'''self_supervise_shapenetpart'''

# 下载shapenet数据集
def download_shapenetpart(DATA_DIR):
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR)):
        os.mkdir(os.path.join(DATA_DIR))
        www = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], os.path.join(DATA_DIR)))
        os.system('rm %s' % (zipfile))

# 加载shapenet数据集
def load_data_partseg(partition, DATA_DIR):
    download_shapenetpart(DATA_DIR)
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(DATA_DIR, '*train*.h5')) \
            + glob.glob(os.path.join(DATA_DIR, '*val*.h5'))
    else:
        file = glob.glob(os.path.join(DATA_DIR, '*%s*.h5' % partition))
    for h5_name in file:
        with h5py.File(h5_name, 'r') as f:
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            seg = f['pid'][:].astype('int64')
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg

# 点云旋转
def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(
        pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

# 加入点云噪声
def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud

# 点云旋转
def rotate_pointcloud(pointcloud):
    theta = np.pi * 2 * np.random.uniform()
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pointcloud[:, [0, 2]] = pointcloud[:, [0, 2]].dot(
        rotation_matrix)  # random rotation (x,z)
    return pointcloud


@DATASETS.register_module()
class ShapeNetPart(Dataset):
    def __init__(self,
                 data_root='data/shapenetpart',
                 num_points=4096,
                 split='train',
                 class_choice=None,
                 shape_classes=16, transform=None):
        self.data, self.label, self.seg = load_data_partseg(split, data_root)
        self.cat2id = {'cotton': 0}
        self.seg_num = [2]
        self.index_start = [0]
        self.num_points = num_points
        self.partition = split
        self.class_choice = class_choice
        self.transform = transform

        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 3
            self.seg_start_index = 0
            self.eye = np.eye(shape_classes)

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'trainval':
            pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]

        # this is model-wise one-hot enocoding for 16 categories of shapes
        feat = np.transpose(self.eye[label, ].repeat(pointcloud.shape[0], 0))
        data = {'pos': pointcloud,
                'x': feat,
                'y': seg}
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return self.data.shape[0]


@DATASETS.register_module()
class ShapeNetPartNormal(Dataset):
    classes = ['cotton']
    seg_num = [2]

    cls_parts = {'cotton': [0, 1]}
    cls2parts = []
    cls2partembed = torch.zeros(1, 2)
    for i, cls in enumerate(classes):
        idx = cls_parts[cls]
        cls2parts.append(idx)
        cls2partembed[i].scatter_(0, torch.LongTensor(idx), 1)
    part2cls = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in cls_parts.keys():
        for label in cls_parts[cat]:
            part2cls[label] = cat

    def __init__(self,
                 data_root='data/useddatasetxyz',
                 num_points=4096,
                 split='train',
                 class_choice=None,
                 use_normal=True,
                 shape_classes=1,
                 presample=False,
                 sampler='fps',
                 transform=None,
                 multihead=False,
                 **kwargs
                 ):
        self.npoints = num_points
        self.root = data_root
        self.transform = transform
        self.catfile = os.path.join(self.root, 'self_supervise_synsetoffset2category.txt')   # synsetoffset2category.txt
        self.cat = {}
        self.use_normal = use_normal
        self.presample = presample
        self.sampler = sampler
        self.split = split
        self.multihead=multihead
        self.part_start = [0]
        self.presample = False

        PointWOLF_1 = not None
        PartAwareAugmentation_1 = None
        self.PointWOLF = PointWOLF() if PointWOLF_1 else None   #not None
        self.PartAwareAugmentation = PartAwareAugmentation() if PartAwareAugmentation_1 else None   #not None

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        if transform is None:
            self.eye = np.eye(shape_classes)
        else:
            self.eye = torch.eye(shape_classes)

        # in the testing, using the uniform sampled 2048 points as input
        # presample
        filename = os.path.join(data_root, 'processed', f'{split}_{num_points}_fps.pkl')
        if presample and not os.path.exists(filename):
            np.random.seed(0)
            self.data, self.cls = [], []
            npoints = []
            for cat, filepath in tqdm(self.datapath, desc=f'Sample ShapeNetPart {split} split'):
                cls = self.classes[cat]
                cls = np.array([cls]).astype(np.int64)
                data = np.loadtxt(filepath).astype(np.float32)
                npoints.append(len(data))
                data = torch.from_numpy(data).to(torch.float32).cuda().unsqueeze(0)
                data = fps(data, num_points).cpu().numpy()[0]
                self.data.append(data)
                self.cls.append(cls)
            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' % (
                split, np.median(npoints), np.average(npoints), np.std(npoints)))
            os.makedirs(os.path.join(data_root, 'processed'), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump((self.data, self.cls), f)
                print(f"{filename} saved successfully")
        elif presample:
            with open(filename, 'rb') as f:
                self.data, self.cls = pickle.load(f)
                print(f"{filename} load successfully")

    def __getitem__(self, index):
        if not self.presample:

            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int64)
            data = np.loadtxt(fn[1]).astype(np.float32)
            #print(len(data))    半监督时将输入的点设置为标注的点数
            if len(data) != 4096:  #4096  50
                data = torch.from_numpy(data).to(torch.float32).cuda().unsqueeze(0)
                data = fps(data, self.npoints).cpu().numpy()[0]
            point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int64)
        else:
            data, cls = self.data[index], self.cls[index]
            point_set, seg = data[:, :6], data[:, 6].astype(np.int64)

        # if 'train' in self.split:
        #     choice = np.random.choice(len(seg), self.npoints, replace=True)
        #     point_set = point_set[choice]
        #     seg = seg[choice]
        # else:
        #     point_set = point_set[:self.npoints]
        #     seg = seg[:self.npoints]
        if self.multihead:
            seg=seg-self.part_start[cls[0]]

        data = {'pos': point_set[:, 0:3],    # XYZ     [8, 3, 4096]
                'x': point_set[:, 3:6],      # 法向量   [8, 3, 4096]
                'cls': cls,                  # [8, 1]
                'y': seg}                    # [8, 4096]

        xyzdata = copy.deepcopy(data)

        # self-supervise
        data2 = copy.deepcopy(data)    # 修改嵌套列表中的元素不会影响原始列表中的相应元素

        # PointWOLF
        if self.PointWOLF is not None:
            data_raw, data['pos'] = self.PointWOLF(data['pos'])
            data2_raw, data2['pos'] = self.PointWOLF(data2['pos'])
        elif self.PartAwareAugmentation is not None:
            data_raw, data['pos'] = self.PartAwareAugmentation(data['pos'])
            data2_raw, data2['pos'] = self.PartAwareAugmentation(data2['pos'])
        else:
            data = self.convertmultiview(data)
            data2 = self.convertmultiview(data2)

            data = self.transformerposition(data)
            data2 = self.transformerposition(data2)




       # print(data.keys())
        """debug 
        from openpoints.dataset import vis_multi_points
        import copy
        old_data = copy.deepcopy(data)
        if self.transform is not None:
            data = self.transform(data)
        vis_multi_points([old_data['pos'][:, :3], data['pos'][:, :3].numpy()], colors=[old_data['x'][:, :3]/255.,data['x'][:, :3].numpy()])
        """
        if self.transform is not None:

            data = self.transform(data)

            # self-supervise
            data2 = self.transform(data2)
            xyzdata = self.transform(xyzdata)

        ##self-supervise
        # for key in data2.keys():
        #     data[key+"1"] = data2[key]

        #print(data.keys())

        return (data,data2,xyzdata)

    def __len__(self):
        return len(self.datapath)

    # self-supervise  # 随机多视角变换
    def convertmultiview(self,data):
        pathlists = ["../../examples/shapenetpart/multiviewparam/viewpoint1.json",
                     "../../examples/shapenetpart/multiviewparam/viewpoint2.json",
                     "../../examples/shapenetpart/multiviewparam/viewpoint3.json",
                     "../../examples/shapenetpart/multiviewparam/viewpoint4.json",
                     "../../examples/shapenetpart/multiviewparam/viewpoint5.json",
                     "../../examples/shapenetpart/multiviewparam/viewpoint6.json",
                     "../../examples/shapenetpart/multiviewparam/viewpoint7.json",
                     "../../examples/shapenetpart/multiviewparam/viewpoint8.json"]
    # def convertmultiview(self,data):
    #     pathlists = ["D:/a-project-T/Pointnext/examples/shapenetpart/multiviewparam/viewpoint1.json",
    #                  "D:/a-project-T/Pointnext/examples/shapenetpart/multiviewparam/viewpoint2.json",
    #                  "D:/a-project-T/Pointnext/examples/shapenetpart/multiviewparam/viewpoint3.json",
    #                  "D:/a-project-T/Pointnext/examples/shapenetpart/multiviewparam/viewpoint4.json",
    #                  "D:/a-project-T/Pointnext/examples/shapenetpart/multiviewparam/viewpoint5.json",
    #                  "D:/a-project-T/Pointnext/examples/shapenetpart/multiviewparam/viewpoint6.json",
    #                  "D:/a-project-T/Pointnext/examples/shapenetpart/multiviewparam/viewpoint7.json",
    #                  "D:/a-project-T/Pointnext/examples/shapenetpart/multiviewparam/viewpoint8.json"]

       # newdatapos = []
       # tempdata = copy.deepcopy(data)
        temppcd = o3d.geometry.PointCloud()
        # temppcd.points = o3d.utility.Vector3dVector(matraix)

        a = data['pos']    # xyz
        param = o3d.io.read_pinhole_camera_parameters(pathlists[random.randint(0, 7)])   # 读取随机多视角变化的参数
        params = param.extrinsic   # 从多视角变化中提取外部参数
        temppcd.points = o3d.utility.Vector3dVector(a)   # 将Python列表或3D向量的NumPy数组转换为与Open3D兼容的数据结构
        # pcd_temp = copy.deepcopy(temppcd)
        temppcd.transform(params)
        nppos = np.asarray(temppcd.points)
        data['pos'] = nppos.astype(np.float32)

        return data

    # self-supervise  # 随机旋转
    def transformerposition(self,data):
        # newdatapos = []
        # newdatafea = []
        # print(data['x'].shape)
        # tempdata = copy.deepcopy(data)

        temppcd = o3d.geometry.PointCloud()
        # temppcd.points = o3d.utility.Vector3dVector(matraix)

        a = data['pos']
        #  b = data['x'][index,:,:].cpu().numpy()

        m = np.eye(3) + np.random.randn(3, 3) * 0.1
        m[0][0] *= np.random.randint(0, 2) * 2 - 1
        # m *= scale
        theta = np.random.rand() * 2 * math.pi
        m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        a = np.matmul(a, m)

        # b = np.matmul(b, m)
        temppcd.points = o3d.utility.Vector3dVector(a)
        temppcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30) )  # 计算法向量

       # temppcd.compute_point_cloud_distance()

       # newdatapos.append(a)
        #newdatafea.append(np.asarray(temppcd.normals))

        nppos = np.asarray(temppcd.points)
        npx = np.asarray(temppcd.normals)

        data['pos'] = nppos.astype(np.float32)
        data['x'] = npx.astype(np.float32)

        # print(tempdata['pos'][0, 0, :])
        # print(tempdata['x'][0, 0, :])
        # time.sleep(50000)

        return data


@DATASETS.register_module()
class ShapeNetPartNormal_raw(Dataset):
    classes = ['cotton']
    seg_num = [2]

    cls_parts = {'cotton': [0, 1]}
    cls2parts = []
    cls2partembed = torch.zeros(1, 2)
    for i, cls in enumerate(classes):
        idx = cls_parts[cls]
        cls2parts.append(idx)
        cls2partembed[i].scatter_(0, torch.LongTensor(idx), 1)
    part2cls = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in cls_parts.keys():
        for label in cls_parts[cat]:
            part2cls[label] = cat

    def __init__(self,
                 data_root='data/useddatasetxyz',
                 num_points=4096,
                 split='train',
                 class_choice=None,
                 use_normal=True,
                 shape_classes=1,
                 presample=False,
                 sampler='fps', 
                 transform=None,
                 multihead=False,
                 **kwargs
                 ):
        self.npoints = num_points
        self.root = data_root
        self.transform = transform
        self.catfile = os.path.join(self.root, 'self_supervise_synsetoffset2category.txt')   # synsetoffset2category.txt
        self.cat = {}
        self.use_normal = use_normal
        self.presample = presample
        self.sampler = sampler 
        self.split = split
        self.multihead=multihead
        self.part_start = [0]
        self.presample = False

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        if transform is None:
            self.eye = np.eye(shape_classes)
        else:
            self.eye = torch.eye(shape_classes)

        # in the testing, using the uniform sampled 2048 points as input
        # presample
        filename = os.path.join(data_root, 'processed', f'{split}_{num_points}_fps.pkl')
        if presample and not os.path.exists(filename):
            np.random.seed(0)
            self.data, self.cls = [], []
            npoints = []
            for cat, filepath in tqdm(self.datapath, desc=f'Sample ShapeNetPart {split} split'):
                cls = self.classes[cat]
                cls = np.array([cls]).astype(np.int64)
                data = np.loadtxt(filepath).astype(np.float32)
                npoints.append(len(data))
                data = torch.from_numpy(data).to(torch.float32).cuda().unsqueeze(0)
                data = fps(data, num_points).cpu().numpy()[0]
                self.data.append(data)
                self.cls.append(cls)
            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' % (
                split, np.median(npoints), np.average(npoints), np.std(npoints)))
            os.makedirs(os.path.join(data_root, 'processed'), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump((self.data, self.cls), f)
                print(f"{filename} saved successfully")
        elif presample:
            with open(filename, 'rb') as f:
                self.data, self.cls = pickle.load(f)
                print(f"{filename} load successfully")

    def __getitem__(self, index):
        if not self.presample:

            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int64)
            data = np.loadtxt(fn[1]).astype(np.float32)
            #print(len(data))    半监督时将输入的点设置为标注的点数
            if len(data) != 4096:  #4096  50
                data = torch.from_numpy(data).to(torch.float32).cuda().unsqueeze(0)
                data = fps(data, self.npoints).cpu().numpy()[0]
            point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int64)
        else:
            data, cls = self.data[index], self.cls[index]
            point_set, seg = data[:, :6], data[:, 6].astype(np.int64)

        # if 'train' in self.split:
        #     choice = np.random.choice(len(seg), self.npoints, replace=True)
        #     point_set = point_set[choice]
        #     seg = seg[choice]
        # else:
        #     point_set = point_set[:self.npoints]
        #     seg = seg[:self.npoints]
        if self.multihead:
            seg=seg-self.part_start[cls[0]]

        data = {'pos': point_set[:, 0:3],    # XYZ     [8, 3, 4096]
                'x': point_set[:, 3:6],      # 法向量   [8, 3, 4096]
                'cls': cls,                  # [8, 1]
                'y': seg}                    # [8, 4096]

        xyzdata = copy.deepcopy(data)

        #self-supervise
        data2 = copy.deepcopy(data)    # 修改嵌套列表中的元素不会影响原始列表中的相应元素

        data = self.convertmultiview(data)
        data2 = self.convertmultiview(data2)

        data = self.transformerposition(data)
        data2 = self.transformerposition(data2)

       # print(data.keys())
        """debug 
        from openpoints.dataset import vis_multi_points
        import copy
        old_data = copy.deepcopy(data)
        if self.transform is not None:
            data = self.transform(data)
        vis_multi_points([old_data['pos'][:, :3], data['pos'][:, :3].numpy()], colors=[old_data['x'][:, :3]/255.,data['x'][:, :3].numpy()])
        """
        if self.transform is not None:

            data = self.transform(data)

            # self-supervise
            data2 = self.transform(data2)
            xyzdata = self.transform(xyzdata)

        ##self-supervise
        # for key in data2.keys():
        #     data[key+"1"] = data2[key]

        #print(data.keys())

        return (data,data2,xyzdata)

    def __len__(self):
        return len(self.datapath)

    # self-supervise  # 随机多视角变换
    def convertmultiview(self,data):
        pathlists = ["../../examples/shapenetpart/multiviewparam/viewpoint1.json",
                     "../../examples/shapenetpart/multiviewparam/viewpoint2.json",
                     "../../examples/shapenetpart/multiviewparam/viewpoint3.json",
                     "../../examples/shapenetpart/multiviewparam/viewpoint4.json",
                     "../../Pointnext/examples/shapenetpart/multiviewparam/viewpoint5.json",
                     "../../Pointnext/examples/shapenetpart/multiviewparam/viewpoint6.json",
                     "../../Pointnext/examples/shapenetpart/multiviewparam/viewpoint7.json",
                     "../../Pointnext/examples/shapenetpart/multiviewparam/viewpoint8.json"]

       # newdatapos = []
       # tempdata = copy.deepcopy(data)
        temppcd = o3d.geometry.PointCloud()
        # temppcd.points = o3d.utility.Vector3dVector(matraix)

        a = data['pos']    # xyz
        param = o3d.io.read_pinhole_camera_parameters(pathlists[random.randint(0, 7)])   # 读取随机多视角变化的参数
        params = param.extrinsic   # 从多视角变化中提取外部参数
        temppcd.points = o3d.utility.Vector3dVector(a)   # 将Python列表或3D向量的NumPy数组转换为与Open3D兼容的数据结构
        # pcd_temp = copy.deepcopy(temppcd)
        temppcd.transform(params)
        nppos = np.asarray(temppcd.points)
        data['pos'] = nppos.astype(np.float32)

        return data

    # self-supervise  # 随机旋转
    def transformerposition(self,data):
        # newdatapos = []
        # newdatafea = []
        # print(data['x'].shape)
        # tempdata = copy.deepcopy(data)

        temppcd = o3d.geometry.PointCloud()
        # temppcd.points = o3d.utility.Vector3dVector(matraix)

        a = data['pos']
        #  b = data['x'][index,:,:].cpu().numpy()

        m = np.eye(3) + np.random.randn(3, 3) * 0.1
        m[0][0] *= np.random.randint(0, 2) * 2 - 1
        # m *= scale
        theta = np.random.rand() * 2 * math.pi
        m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        a = np.matmul(a, m)

        # b = np.matmul(b, m)
        temppcd.points = o3d.utility.Vector3dVector(a)
        temppcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30) )  # 计算法向量

       # temppcd.compute_point_cloud_distance()

       # newdatapos.append(a)
        #newdatafea.append(np.asarray(temppcd.normals))

        nppos = np.asarray(temppcd.points)
        npx = np.asarray(temppcd.normals)

        data['pos'] = nppos.astype(np.float32)
        data['x'] = npx.astype(np.float32)

        # print(tempdata['pos'][0, 0, :])
        # print(tempdata['x'][0, 0, :])
        # time.sleep(50000)

        return data


# CurveNet DatSet of ShapenetPart
@DATASETS.register_module()
class ShapeNetPartCurve(Dataset):
    classes = ['cotton']
    seg_num = [3]

    cls_parts = {'cotton': [0, 1]}
    cls2parts = []
    cls2partembed = torch.zeros(1, 2)
    for i, cls in enumerate(classes):
        idx = cls_parts[cls]
        cls2parts.append(idx)
        cls2partembed[i].scatter_(0, torch.LongTensor(idx), 1)
    part2cls = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in cls_parts.keys():
        for label in cls_parts[cat]:
            part2cls[label] = cat


    def __init__(self,
                 data_root='data/ShapeNetPart/hdf5_data',
                 num_points=4096,
                 split='train', class_choice=None, use_normal=True, transform=None, **kwargs):
        self.data, self.label, self.seg = load_data_partseg(split, data_root)
        self.cat2id = {'cotton': 0}
        self.seg_num = [2]
        self.index_start = [0]
        self.num_points = num_points
        self.partition = split
        self.use_normal = use_normal
        self.class_choice = class_choice
        self.transform = transform
        self.in_channels = 3
        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 3
            self.seg_start_index = 0

    def __getitem__(self, index):
        point_set = self.data[index][:self.num_points]
        cls = self.label[index]
        seg = self.seg[index][:self.num_points]
        point_set = point_set[:self.num_points]
        seg = seg[:self.num_points]

        if 'train' in self.partition:
            indices = list(range(point_set.shape[0]))
            np.random.shuffle(indices)
            point_set = point_set[indices]
            seg = seg[indices]

        data = {'pos': point_set[:, 0:3],
                'cls': cls.astype(np.int64),
                'y': seg}
        if self.transform is not None:
            data = self.transform(data)

        if 'heights' in data.keys():
            data['x'] = data['heights']
        return data

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ShapeNetPartNormal(num_points=2048, split='trainval')
    test = ShapeNetPartNormal(num_points=2048, split='test')
    for dict in train:
        for i in dict:
            print(i, dict[i].shape)