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
import open3d as o3d
import copy
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from openpoints.models.layers import create_grouper

from ..data_augmentation.PointWOLF import PointWOLF
from ..data_augmentation.PartAwareAugmentation import PartAwareAugmentation

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


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


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
                 num_points=2048,
                 split='train',
                 class_choice=None,
                 shape_classes=16, transform=None):
        self.data, self.label, self.seg = load_data_partseg(split, data_root)
        self.cat2id = {'cotton': 0}
        self.seg_num = [3]
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


def point_cloud_normalize(cloud):
    """
    对点云数据进行归一化
    :param cloud: 需要归一化的点云数据
    :return: 归一化后的点云数据
    """
    centroid = cloud.get_center()  # 计算点云质心
    points = np.asarray(cloud.points)
    points = points - centroid     # 去质心
    m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))  # 计算点云中的点与坐标原点的最大距离
    points = points / m  # 对点云进行缩放
    normalize_cloud = o3d.geometry.PointCloud()  # 使用numpy生成点云
    normalize_cloud.points = o3d.utility.Vector3dVector(points)

   # normalize_cloud.colors = cloud.colors  # 获取投影前对应的颜色赋值给投影后的点
    return normalize_cloud


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
                 wi=0.45,
                 is_newsample=False,
                 **kwargs
                 ):
        self.npoints = num_points
        self.root = data_root
        self.transform = transform
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.use_normal = use_normal
        self.presample = presample
        self.sampler = sampler
        self.split = split
        self.multihead=multihead
        self.part_start = [0]
        self.wi = wi
        self.is_newsample = is_newsample


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
            num_points = 4096
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
            # 可视化第一步
            data_1 = copy.deepcopy(data[:, 0:6])

            #################
            import open3d as o3d
            temppcd = o3d.geometry.PointCloud()
            temppcd.points = o3d.utility.Vector3dVector(data[:,:3])
            temppcd = point_cloud_normalize(temppcd)
            temppcd.estimate_normals(  # 计算法向量
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            npp = np.asarray(temppcd.points)
            npx = np.asarray(temppcd.normals)
            data[:,:3] = npp
            data[:,3:6] = npx
            ###################

            if 'train' in self.split:

                if self.is_newsample:

                    spikenums = int(
                        self.npoints * self.wi if len(data[data[:, -1] == 1]) >= self.npoints * self.wi else len(
                            data[data[:, -1] == 1]))
                    spikedata = data[data[:, -1] == 1]
                    leafdata = data[data[:, -1] == 0]
                    if spikenums > 0:
                        if spikenums < len(data[data[:, -1] == 1]):
                            choice = np.random.choice(len(spikedata), spikenums, replace=True)
                            samplespikedata = spikedata[choice]
                            # samplespikedata = torch.from_numpy(spikedata).to(torch.float32).cuda().unsqueeze(0)
                            # samplespikedata = fps(samplespikedata, spikenums).cpu().numpy()[0]
                        else:
                            samplespikedata = spikedata

                        # print("spikenum",spikenums)
                        # group_args = {'NAME': 'ballquery', 'radius_scaling':2.5,
                        #              'radius': 2, 'nsample': 16, 'return_only_idx': True}
                        group_args = {'NAME': 'knn', 'nsample': self.npoints // 8, 'return_only_idx': True}

                        grouper = create_grouper(group_args)
                        leafidx = \
                        grouper(torch.from_numpy(samplespikedata[:, :3]).to(torch.float32).cuda().unsqueeze(0),
                                torch.from_numpy(leafdata[:, :3]).to(torch.float32).cuda().unsqueeze(0)).cpu().numpy()[0]
                        leafidx = leafidx.flatten()
                        leafidx = np.unique(leafidx)
                        knnleafdata = leafdata[leafidx, :]
                        leafnums = int(
                            (self.npoints - spikenums) if len(leafidx) > (self.npoints - spikenums) else len(leafidx))

                        if leafnums < len(leafidx):
                            choice = np.random.choice(len(knnleafdata), leafnums, replace=True)
                            knnleafdata = knnleafdata[choice]
                            # knnleafdata = torch.from_numpy(knnleafdata).to(torch.float32).cuda().unsqueeze(0)
                            # knnleafdata = fps(knnleafdata, leafnums).cpu().numpy()[0]
                            data = np.concatenate((samplespikedata, knnleafdata), axis=0)

                        else:
                            # knnleafdata = knnleafdata
                            remaindernums = self.npoints - spikenums - leafnums
                            if remaindernums > 0:
                                remainderleafdata = np.delete(leafdata, leafidx, axis=0)
                                choice = np.random.choice(len(remainderleafdata), remaindernums, replace=True)
                                remainderleafdata = remainderleafdata[choice]
                                # remainderleafdata = torch.from_numpy(remainderleafdata).to(
                                #     torch.float32).cuda().unsqueeze(0)
                                # remainderleafdata = fps(remainderleafdata, remaindernums).cpu().numpy()[0]
                                data = np.concatenate((samplespikedata, knnleafdata, remainderleafdata), axis=0)
                            else:
                                data = np.concatenate((samplespikedata, knnleafdata), axis=0)
                    else:
                        np.random.seed(0)
                        choice = np.random.choice(len(data), self.npoints, replace=True)  # self.npoints
                        data = data[choice]

                    #print(len(data))
                    point_set = data[:, 0:6]
                    seg = data[:, -1].astype(np.int64)
                   # print(len(data[data[:,-1]==1])/len(data))

                else:
                    np.random.seed(0)
                    # point_set = data[:, 0:6]
                    # seg = data[:, -1].astype(np.int64)
                    if len(data) != self.npoints:
                        data = torch.from_numpy(data).to(torch.float32).cuda().unsqueeze(0)
                        data = fps(data, self.npoints).cpu().numpy()[0]

                    choice = np.random.choice(len(data), self.npoints, replace=True)  # self.npoints
                    data = data[choice]
                    point_set = data[:, 0:6]
                    seg = data[:, -1].astype(np.int64)
                    # point_set = point_set[choice]
                    # seg = seg[choice]
            else:
                if len(data) != self.npoints:
                    data = torch.from_numpy(data).to(torch.float32).cuda().unsqueeze(0)
                    data = fps(data, self.npoints).cpu().numpy()[0]

                # choice = np.random.choice(len(data), self.npoints, replace=True)  # self.npoints
                # data = data[choice]
                point_set = data[:, 0:6]
                seg = data[:, -1].astype(np.int64)

            # if len(data) != 2048:
            #     data = torch.from_numpy(data).to(
            #         torch.float32).cuda().unsqueeze(0)
            #     data = fps(data, self.npoints).cpu().numpy()[0]
                # choice = np.random.choice(len(data), 2048, replace=True)  # self.npoints
                # data = data[choice]
            #
            # point_set = data[:, 0:6]
            # seg = data[:, -1].astype(np.int64)
        else:
            data, cls = self.data[index], self.cls[index]
            point_set, seg = data[:, :6], data[:, 6].astype(np.int64)



        # if 'train' in self.split:
        #
        #     np.random.seed(0)
        #     data = torch.from_numpy(data).to(
        #         torch.float32).cuda().unsqueeze(0)
        #     data = fps(data, self.npoints).cpu().numpy()[0]  # self.npoints
        #     point_set = data[:, 0:6]
        #     seg = data[:, -1].astype(np.int64)
        #
        #     choice = np.random.choice(len(seg), self.npoints , replace=True)  #self.npoints
        #     point_set = point_set[choice]
        #     seg = seg[choice]
        # else:
        #     point_set = point_set[:self.npoints]
        #     seg = seg[:self.npoints]

        if self.multihead:
            seg=seg-self.part_start[cls[0]]


        # data = {
        #     'points': point_set[:, 0:3],
        #     'seg_id': seg,
        #     'cls_tokens': cls,
        #     'norms': point_set[:, 3:6]
        # }


        #######pointnext and pointnet2,
        ori_data = data_1       # 测试可视化时加的，第二步
        data = {'pos': point_set[:, 0:3],
                'x': point_set[:, 3:6],
                'cls': cls,
                'y': seg}
        data['ori_data'] = ori_data   # 测试可视化时加的，第三步
        ###############################

        # data = {'pos': point_set,
        #         'cls': cls,
        #         'y': seg}

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

        '''跑pointstack时需要加上此代码'''
        # data = {
        #     'points': data['pos'],
        #     'seg_id': data['y'],
        #     'cls_tokens': data['cls'],
        #     'norms': data['x']
        # }
        ''''''

        return data

    def __len__(self):
        return len(self.datapath)


# CurveNet DatSet of ShapenetPart
@DATASETS.register_module()
class ShapeNetPartCurve(Dataset):
    classes = ['cotton']
    seg_num = [3]

    cls_parts = {'cotton': [0, 1, 2]}
    cls2parts = []
    cls2partembed = torch.zeros(1, 3)
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
                 num_points=2048,
                 split='train', class_choice=None, use_normal=True, transform=None, **kwargs):
        self.data, self.label, self.seg = load_data_partseg(split, data_root)
        self.cat2id = {'cotton': 0}
        self.seg_num = [3]
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