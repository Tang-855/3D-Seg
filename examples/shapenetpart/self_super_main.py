# -*- coding: utf-8 -*-
"""
Distributed training script for scene segmentation with S3DIS dataset
"""
import argparse
import random

import yaml
# from datautils.forafterpointDataLoader import ShapeNetPartNormal
import os
import sys
import logging
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
#from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
#from sklearn.metrics import confusion_matrix
from collections import defaultdict, Counter

torch.backends.cudnn.benchmark = False
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

from openpoints.models.pointnet.pointnet_part_seg import pointnet_part     # 1-pointnet模型
from openpoints.models.pointnet2.pointnet2_part_seg_msg import pointnet2   # 2-pointnet++模型
from openpoints.models.dgcnn.model import DGCNN_partseg                    # 3-DGCNN模型
from openpoints.models.pct.pct import Point_Transformer_partseg            # 4-Point_Transformer模型
from openpoints.models.GDANet.GDANet_ptseg import GDANet                   # 5-GDANet模型
from openpoints.models.pointmlp.pointMLP import pointMLP                   # 6-pointMLP模型
from openpoints.models.curvenet.curvenet_seg import CurveNet               # 7-CurveNet模型
from openpoints.models.pointstack.PointStack import MyPointStack           # 8-pointstack模型
from openpoints.models.mymodel_pct.model import PointTransformerSeg   # 自己的模型

from openpoints.models import build_model_from_cfg
from openpoints.models.layers import torch_grouping_operation, knn_point
from openpoints.loss import build_criterion_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.dataset import build_dataloader_from_cfg, get_class_weights, get_features_by_keys
from openpoints.transforms import build_transforms_from_cfg
from openpoints.utils import AverageMeter, ConfusionMatrix
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port
from openpoints.models.layers import furthest_point_sample
import open3d as o3d
from openpoints.models.layers import myfps
from datautils.SelfSuperviseLoss import My_super_loss, ChamferLoss, MultiLossLayer, My_super_loss_groupouter, My_super_loss_kmeans2
import math
import time
import random
import copy
from audtorch.metrics.functional import pearsonr

global record_iter_train
record_iter_train = {'epoch':[], 'loss':[]}
global record_epoch
record_epoch = {'epoch':[], 'train':[]}


# 互相关矩阵计算
def calcorrcoef(data,data1):
    # datatemp = torch.broadcast_tensors(data[0], data1)[0]
    # datatemp = pearsonr(datatemp, data1)
    # for i in range(1, len(data)):
    #     datatemp = torch.cat((datatemp, pearsonr(torch.broadcast_tensors(data[i], data1)[0], data1)), dim=1)
    # print(datatemp)

    data = data.permute(0,2,1)
    cofmatrix = (torch.matmul(data, data1))/512  #互相关矩阵，即X Y.T的期望
    return cofmatrix

#随机多视角变换
def convertmultiview(data):
    pathlists = ["./examples/shapenetpart/multiviewparam/viewpoint1.json","./examples/shapenetpart/multiviewparam/viewpoint2.json",
                 "./examples/shapenetpart/multiviewparam/viewpoint3.json","./examples/shapenetpart/multiviewparam/viewpoint4.json",
                 "./examples/shapenetpart/multiviewparam/viewpoint5.json","./examples/shapenetpart/multiviewparam/viewpoint6.json",
                 "./examples/shapenetpart/multiviewparam/viewpoint7.json","./examples/shapenetpart/multiviewparam/viewpoint8.json"]
    newdatapos = []
    tempdata = copy.deepcopy(data)
    temppcd = o3d.geometry.PointCloud()
    # temppcd.points = o3d.utility.Vector3dVector(matraix)

    for pcpos in data['pos'][0:, :, :]:
        a = pcpos.cpu().numpy()
        param = o3d.io.read_pinhole_camera_parameters(pathlists[random.randint(0,7)])
        temppcd.points = o3d.utility.Vector3dVector(a)
       # pcd_temp = copy.deepcopy(temppcd)
        params = param.extrinsic
        temppcd.transform(params)
        newdatapos.append(np.asarray(temppcd.points))

    nppos = np.asarray(newdatapos)
    tempdata['pos'] = torch.from_numpy(nppos.astype(np.float32))

    return tempdata

#随机旋转
def transformerposition(data):
    newdatapos = []
    newdatafea = []
    #print(data['x'].shape)
    #tempdata = copy.deepcopy(data)
    temppcd = o3d.geometry.PointCloud()
    #temppcd.points = o3d.utility.Vector3dVector(matraix)

    for pcpos in data['pos'][0:,:,:]:
        a = pcpos.cpu().numpy()
        # b = data['x'][index,:,:].cpu().numpy()
        m = np.eye(3) + np.random.randn(3, 3) * 0.1
        m[0][0] *= np.random.randint(0, 2) * 2 - 1
        # m *= scale
        theta = np.random.rand() * 2 * math.pi
        m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        a = np.matmul(a, m)

       # b = np.matmul(b, m)
        temppcd.points = o3d.utility.Vector3dVector(a)
        temppcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))       #计算法向量
        newdatapos.append(a)
        newdatafea.append(np.asarray(temppcd.normals))

    nppos = np.asarray(newdatapos)
    npx = np.asarray(newdatafea)
    data['pos'] = torch.from_numpy(nppos.astype(np.float32))
    data['x'] = torch.from_numpy(npx.astype(np.float32))
    # print(tempdata['pos'][0, 0, :])
    # print(tempdata['x'][0, 0, :])
    # time.sleep(50000)
    return data

###随机几何变换
def Transform(a):
    #a = a[:, 1:]
    m = np.eye(3)+np.random.randn(3, 3)*0.1
    m[0][0] *= np.random.randint(0, 2)*2-1
    #m *= scale
    theta = np.random.rand()*2*math.pi
    m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    a = np.matmul(a, m)
    return a


# 返回矩阵的非对角线元素的展平视图
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def lossforward(x, y):
    lambd = 3.9e-3
    #print(x)
    # x_mean = torch.mean(x, dim=0)
    # y_mean = torch.mean(y, dim=0)
    # x_std = torch.std(x, dim=0)
    # y_std = torch.std(y, dim=0)
    # x = torch.div(torch.sub(x, x_mean), x_std)
    # y = torch.div(torch.sub(y, y_mean), y_std)
    up = torch.mm(x.t(), y)
    # print(up)
    # down1 = x.pow(2).sum(0, keepdim=True).sqrt()
    # down2 = y.pow(2).sum(0, keepdim=True).sqrt()
    # down = torch.mm(down1.t(), down2)
    #cov = up / down
    cov = up

    # covnp = cov.cpu().detach().numpy()
    # np.save('/home/aidrive1/workspace/luoly/dataset/Min_scan/bt_train/cov/cov_%02d.npy' % (i), covnp)
    ret = cov - torch.eye(cov.shape[0]).cuda()
    on_diag = torch.diagonal(ret).add_(-1).pow_(2).sum().mul(1 / 512)
    off_diag = off_diagonal(ret).pow_(2).sum().mul(1 / 512)
    loss = on_diag + lambd * off_diag
    return loss


def batched_bincount(x, dim, max_value):
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target


def part_seg_refinement(pred, pos, cls, cls2parts, n=10):
    pred_np = pred.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):  # sample_idx
        parts = cls2parts[cls[shape_idx]]
        counter_part = Counter(pred_np[shape_idx])
        if len(counter_part) > 1:
            for i in counter_part:
                if counter_part[i] < n or i not in parts:
                    less_idx = np.where(pred_np[shape_idx] == i)[0]
                    less_pos = pos[shape_idx][less_idx]
                    knn_idx = knn_point(n + 1, torch.unsqueeze(less_pos, axis=0), torch.unsqueeze(pos[shape_idx], axis=0))[1]
                    neighbor = torch_grouping_operation(pred[shape_idx:shape_idx + 1].unsqueeze(1), knn_idx)[0][0]
                    counts = batched_bincount(neighbor, 1, cls2parts[-1][-1] + 1)
                    counts[:, i] = 0
                    pred[shape_idx][less_idx] = counts.max(dim=1)[1]
    return pred


def get_ins_mious(pred, target, cls, cls2parts, multihead=False,):
    """Get the Shape IoU
    shape IoU: the mean part iou for each shape
    Args:
        pred (_type_): _description_
        target (_type_): _description_
        num_classes (_type_): _description_
    Returns:
        _type_: _description_
    """
    ins_mious = []
    for shape_idx in range(pred.shape[0]):  # sample_idx
        part_ious = []
        parts = cls2parts[cls[shape_idx]]
        if multihead:
            parts = np.arange(len(parts))
        for part in parts:
            pred_part = pred[shape_idx] == part
            target_part = target[shape_idx] == part
            I = torch.logical_and(pred_part, target_part).sum()
            U = torch.logical_or(pred_part, target_part).sum()
            if U == 0:
                iou = torch.tensor(100, device=pred.device, dtype=torch.float32)
            else:
                iou = I * 100 / float(U)
            part_ious.append(iou)
        ins_mious.append(torch.mean(torch.stack(part_ious)))

    return ins_mious


def main(gpu, cfg):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url, world_size=cfg.world_size, rank=cfg.rank)   # cfg.world_size
        dist.barrier()
    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0:
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        #writer = SummaryWriter(log_dir=cfg.run_dir)
        writer = None
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    # build dataset
    val_loader = build_dataloader_from_cfg(cfg.batch_size, cfg.dataset, cfg.dataloader, datatransforms_cfg=cfg.datatransforms, split='val', distributed=cfg.distributed)
    test_loader = build_dataloader_from_cfg(cfg.batch_size, cfg.dataset, cfg.dataloader, datatransforms_cfg=cfg.datatransforms, split='test', distributed=cfg.distributed)
    train_loader = build_dataloader_from_cfg(cfg.batch_size, cfg.dataset, cfg.dataloader,datatransforms_cfg=cfg.datatransforms, split='train', distributed=cfg.distributed,)
    logging.info(f"length of validation dataset: {len(test_loader.dataset)}")
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")

    num_classes = val_loader.dataset.num_classes if hasattr(val_loader.dataset, 'num_classes') else None
    if num_classes is not None:
        assert cfg.num_classes == num_classes
    logging.info(f"number of classes of the dataset: {num_classes}")
    cfg.cls2parts = val_loader.dataset.cls2parts
    validate_fn = eval(cfg.get('val_fn', 'validate'))

   # if cfg.model.get('decoder_args', None):
        #cfg.model.decoder_args.cls2partembed = val_loader.dataset.cls2partembed

    cfg.model.decoder_args.cls2partembed = val_loader.dataset.cls2partembed

    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

    # model = build_model_from_cfg(cfg.model).cuda()
    device = torch.device('cuda:0')
    # model = pointnet_part().to(device)    # 1-PointNet模型
    # model = pointnet2().cuda()            # 2-PointNet++模型
    model = DGCNN_partseg().to(device)    # 3-DGCNN_partseg模型
    # model = Point_Transformer_partseg().to(device)       # 4-Point_Transformer模型
    # model = GDANet().to(device)           # 5-GDANet模型
    # model = pointMLP(2,4096).to(device)     # 6-pointMLP模型   partnums, points
    # model = CurveNet().to(device)           # 7-CurveNet模型
    # model = MyPointStack().to(device)       # 8-PointStack模型  partnums, points
    # model = build_model_from_cfg(cfg.model).cuda()   # 9-Spotr模型   10-Pointnext模型

    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))

    ##加载自监督参数
    # #加载自监督参数 (在selfprepath和temppath中任选一种 即可)
    '''方法一'''
    try:
        # selfprepath = r"D:\a-project-T\Pointnext\examples\shapenetpart\log\shapenetpart\1_UnGlobal\checkpoint\_ckpt_latest.pth"
        selfprepath = r"/data/home/whs/Pointnext/examples/shapenetpart/log/shapenetpart/DGCNN/checkpoint/_ckpt_latest.pth"
        # selfprepath = r".\log\shapenetpart\DGCNN\checkpoint\_ckpt_latest.pth"
        logging.info('load self-pretrained params')
        current_weights = model.state_dict()  # 参看当前模型的参数模型
        load_checkpoint(model, pretrained_path=selfprepath)
        pretrained_model = torch.load(selfprepath)
        # pretrained_model = torch.load(r".\log\shapenetpart\supervise-pointnet2_seed14\checkpoint\_ckpt_latest.pth")  # 加载预训练模型
        cfg.start_epoch = pretrained_model['epoch']+1  # 加载当前的预训练epoch
        logging.info('**************************Use pretrain model**************************')
    except:
        logging.info('No existing model, starting training from scratch...')
    # selfprepath = r"D:\pythonproject\Pointnext\examples\shapenetpart\log\shapenetpart\virtuepart4\checkpoint\ckpt_best.pth"
    # logging.info('load self-pretrained params')
    # load_checkpoint(model, pretrained_path=selfprepath)

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # transforms
    if 'vote' in cfg.datatransforms:
        voting_transform = build_transforms_from_cfg('vote', cfg.datatransforms)
    else:
        voting_transform = None

    model_module = model.module if hasattr(model, 'module') else model
    # optionally resume from a checkpoint
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler, pretrained_path=cfg.pretrained_path)
            test_ins_miou, test_cls_miou, test_cls_mious = validate_fn(model, val_loader, cfg, num_votes=cfg.num_votes, data_transform=voting_transform)
            logging.info(f'\nresume val instance mIoU is {test_ins_miou}, val class mIoU is {test_cls_miou} \n ')
        else:
            if cfg.mode in ['val', 'test']:
                load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                test_ins_miou, test_cls_miou, test_cls_mious = validate_fn(model, val_loader, cfg, num_votes=cfg.num_votes, data_transform=voting_transform)
                return test_ins_miou
            elif cfg.mode == 'finetune':
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model, pretrained_path=cfg.pretrained_path)
            elif cfg.mode == 'finetune_encoder':
                logging.info(f'Load encoder only, finetuning from {cfg.pretrained_path}')
                load_checkpoint(model_module.encoder, pretrained_path=cfg.pretrained_path)
    else:
        logging.info('Training from scratch')



    # 加载权重和损失
    if cfg.get('cls_weighed_loss', False):
        if hasattr(train_loader.dataset, 'num_per_class'):
            cfg.criterion_args.weight = None
            cfg.criterion_args.weight = get_class_weights(train_loader.dataset.num_per_class, normalize=True)
        else:
            logging.info('`num_per_class` attribute is not founded in dataset')

    #criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()

    globalcriterion = My_super_loss_kmeans2(cfg.batch_size).cuda()  # batch_size   全局损失
    # globalcriterion = My_super_loss_groupouter(cfg.batch_size).cuda()  # batch_size   全局损失
    # globalcriterion = My_super_loss(cfg.batch_size).cuda()  # batch_size   全局损失
    localcriterion = ChamferLoss().cuda()                   # batch_size   局部损失
    #finalcriterion = Finalloss().cuda()
    finalcriterion = MultiLossLayer(2).cuda()               # 3   局部损失

    # ===> start training
    best_ins_miou, cls_miou_when_best, cls_mious_when_best = 0., 0., []

    minloss = 99

    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        # some dataset sets the dataset length as a fixed steps.
        if hasattr(train_loader.dataset, 'epoch'):
            train_loader.dataset.epoch = epoch - 1
        cfg.epoch = epoch
        train_loss = \
            train_one_epoch(model, train_loader, globalcriterion, localcriterion, finalcriterion, optimizer, scheduler, epoch, cfg)


        record_epoch['epoch'].append(epoch)
        record_epoch['train'].append(train_loss)  # 每次epoch的loss

        if train_loss < minloss:
            # mysavepath = "D:/pythonproject/Pointnext/examples/shapenetpart/mybestself.pth"
            logging.info('finding the new best model params...')
            minloss = train_loss
            save_checkpoint(cfg, model, epoch, optimizer, scheduler)
            #torch.save(model.state_dict(),mysavepath)

        # if epoch % cfg.val_freq == 0:
        #     val_ins_miou, val_cls_miou, val_cls_mious = validate_fn(model, val_loader, cfg)
        #     if val_ins_miou > best_ins_miou:
        #         best_ins_miou = val_ins_miou
        #         cls_miou_when_best = val_cls_miou
        #         cls_mious_when_best = val_cls_mious
        #         best_epoch = epoch
        #         is_best = True
        #         with np.printoptions(precision=2, suppress=True):
        #             logging.info(
        #                 f'Find a better ckpt @E{epoch}, val_ins_miou {best_ins_miou:.2f} val_cls_miou {cls_miou_when_best:.2f}, '
        #                 f'\ncls_mious: {cls_mious_when_best}')


       # lr = optimizer.param_groups[0]['lr']


        # if writer is not None:
        #     writer.add_scalar('val_ins_miou', val_ins_miou, epoch)
        #     writer.add_scalar(1'val_class_miou', val_cls_miou, epoch)
        #     writer.add_scalar('best_val_instance_miou',
        #                       best_ins_miou, epoch)
        #     writer.add_scalar('val_class_miou_when_best', cls_miou_when_best, epoch)
        #     writer.add_scalar('train_loss', train_loss, epoch)
        #     writer.add_scalar('lr', lr, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)

        # 保存为Excel文件
    model_name = cfg.cfg_basename.split('_')[0]
    save_path = 'log/pictures/' + model_name + '1'  # 指定
    if not os.path.exists(save_path):  # 如果文件夹不存在，则创建它
        os.makedirs(save_path)

    df1 = pd.DataFrame(record_iter_train)
    save_path_2 = save_path + '/' + 'record_iter_train.xlsx'  # 指定文件夹路径
    df1.to_excel(save_path_2, index=False)

    df3 = pd.DataFrame(record_epoch)
    save_path_4 = save_path + '/' + 'record_epoch.xlsx'  # 指定文件夹路径
    df3.to_excel(save_path_4, index=False)

    '''画图代码
    # 设置x轴和y轴范围及刻度
    fig, ax = plt.subplots()
    plt.xlim(0, cfg.epochs + 1)
    plt.xticks(np.arange(0, cfg.epochs + 1, 3))
    max_y1 = math.ceil(max(record_iter_train['loss']))
    if max_y1 <= 1:
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, max_y1, 0.1))
    else:
        plt.ylim(0, max_y1)
        plt.yticks(np.arange(0, max_y1, max_y1 / 10))
    ax.legend()
    fig.savefig(os.path.join(save_path, 'Iter_Loss.png'))
    # plt.show()

    # 绘制loss的折线图并保存到指定文件夹
    fig, ax = plt.subplots()
    ax.plot(record_epoch['train'], label='Train Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    # 设置x轴和y轴范围及刻度
    plt.xlim(0, epoch + 1)
    plt.xticks(np.arange(0, epoch + 1, 3))
    max_y1 = math.ceil(max(record_epoch['train']))
    if max_y1 <= 1:
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, max_y1, 0.1))
    else:
        plt.ylim(0, max_y1)
        plt.yticks(np.arange(0, max_y1, max_y1 / 10))
    ax.legend()
    fig.savefig(os.path.join(save_path, 'Loss.png'))
    '''

    # plt.show()
    # if writer is not None:
    #     Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    # Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.logname}_ckpt_latest.pth'))
    # with np.printoptions(precision=2, suppress=True):
    #     logging.info(f'Best Epoch {best_epoch},'
    #                  f'Instance mIoU {best_ins_miou:.2f}, '
    #                  f'Class mIoU {cls_miou_when_best:.2f}, '
    #                  f'\n Class mIoUs {cls_mious_when_best}')

    # if cfg.get('num_votes', 0) > 0:
    #     load_checkpoint(model, pretrained_path=os.path.join(
    #         cfg.ckpt_dir, 'ckpt_best.pth'))
    #     set_random_seed(cfg.seed)
                                                              #     test_ins_miou, test_cls_miou, test_cls_mious  = validate_fn(model, val_loader, cfg, num_votes=cfg.get('num_votes', 0),
    #                              data_transform=voting_transform)
    #     with np.printoptions(precision=2, suppress=True):
    #         logging.info(f'---Voting---\nBest Epoch {best_epoch},'
    #                     f'Voting Instance mIoU {test_ins_miou:.2f}, '
    #                     f'Voting Class mIoU {test_cls_miou:.2f}, '
    #                     f'\n Voting Class mIoUs {test_cls_mious}')

        # if writer is not None:
        #     writer.add_scalar('test_ins_miou_voting', test_ins_miou, epoch)
        #     writer.add_scalar('test_class_miou_voting', test_cls_miou, epoch)
    torch.cuda.synchronize()
    # if writer is not None:
    #     writer.close()
    #dist.destroy_process_group()
    wandb.finish(exit_code=True)


def train_one_epoch(model, train_loader, globalcriterion, localcriterion, finalcriterion, optimizer, scheduler, epoch, cfg):

    loss_meter = AverageMeter()
    model.train()  # set model to training mode

    # # fine-tuning
    # # freeze_layer = ["encoder","decoder"]
    # freeze_layer = ["encoder.0","encoder.1","encoder.2","encoder.3"]
    # for k, v in model.named_parameters():
    #     if any(x in k for x in freeze_layer):
    #         #print('freezing {}'.format(k))
    #         v.requires_grad = False
    #
    # print("*" * 50)
    # for k, v in model.named_parameters():
    #     if v.requires_grad:
    #         print("可训练的模型参数名称 {}".format(k))
    #     else:
    #         print("已被冻结的模型参数名称 {}".format(k))
    # print("*" * 50)

    paramcout = count_param(model)
    logging.info('Number of Network Params: %.4f M' % (paramcout / 1e6))
    print("网络参数量:", paramcout)

    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, (data, data2, xyzdata) in pbar:
        # data：变换后的data1, data2:变换后的data2, xyzdata：原始坐标
       # oridataxyz = data['pos']
        # data1 = copy.deepcopy(data)
        #
        # for key in data.keys():
        #     if "1" in key:
        #         data1[key[:-1]] = data[key]

        # data1 = convertmultiview(data)
        # data2 = convertmultiview(data)

        # data1 = transformerposition(data1)
        # data2 = transformerposition(data2)

        #print(data.keys())
        num_iter += 1
        batch_size, num_point, _ = data['pos'].size()
        #print(type(data['cls']))
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        for key in data2.keys():
            data2[key] = data2[key].cuda(non_blocking=True)


        target = data['y']
        # data['x'] = get_features_by_keys(data, cfg.feature_keys)
        # data2['x'] = get_features_by_keys(data2, cfg.feature_keys)
        p0first = data['pos']   # [8,4096,3]
        p0sec = data2['pos']    # [8,4096,3]

        '''Spotr、Pointnext模型从数据变化到输入使用以下代码，剩下代码全部注释'''
        # data['x'] = get_features_by_keys(data, cfg.feature_keys)
        # data2['x'] = get_features_by_keys(data2, cfg.feature_keys)
        # logits = model(data)  # 直接按键值查找data，包括xyz和法向量
        # logits2 = model(data2)  # 直接按键值查找data，包括xyz和法向量

        '''第一个分支的数据变化'''
        data['pos'] = data['pos'].permute(0, 2, 1)
        data['x'] = data['x'].permute(0, 2, 1)
        data_all = torch.cat((data['pos'], data['x']), dim=1)  # 在第二个维度上进行拼接
        label_one_hot = np.zeros((data['cls'].shape[0], 1))
        for idx in range(data['cls'].shape[0]):
            label_one_hot[idx, data['cls'][idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        label_one_hot = label_one_hot.cuda()    # [8,1]

        '''第二个分支的数据变化'''
        data2['pos'] = data2['pos'].permute(0, 2, 1)
        data2['x'] = data2['x'].permute(0, 2, 1)
        data_all2 = torch.cat((data2['pos'], data2['x']), dim=1)  # 在第二个维度上进行拼接
        label_one_hot2 = np.zeros((data2['cls'].shape[0], 1))
        for idx in range(data2['cls'].shape[0]):
            label_one_hot2[idx, data2['cls'][idx]] = 1
        label_one_hot2 = torch.from_numpy(label_one_hot2.astype(np.float32))
        label_one_hot2 = label_one_hot2.cuda()  # [8,1]

        '''Curvenet时用的'''
        # logits = model(data['x'], label_one_hot)  # [8,128,4096]
        # logits2 = model(data2['x'], label_one_hot2)  # [8,128,4096]
        '''DGCNN、Pointnet、Pointnet2、PCT时用的'''
        logits = model(data_all, label_one_hot)  # [8,128,4096]
        logits2 = model(data_all2, label_one_hot2)  # [8,128,4096]
        '''GDANet、PointMLP使用'''
        # logits = model(data['pos'], data['x'], label_one_hot)
        # logits2 = model(data2['pos'], data2['x'], label_one_hot2)
        '''Pointnext时用的'''
        # logits,_ , _, p0first = model(data)
        # logits2, _, _, p0sec = model(data2)

        '''MY-MODEL时用的'''
        # logits = model(data_all)        # [8,6,4096]-->[8,128,4096]
        # logits2 = model(data_all2)     # [8,6,4096]-->[8,128,4096]

        #互相关矩阵LOSS   global loss
       # logits, logits2 = myfps(oridataxyz.cuda(),logits.permute(0,2,1),logits2.permute(0,2,1),512) #最远点下采样
        logits, logits2 = logits.permute(0,2,1),logits2.permute(0,2,1)

        # p0first=[8,4096,3],p0sec=[8,4096,3]
        # coef = calcorrcoef(logits,logits2)
        loss = globalcriterion(logits,logits2,p0first,p0sec,xyzdata['pos'])

        #Chamfer Loss   local loss
        # sa3first = sa3first.permute(0, 2, 1)
        # sa3sec = sa3sec.permute(0, 2, 1)
        #
        # sa1first = sa1first.permute(0, 2, 1)
        # sa1sec = sa1sec.permute(0, 2, 1)

        #localloss1 = localcriterion(sa1first,sa1sec)

        #localloss1 = localcriterion(sa1first, sa1sec)
        #localloss2 = localcriterion(sa3first, sa3sec)

        #loss = finalcriterion([globalloss, globalpointloss, crossgroupinnerloss, reggroupinnerloss])
           # localgroupdisloss = 0

        #loss = finalcriterion(goballoss,localloss1,localloss2)
        #loss = finalcriterion([goballoss, localloss1, localloss2])
        #loss = finalcriterion([goballoss, localloss2,localgroupdisloss, groupdisloss, groupinnerloss])

        #logging.info(f"globalloss:{globalloss}  globalpointloss:{globalpointloss}  crossgroupinnerloss:{crossgroupinnerloss}  reggroupinnerloss:{reggroupinnerloss}")


        # if cfg.criterion_args.NAME != 'MultiShapeCrossEntropy':
        #     loss = criterion(logits, target)
        # else:
        #     loss = criterion(logits, target, data['cls'])

        # # 异常检测开启
        # torch.autograd.set_detect_anomaly(True)
        # # 反向传播时检测是否有异常值，定位code
        # with torch.autograd.detect_anomaly():
        #     loss.backward()

        loss.backward()

        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
            num_iter = 0
            optimizer.step()
            optimizer.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        loss_meter.update(loss.item(), n=batch_size)
        if idx % cfg.print_freq:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.avg:.3f} ")  #loss_meter.avg

        record_iter_train['loss'].append(loss_meter.avg)  # loss_meter.avg
        record_iter_train['epoch'].append(epoch)

    train_loss = loss_meter.avg
    return train_loss


def train_one_epoch_pointstak(model, train_loader, globalcriterion, localcriterion, finalcriterion, optimizer, scheduler, epoch, cfg):
    loss_meter = AverageMeter()
    model.train()  # set model to training mode

    # # fine-tuning
    # # freeze_layer = ["encoder","decoder"]
    # freeze_layer = ["encoder.0","encoder.1","encoder.2","encoder.3"]
    # for k, v in model.named_parameters():
    #     if any(x in k for x in freeze_layer):
    #         #print('freezing {}'.format(k))
    #         v.requires_grad = False
    #
    # print("*" * 50)
    # for k, v in model.named_parameters():
    #     if v.requires_grad:
    #         print("可训练的模型参数名称 {}".format(k))
    #     else:
    #         print("已被冻结的模型参数名称 {}".format(k))
    # print("*" * 50)

    paramcout = count_param(model)
    logging.info('Number of Network Params: %.4f M' % (paramcout / 1e6))
    print("网络参数量:", paramcout)

    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, (data, data2, xyzdata) in pbar:
        # data：变换后的data1, data2:变换后的data2, xyzdata：原始坐标
        # oridataxyz = data['pos']
        # data1 = copy.deepcopy(data)
        #
        # for key in data.keys():
        #     if "1" in key:
        #         data1[key[:-1]] = data[key]

        # data1 = convertmultiview(data)
        # data2 = convertmultiview(data)

        # data1 = transformerposition(data1)
        # data2 = transformerposition(data2)

        # print(data.keys())
        num_iter += 1
        batch_size, num_point, _ = data['points'].size()
        # print(type(data['cls']))
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        for key in data2.keys():
            data2[key] = data2[key].cuda(non_blocking=True)

        target = data['seg_id']
        # data['x'] = get_features_by_keys(data, cfg.feature_keys)
        # data2['x'] = get_features_by_keys(data2, cfg.feature_keys)
        p0first = data['points']  # [8,4096,3]
        p0sec = data2['points']  # [8,4096,3]

        '''第一个分支的数据变化'''
        data['points'] = data['points']
        data['norms'] = data['norms']
        data_all = torch.cat((data['points'], data['norms']), dim=1)  # 在第二个维度上进行拼接
        label_one_hot = np.zeros((data['cls_tokens'].shape[0], 1))
        for idx in range(data['cls_tokens'].shape[0]):
            label_one_hot[idx, data['cls_tokens'][idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        label_one_hot = label_one_hot.cuda()  # [8,1]

        '''第二个分支的数据变化'''
        data2['points'] = data2['points']
        data2['norms'] = data2['norms']
        data_all2 = torch.cat((data2['points'], data2['norms']), dim=1)  # 在第二个维度上进行拼接
        label_one_hot2 = np.zeros((data2['cls_tokens'].shape[0], 1))
        for idx in range(data2['cls_tokens'].shape[0]):
            label_one_hot2[idx, data2['cls_tokens'][idx]] = 1
        label_one_hot2 = torch.from_numpy(label_one_hot2.astype(np.float32))
        label_one_hot2 = label_one_hot2.cuda()  # [8,1]

        logits = model(data)    # 直接按键值查找data，包括xyz和法向量
        logits2 = model(data2)  # [8,128,4096]


        # 互相关矩阵LOSS   global loss
        # logits, logits2 = myfps(oridataxyz.cuda(),logits.permute(0,2,1),logits2.permute(0,2,1),512) #最远点下采样
        logits, logits2 = logits.permute(0, 2, 1), logits2.permute(0, 2, 1)

        # p0first=[8,4096,3],p0sec=[8,4096,3]
        # coef = calcorrcoef(logits,logits2)
        loss = globalcriterion(logits, logits2, p0first, p0sec, xyzdata['points'])

        # Chamfer Loss   local loss
        # sa3first = sa3first.permute(0, 2, 1)
        # sa3sec = sa3sec.permute(0, 2, 1)
        #
        # sa1first = sa1first.permute(0, 2, 1)
        # sa1sec = sa1sec.permute(0, 2, 1)

        # localloss1 = localcriterion(sa1first,sa1sec)

        # localloss1 = localcriterion(sa1first, sa1sec)
        # localloss2 = localcriterion(sa3first, sa3sec)

        # loss = finalcriterion([globalloss, globalpointloss, crossgroupinnerloss, reggroupinnerloss])
        # localgroupdisloss = 0

        # loss = finalcriterion(goballoss,localloss1,localloss2)
        # loss = finalcriterion([goballoss, localloss1, localloss2])
        # loss = finalcriterion([goballoss, localloss2,localgroupdisloss, groupdisloss, groupinnerloss])

        # logging.info(f"globalloss:{globalloss}  globalpointloss:{globalpointloss}  crossgroupinnerloss:{crossgroupinnerloss}  reggroupinnerloss:{reggroupinnerloss}")

        # if cfg.criterion_args.NAME != 'MultiShapeCrossEntropy':
        #     loss = criterion(logits, target)
        # else:
        #     loss = criterion(logits, target, data['cls'])

        # # 异常检测开启
        # torch.autograd.set_detect_anomaly(True)
        # # 反向传播时检测是否有异常值，定位code
        # with torch.autograd.detect_anomaly():
        #     loss.backward()

        loss.backward()

        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
            num_iter = 0
            optimizer.step()
            optimizer.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        loss_meter.update(loss.item(), n=batch_size)
        if idx % cfg.print_freq:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.avg:.3f} ")  # loss_meter.avg

        record_iter_train['loss'].append(loss_meter.avg)  # loss_meter.avg
        record_iter_train['epoch'].append(epoch)

    train_loss = loss_meter.avg
    return train_loss


@torch.no_grad()
def validate(model, val_loader, cfg, num_votes=0, data_transform=None):
    model.eval()  # set model to eval mode
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    cls_mious = torch.zeros(cfg.shape_classes, dtype=torch.float32).cuda(non_blocking=True)
    cls_nums = torch.zeros(cfg.shape_classes, dtype=torch.int32).cuda(non_blocking=True)
    ins_miou_list = []

    # label_size: b, means each sample has one corresponding class
    for idx, data in pbar:

        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']
        cls = data['cls']
        data['x'] = get_features_by_keys(data, cfg.feature_keys)
        batch_size, num_point, _ = data['pos'].size()

        '''pointMLP'''
        data['pos'] = data['pos'].transpose(2, 1)
        data['x'] = data['x'].transpose(2, 1)
        label_one_hot = np.zeros((data['cls'].shape[0], 1))
        for idx in range(data['cls'].shape[0]):
            label_one_hot[idx, data['cls'][idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        label_one_hot = label_one_hot.cuda()
        '''pointMLP结束'''

        logits = 0
        for v in range(num_votes+1):
            set_random_seed(v)
            if v > 0:
                data['pos'] = data_transform(data['pos'])
            '''PointMLP时用的'''
            # logits += model(data)
            '''MY_MODEL时使用'''
            logits += model(data['pos'], data['x'], label_one_hot)
        logits /= (num_votes + 1)
        preds = logits.max(dim=1)[1]   #preds即为所求类别
        #print("preds",preds.shape)
        #print(preds)
        if cfg.get('refine', False):
            part_seg_refinement(preds, data['pos'], data['cls'], cfg.cls2parts, cfg.get('refine_n', 10))

        if cfg.criterion_args.NAME != 'MultiShapeCrossEntropy':
            batch_ins_mious = get_ins_mious(preds, target, data['cls'], cfg.cls2parts)
            ins_miou_list += batch_ins_mious
        else:
            iou_array = []
            for ib in range(batch_size):
                sl = data['cls'][ib][0]
                iou = get_ins_mious(preds[ib:ib + 1], target[ib:ib + 1], sl.unsqueeze(0), cfg.cls2parts,
                                    multihead=True)
                iou_array.append(iou)
            ins_miou_list += iou_array

        # per category iou at each batch_size:
        for shape_idx in range(batch_size):  # sample_idx
            cur_gt_label = int(cls[shape_idx].cpu().numpy())
            # add the iou belongs to this cat
            cls_mious[cur_gt_label] += batch_ins_mious[shape_idx]
            cls_nums[cur_gt_label] += 1

    ins_mious_sum, count = torch.sum(torch.stack(ins_miou_list)), torch.tensor(len(ins_miou_list)).cuda()
    if cfg.distributed:
        dist.all_reduce(cls_mious), dist.all_reduce(cls_nums), dist.all_reduce(ins_mious_sum), dist.all_reduce(count)

    for cat_idx in range(cfg.shape_classes):
        # indicating this cat is included during previous iou appending
        if cls_nums[cat_idx] > 0:
            cls_mious[cat_idx] = cls_mious[cat_idx] / cls_nums[cat_idx]

    ins_miou = ins_mious_sum/count
    cls_miou = torch.mean(cls_mious)
    with np.printoptions(precision=2, suppress=True):
        logging.info(f'Test Epoch [{cfg.epoch}/{cfg.epochs}],'
                        f'Instance mIoU {ins_miou:.2f}, '
                        f'Class mIoU {cls_miou:.2f}, '
                        f'\n Class mIoUs {cls_mious}')
    return ins_miou, cls_miou, cls_mious


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser('ShapeNetPart Part segmentation training')
    #  pointnet_default.yaml  pointnet2_default.yaml（point_transformer用这个）    dgcnn.yaml    GDANet_default.yaml
    #  pointmlp_default.yaml  curvenet_default.yaml    pointstack_default.yaml   spotr_default.yaml  pointnext_default.yaml
    parser.add_argument('--cfg', type=str, default="../../cfgs/shapenetpart/dgcnn.yaml", help='config file')
    args, opts = parser.parse_known_args()
   #args.cfg = "D:/pythonproject/Pointnext/cfgs/shapenetpart/default.yaml"
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)
    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1
    #print("cfg.distributed",cfg.distributed)
    # logger
    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]  # task/dataset name, \eg s3dis, modelnet40_cls
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]  # cfg_basename, \eg pointnext-xl
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.cfg_basename,  # cfg file name
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    opt_list = [] # for checking experiment configs from logging file
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            opt_list.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.opts = '-'.join(opt_list)

    cfg.is_training = cfg.mode not in ['test', 'testing', 'val', 'eval', 'evaluation']

    if cfg.mode in ['resume', 'test', 'val']:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = [cfg.mode]
    else:
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
        cfg.wandb.tags = tags
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path

    # wandb config
    cfg.wandb.name = cfg.run_name


    main(0, cfg)
    # multi processing.
    # if cfg.mp:
    #     port = find_free_port()
    #     cfg.dist_url = f"tcp://localhost:{port}"
    #     print('using mp spawn for distributed training')
    #     mp.spawn(main, nprocs=cfg.world_size, args=(cfg,))
    # else:
    #     main(0, cfg)


