"""
Distributed training script for scene segmentation with S3DIS dataset
"""
import argparse
import copy
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

from openpoints.models.pointmlp.pointMLP import pointMLP
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

from datautils.SelfSuperviseLoss import  SegmentLoss

global record_iter_train
record_iter_train = {'epoch':[], 'loss':[]}
global record_iter_val
record_iter_val = {'epoch':[], 'loss':[]}
global record_epoch
record_epoch = {'epoch':[], 'train':[],'val':[]}
global record_all
record_all = {'epoch':[], 'IoU spike': [], 'IoU leaf': [], 'IoU': [], 'test acc': [],
              'spike test pre': [], 'leaf test pre': [], 'test pre': [],
              'spike test recall': [], 'leaf test recall': [], 'test recall': [] }

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


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda(non_blocking=True)
    return new_y


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)

# 每个类的实例分割的iou
def get_part_mious(pred, target, cls, cls2parts, multihead=False,):
    """Get the Shape IoU
    shape IoU: the mean part iou for each shape
    Args:
        pred (_type_): _description_
        target (_type_): _description_
        num_classes (_type_): _description_
    Returns:
        _type_: _description_
    """
    # pred = （8，4096）, target = （8，4096）, cls = （8，1）, cls2parts = 类别数（0，1）
    ins_mious = []
    batchpart_iou = []
    for shape_idx in range(pred.shape[0]):  # sample_idx
        part_ious = []
        mypart_iou = []
        parts = cls2parts[cls[shape_idx]]   # parts = [0,1]
        if multihead:
            parts = np.arange(len(parts))

        for part in parts:
            pred_part = pred[shape_idx] == part                   # 正确预测为0（1）的值
            target_part = target[shape_idx] == part               # 真实为0（1）的值
            I = torch.logical_and(pred_part, target_part).sum()   # 真实和预测都为0（1）的值
            U = torch.logical_or(pred_part, target_part).sum()    # 真实或预测都为0（1）的值
            if U == 0:
                iou = torch.tensor(100, device=pred.device, dtype=torch.float32)
            else:
                iou = I * 100 / float(U)
            part_ious.append(iou)
            mypart_iou.append(iou.cpu().float())  #自加  [0,1]
        batchpart_iou.append(mypart_iou)          #   [0:[0,1],1:[0,1],1:[0,1].......,,batch:[0,1]]
        ins_mious.append(torch.mean(torch.stack(part_ious)))
    batchpart_iou = np.array(batchpart_iou)       # [8,2]
    batchpart_iou = np.mean(batchpart_iou,axis=0) # [1,2]   求均值
    #print(f"batchpart_iou{batchpart_iou}")

    return ins_mious,batchpart_iou


# 每个类的实例分割的acc
def get_part_acc(pred, target, cls, cls2parts, multihead=False,):
    """Get the Shape IoU
    shape IoU: the mean part iou for each shape
    Args:
        pred (_type_): _description_
        target (_type_): _description_
        num_classes (_type_): _description_
    Returns:
        _type_: _description_
    """
    # pred = （8，4096）, target = （8，4096）, cls = （8，1）, cls2parts = 类别数（0，1）
    ins_mious = []
    batchpart_iou = []
    for shape_idx in range(pred.shape[0]):  # sample_idx
        part_ious = []
        mypart_iou = []
        parts = cls2parts[cls[shape_idx]]   # parts = [0,1]
        if multihead:
            parts = np.arange(len(parts))
        for part in parts:
            pred_part = pred[shape_idx] == part                        # 正确预测为0的值
            target_part = target[shape_idx] == part                    # 真实为0的值
            pred_part_1 = pred[shape_idx] != part                      # 正确预测为1的值
            target_part_1 = target[shape_idx] != part                  # 真实为1的值
            I1 = torch.logical_and(pred_part, target_part).sum()       # 真实和预测都为0的值
            I2 = torch.logical_and(pred_part_1, target_part_1).sum()   # 真实和预测都为1的值
            I = I1 + I2
            U1 = target_part.sum()                                     # 真实或预测都为0的值
            U2 = target_part_1.sum()                                   # 真实或预测都为1的值# 真实或预测都为1的值
            U = U1 + U2
            if U == 0:
                iou = torch.tensor(100, device=pred.device, dtype=torch.float32)
            else:
                iou = I * 100 / float(U)
            part_ious.append(iou)
            mypart_iou.append(iou.cpu().float())  #自加  [0,1]

        batchpart_iou.append(mypart_iou)          #   [0:[0,1],1:[0,1],1:[0,1].......,,batch:[0,1]]
        ins_mious.append(torch.mean(torch.stack(part_ious)))
    batchpart_iou = np.array(batchpart_iou)       # [8,2]
    batchpart_iou = np.mean(batchpart_iou,axis=0) # [1,2]
    #print(f"batchpart_iou{batchpart_iou}")

    return ins_mious,batchpart_iou


# 每个类的实例分割的pre
def get_part_pre(pred, target, cls, cls2parts, multihead=False,):
    """Get the Shape IoU
    shape IoU: the mean part iou for each shape
    Args:
        pred (_type_): _description_
        target (_type_): _description_
        num_classes (_type_): _description_
    Returns:
        _type_: _description_
    """
    # pred = （8，4096）, target = （8，4096）, cls = （8，1）, cls2parts = 类别数（0，1）
    ins_mious = []
    batchpart_iou = []
    for shape_idx in range(pred.shape[0]):  # sample_idx
        part_ious = []
        mypart_iou = []
        parts = cls2parts[cls[shape_idx]]   # parts = [0,1]
        if multihead:
            parts = np.arange(len(parts))
        for part in parts:
            pred_part = pred[shape_idx] == part                   # 正确预测为0（1）的值
            target_part = target[shape_idx] == part               # 真实为0（1）的值
            I = torch.logical_and(pred_part, target_part).sum()   # 真实和预测都为0（1）的值
            U = pred_part.sum()    # 真实或预测都为0（1）的值
            if U == 0:
                iou = torch.tensor(100, device=pred.device, dtype=torch.float32)
            else:
                iou = I * 100 / float(U)
            part_ious.append(iou)
            mypart_iou.append(iou.cpu().float())  #自加  [0,1]

        batchpart_iou.append(mypart_iou)          #   [0:[0,1],1:[0,1],1:[0,1].......,,batch:[0,1]]
        ins_mious.append(torch.mean(torch.stack(part_ious)))
    batchpart_iou = np.array(batchpart_iou)       # [8,2]
    batchpart_iou = np.mean(batchpart_iou,axis=0) # [1,2]
    #print(f"batchpart_iou{batchpart_iou}")

    return ins_mious,batchpart_iou


# 每个类的实例分割的recall
def get_part_recall(pred, target, cls, cls2parts, multihead=False,):
    """Get the Shape IoU
    shape IoU: the mean part iou for each shape
    Args:
        pred (_type_): _description_
        target (_type_): _description_
        num_classes (_type_): _description_
    Returns:
        _type_: _description_
    """
    # pred = （8，4096）, target = （8，4096）, cls = （8，1）, cls2parts = 类别数（0，1）
    ins_mious = []
    batchpart_iou = []
    for shape_idx in range(pred.shape[0]):  # sample_idx
        part_ious = []
        mypart_iou = []
        parts = cls2parts[cls[shape_idx]]   # parts = [0,1]
        if multihead:
            parts = np.arange(len(parts))

        for part in parts:
            pred_part = pred[shape_idx] == part                   # 正确预测为0（1）的值
            target_part = target[shape_idx] == part               # 真实为0（1）的值
            I = torch.logical_and(pred_part, target_part).sum()   # 真实和预测都为0（1）的值
            U = target_part.sum()    # 真实或预测都为0（1）的值
            if U == 0:
                iou = torch.tensor(100, device=pred.device, dtype=torch.float32)
            else:
                iou = I * 100 / float(U)
            part_ious.append(iou)
            mypart_iou.append(iou.cpu().float())  #自加  [0,1]

        batchpart_iou.append(mypart_iou)          #   [0:[0,1],1:[0,1],1:[0,1].......,,batch:[0,1]]
        ins_mious.append(torch.mean(torch.stack(part_ious)))
    batchpart_iou = np.array(batchpart_iou)       # [8,2]
    batchpart_iou = np.mean(batchpart_iou,axis=0) # [1,2]
    #print(f"batchpart_iou{batchpart_iou}")

    return ins_mious,batchpart_iou


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
    # pred = [8, 4096], target = [8, 4096], cls = [8, 1], cls2parts = [0, 1]
    ins_mious = []
    for shape_idx in range(pred.shape[0]):  # sample_idx
        part_ious = []
        parts = cls2parts[cls[shape_idx]]
        if multihead:
            parts = np.arange(len(parts))

        for part in parts:
            pred_part = pred[shape_idx] == part          # 找出预测值pred中每个batch中为叶（0）或穗（1）的点
            target_part = target[shape_idx] == part      # 找出真实值target中每个batch中为叶（0）或穗（1）的点
            I = torch.logical_and(pred_part, target_part).sum()         # 找出真实值和预测值都为true的点的总数
            U = torch.logical_or(pred_part, target_part).sum()          # 找出真实值或预测值都为true的点的总数
            if U == 0:
                iou = torch.tensor(100, device=pred.device, dtype=torch.float32)
            else:
                iou = I * 100 / float(U)         # iou = （I的点数 * 100）/ U的点数
            part_ious.append(iou)

        ins_mious.append(torch.mean(torch.stack(part_ious)))

    return ins_mious


def main(gpu, cfg):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url, world_size=cfg.world_size, rank=cfg.rank)
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
   # print(type(cfg.dataset))
    logging.info(cfg)

    # build dataset
    val_loader = build_dataloader_from_cfg(cfg.batch_size, cfg.dataset, cfg.dataloader, datatransforms_cfg=cfg.datatransforms, split='val', distributed=cfg.distributed)
    test_loader = build_dataloader_from_cfg(cfg.batch_size, cfg.dataset, cfg.dataloader, datatransforms_cfg=cfg.datatransforms, split='test', distributed=cfg.distributed)
    train_loader = build_dataloader_from_cfg(cfg.batch_size, cfg.dataset, cfg.dataloader, datatransforms_cfg=cfg.datatransforms, split='train', distributed=cfg.distributed,)
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")
    logging.info(f"length of test dataset: {len(test_loader.dataset)}")
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

    #model = build_model_from_cfg(cfg.model).cuda()
    device = torch.device('cuda:0')
    model = pointMLP(2,4096).to(device)  #partnums, points
    model.apply(weight_init)

    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))

    # #加载自监督参数 (在selfprepath和temppath中任选一种 即可)
    '''方法一'''
    try:
        selfprepath = r"/data_1/home/whs/Pointnext/examples/shapenetpart/log/shapenetpart/rice_unsupervised/rice_unsupervised_pointmlp/checkpoint/_ckpt_latest.pth"
        logging.info('load self-pretrained params')
        current_weights = model.state_dict()  # 参看当前模型的参数模型
        load_checkpoint(model, pretrained_path=selfprepath)
        pretrained_model = torch.load(selfprepath)
        # cfg.start_epoch = pretrained_model['epoch']  # 加载当前的预训练epoch
        logging.info('**************************Use pretrain model**************************')
    except:
        logging.info('No existing model, starting training from scratch...')


    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
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


    if cfg.get('cls_weighed_loss', False):
        if hasattr(train_loader.dataset, 'num_per_class'):
            cfg.criterion_args.weight = None
            cfg.criterion_args.weight = get_class_weights(train_loader.dataset.num_per_class, normalize=True)
        else:
            logging.info('`num_per_class` attribute is not founded in dataset')
    criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
    segcriterion = SegmentLoss().cuda()
    # ===> start training
    best_ins_miou, cls_miou_when_best, cls_mious_when_best = 0., 0., []

    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        # some dataset sets the dataset length as a fixed steps.
        if hasattr(train_loader.dataset, 'epoch'):
            train_loader.dataset.epoch = epoch - 1
        cfg.epoch = epoch                       # 训练
        train_loss = \
            train_one_epoch(model, train_loader, criterion,segcriterion, optimizer, scheduler, epoch, cfg)

        record_epoch['epoch'].append(epoch)
        record_epoch['train'].append(train_loss)  # 每次epoch的loss
        record_all['epoch'].append(epoch)

        is_best = False
        if epoch % cfg.val_freq == 0:     # 验证
            val_ins_miou, val_cls_miou, val_cls_mious = validate_fn(model, val_loader, cfg, epoch, criterion)  ###############第6步  加criterion
            if val_ins_miou > best_ins_miou:
                best_ins_miou = val_ins_miou
                cls_miou_when_best = val_cls_miou
                cls_mious_when_best = val_cls_mious
                best_epoch = epoch
                is_best = True
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Find a better ckpt @E{epoch}, val_ins_miou {best_ins_miou:.2f} val_cls_miou {cls_miou_when_best:.2f}, '
                        f'\ncls_mious: {cls_mious_when_best}')

        lr = optimizer.param_groups[0]['lr']
        if writer is not None:
            writer.add_scalar('val_ins_miou', val_ins_miou, epoch)
            writer.add_scalar('val_class_miou', val_cls_miou, epoch)
            writer.add_scalar('best_val_instance_miou', best_ins_miou, epoch)
            writer.add_scalar('val_class_miou_when_best', cls_miou_when_best, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('lr', lr, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)

        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'ins_miou': best_ins_miou,
                                             'cls_miou': cls_miou_when_best}, is_best=is_best)

        # 保存为Excel文件
        model_name = cfg.cfg_basename.split('_')[0]
        # save_path = 'log/pictures/' + model_name  # 指定
        save_path = r"log/pictures/pointmlp/rice_pointmlp_super_80/"
        if not os.path.exists(save_path):  # 如果文件夹不存在，则创建它
            os.makedirs(save_path)
        df = pd.DataFrame(record_all)
        save_path_1 = save_path + '/' + 'record_all.xlsx'  # 指定文件夹路径
        df.to_excel(save_path_1, index=False)

        df1 = pd.DataFrame(record_iter_train)
        save_path_2 = save_path + '/' + 'record_iter_train.xlsx'  # 指定文件夹路径
        df1.to_excel(save_path_2, index=False)

        df2 = pd.DataFrame(record_iter_val)
        save_path_3 = save_path + '/' + 'record_iter_val.xlsx'  # 指定文件夹路径
        df2.to_excel(save_path_3, index=False)

        df3 = pd.DataFrame(record_epoch)
        save_path_4 = save_path + '/' + 'record_epoch.xlsx'  # 指定文件夹路径
        df3.to_excel(save_path_4, index=False)

    # if writer is not None:
    #     Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    # Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.logname}_ckpt_latest.pth'))
    with np.printoptions(precision=2, suppress=True):
        logging.info(f'Best Epoch {best_epoch},'
                     f'Instance mIoU {best_ins_miou:.2f}, '
                     f'Class mIoU {cls_miou_when_best:.2f}, '
                     f'\n Class mIoUs {cls_mious_when_best}')

    if cfg.get('num_votes', 0) > 0:
        # pretrained_path = r"/data/home/whs/Pointnext/examples/shapenetpart/log/shapenetpart/pointmlp_wheat_90/checkpoint/ckpt_best.pth"
        # load_checkpoint(model, pretrained_path)
        load_checkpoint(model, pretrained_path=os.path.join(cfg.ckpt_dir, 'ckpt_best.pth'))
        print(load_checkpoint)
        set_random_seed(cfg.seed)   # 测试
        # test_ins_miou, test_cls_miou, test_part_iou, test_cls_mious = testmetric(model, test_loader, cfg,num_votes=cfg.get('num_votes', 0),data_transform=voting_transform)

        test_ins_miou, test_cls_miou, test_part_iou, test_part_acc, test_part_pre, test_part_recall, test_cls_mious  = testmetric(model, test_loader, cfg, num_votes=cfg.get('num_votes', 0), data_transform=voting_transform)
        with np.printoptions(precision=2, suppress=True):
            logging.info(f'---Voting---\nBest Epoch {best_epoch},'
                         f''f'Voting Instance mIoU {test_ins_miou:.2f}, '
                         f''f'Voting Class mIoU {test_cls_miou:.2f}, '
                         # f''f'Voting Class Part_IoU {test_part_iou:.2f}, '
                         # f''f'Voting Class Accuracy {test_part_acc:.2f}, '
                         # f''f'Voting Class Precision {test_part_pre:.2f}, '
                         # f''f'Voting Class Recall {test_part_recall:.2f}, '
                         f''f'\n Voting Class mIoUs {test_cls_mious}')

        # if writer is not None:
        #     writer.add_scalar('test_ins_miou_voting', test_ins_miou, epoch)
        #     writer.add_scalar('test_class_miou_voting', test_cls_miou, epoch)
    torch.cuda.synchronize()
    if writer is not None:
        writer.close()

    # # 保存为Excel文件
    # model_name = cfg.cfg_basename.split('_')[0]
    # save_path = 'log/pictures/' + model_name     # 指定
    # if not os.path.exists(save_path):  # 如果文件夹不存在，则创建它
    #     os.makedirs(save_path)
    # df = pd.DataFrame(record_all)
    # save_path_1 = save_path + '/' + 'record_all.xlsx'  # 指定文件夹路径
    # df.to_excel(save_path_1, index=False)
    #
    # df1 = pd.DataFrame(record_iter_train)
    # save_path_2 = save_path + '/' + 'record_iter_train.xlsx'  # 指定文件夹路径
    # df1.to_excel(save_path_2, index=False)
    #
    # df2 = pd.DataFrame(record_iter_val)
    # save_path_3 = save_path + '/' + 'record_iter_val.xlsx'  # 指定文件夹路径
    # df2.to_excel(save_path_3, index=False)
    #
    # df3 = pd.DataFrame(record_epoch)
    # save_path_4 = save_path + '/' + 'record_epoch.xlsx'  # 指定文件夹路径
    # df3.to_excel(save_path_4, index=False)

    # 绘制loss的折线图并保存到指定文件夹
    fig, ax = plt.subplots()
    ax.plot(record_iter_val['loss'], label='iter_loss_val')
    ax.plot(record_iter_train['loss'], label='iter_loss_train')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    # 设置x轴和y轴范围及刻度
    plt.xlim(0, epoch + 1)
    plt.xticks(np.arange(0, epoch + 1, 3))
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
    ax.plot(record_epoch['val'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    # 设置x轴和y轴范围及刻度
    plt.xlim(0, epoch + 1)
    plt.xticks(np.arange(0, epoch + 1, 3))
    max_y1 = math.ceil(max(record_epoch['val']))
    if max_y1 <= 1:
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, max_y1, 0.1))
    else:
        plt.ylim(0, max_y1)
        plt.yticks(np.arange(0, max_y1, max_y1 / 10))
    ax.legend()
    fig.savefig(os.path.join(save_path, 'Loss.png'))
    # plt.show()

    # 绘制iou的折线图并保存到指定文件夹
    fig, ax = plt.subplots()
    ax.plot(record_all['IoU spike'], label='IoU of spike')
    ax.plot(record_all['IoU leaf'], label='IoU of leaf')
    ax.plot(record_all['IoU'], label='IoU')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test IoU')
    # 设置x轴和y轴范围及刻度
    plt.xlim(0, epoch + 1)
    plt.xticks(np.arange(0, epoch + 1, 3))
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1, 0.1))
    ax.legend()
    fig.savefig(os.path.join(save_path, 'IoU.png'))
    # plt.show()

    # 绘制准确度的折线图并保存到指定文件夹
    fig, ax = plt.subplots()
    ax.plot(record_all['test acc'], label='test acc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test accuracy')
    # 设置x轴和y轴范围及刻度
    plt.xlim(0, epoch + 1)
    plt.xticks(np.arange(0, epoch + 1, 3))
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1, 0.1))
    ax.legend()
    fig.savefig(os.path.join(save_path, 'Accuracy.png'))
    # plt.show()

    # 绘制精度的折线图并保存到指定文件夹
    fig, ax = plt.subplots()
    ax.plot(record_all['spike test pre'], label='spike test precision')
    ax.plot(record_all['leaf test pre'], label='leaf test precision')
    ax.plot(record_all['test pre'], label='test precision')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test precision')
    # 设置x轴和y轴范围及刻度
    plt.xlim(0, epoch + 1)
    plt.xticks(np.arange(0, epoch + 1, 3))
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1, 0.1))
    ax.legend()
    fig.savefig(os.path.join(save_path, 'Precision.png'))
    # plt.show()

    # 绘制recall的折线图并保存到指定文件夹
    fig, ax = plt.subplots()
    ax.plot(record_all['spike test recall'], label='spike test recall')
    ax.plot(record_all['leaf test recall'], label='leaf test recall')
    ax.plot(record_all['test recall'], label='test recall')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test recall')
    # 设置x轴和y轴范围及刻度
    plt.xlim(0, epoch + 1)
    plt.xticks(np.arange(0, epoch + 1, 3))
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1, 0.1))
    ax.legend()
    fig.savefig(os.path.join(save_path, 'Recall.png'))
    # plt.show()

    #dist.destroy_process_group()
    wandb.finish(exit_code=True)


def train_one_epoch(model, train_loader, criterion, segcriterion, optimizer, scheduler, epoch, cfg):

    loss_meter = AverageMeter()
    model.train()  # set model to training mode

    #fine-tuning
    #freeze_layer = ["encoder","decoder"]
    #freeze_layer = ["encoder.0","encoder.1","encoder.2","encoder.3"]
    # for k, v in model.named_parameters():
    #     if any(x in k for x in freeze_layer):
    #         #print('freezing {}'.format(k))
    #         v.requires_grad = False


    # print("*" * 50)
    # for k, v in model.named_parameters():
    #     if v.requires_grad:
    #         print("可训练的模型参数名称 {}".format(k))
    #     else:
    #         print("已被冻结的模型参数名称 {}".format(k))
    # print("*" * 50)

    paramcout = count_param(model)   # 计算模型参数量
    # logging.info('Number of Network Params: %.4f M' % (paramcout / 1e6))
    print("Number of Network Params: ", (paramcout / 1e6),"M")

    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        #print(data['x'].shape)
        #print(data['heights'])
        num_iter += 1
        batch_size, num_point, _ = data['pos'].size()
        #print(type(data['cls']))
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']

        data['pos'] = data['pos'].transpose(2, 1)
        data['x'] = data['x'].transpose(2, 1)

        label_one_hot = np.zeros((data['cls'].shape[0], 1))
        for idx in range(data['cls'].shape[0]):
            label_one_hot[idx, data['cls'][idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        label_one_hot = label_one_hot.cuda()
        # data['pos']是坐标轴xyz, data['x']是后面三列, label_one_hot=[8,1]
        logits = model(data['pos'], data['x'], label_one_hot)      # [8,2,4096]
       # logits = logits.permute(0,2,1).contiguous()

        if cfg.criterion_args.NAME != 'MultiShapeCrossEntropy':

            ############ new loss ##################
            # spikeloss = 0
            # leafloss= 0
            #
            # for i in range(len(target)):
            #     temptarget = target[i]
            #     spikeindexs = np.where(temptarget.cpu().numpy() == 1)[0]
            #     leafindexs = np.where(temptarget.cpu().numpy() == 0)[0]
            #     spiketartget = temptarget[spikeindexs]
            #     leaftarget = temptarget[leafindexs]
            #     spikelogits = logits[i, :, spikeindexs]
            #     leaflogits = logits[i, :, leafindexs]
            #     if len(spikeindexs) > 0:
            #         spikeloss = spikeloss + criterion(torch.unsqueeze(spikelogits,0), torch.unsqueeze(spiketartget,0))
            #     else:
            #         spikeloss = spikeloss + 0
            #     leafloss = leafloss + criterion(torch.unsqueeze(leaflogits,0), torch.unsqueeze(leaftarget,0))
            #
            # spikeloss = spikeloss / len(target)
            # leafloss = leafloss / len(target)
            # loss = segcriterion((spikeloss,leafloss))
            #logging.info(f"spikeloss:{spikeloss}    leafloss:{leafloss}")
            ############ new loss ##################

            #######Ori Loss
            loss = criterion(logits, target)     # logits=[8,2,4096], target=[8,4096]
            ##############

        else:
            loss = criterion(logits, target, data['cls'])

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
                                 f"Loss {loss_meter.avg:.3f} " )        # 每次迭代的loss

        # record_iter_train['loss'].append(loss.item())  #  loss_meter.avg
        record_iter_train['loss'].append(loss_meter.avg)  # loss_meter.avg
        record_iter_train['epoch'].append(epoch)

    train_loss = loss_meter.avg
    # print(record_iter)
    # print(record_epoch)
    return train_loss


@torch.no_grad()
def validate(model, val_loader, cfg, epoch, criterion, num_votes=0, data_transform=None):  ##############第5步

    loss_meter = AverageMeter()      ##############第一步

    model.eval()  # set model to eval mode    不再更新模型权重
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    cls_mious = torch.zeros(cfg.shape_classes, dtype=torch.float32).cuda(non_blocking=True)
    cls_nums = torch.zeros(cfg.shape_classes, dtype=torch.int32).cuda(non_blocking=True)
    ins_miou_list, ins_macc_list, ins_mpre_list, ins_mrecall_list = [], [], [], []
    part_iou, part_acc, part_pre, part_recall = [], [], [], []

    # label_size: b, means each sample has one corresponding class
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']     # [8, 4096]
        cls = data['cls']      # [8, 1]
      #  data['x'] = get_features_by_keys(data, cfg.feature_keys)
        batch_size, num_point, _ = data['pos'].size()

        data['pos'] = data['pos'].transpose(2, 1)
        data['x'] = data['x'].transpose(2, 1)

        label_one_hot = np.zeros((data['cls'].shape[0], 1))
        for idx in range(data['cls'].shape[0]):
            label_one_hot[idx, data['cls'][idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        label_one_hot = label_one_hot.cuda()

        logits = 0
        for v in range(num_votes+1):
            set_random_seed(v)
            if v > 0:
                data['pos'] = data_transform(data['pos'])
            logits += model(data['pos'], data['x'], label_one_hot)
        logits /= (num_votes + 1)
        preds = logits.max(dim=1)[1]   #preds即为所求类别   [8, 4096]
        # if cfg.get('refine', False):
        #     part_seg_refinement(preds, data['pos'], data['cls'], cfg.cls2parts, cfg.get('refine_n', 3))

        if cfg.criterion_args.NAME != 'MultiShapeCrossEntropy':
            batch_ins_mious,batch_part_iou = get_part_mious(preds, target, data['cls'], cfg.cls2parts)
            batch_ins_macc,batch_part_acc = get_part_acc(preds, target, data['cls'], cfg.cls2parts)
            batch_ins_mpre,batch_part_pre = get_part_pre(preds, target, data['cls'], cfg.cls2parts)
            batch_ins_mrecall,batch_part_recall = get_part_recall(preds, target, data['cls'], cfg.cls2parts)
            ins_miou_list += batch_ins_mious    # IoU
            part_iou.append(batch_part_iou)
            ins_macc_list += batch_ins_macc     # acc
            part_acc.append(batch_part_acc)
            ins_mpre_list += batch_ins_mpre    # pre
            part_pre.append(batch_part_pre)
            ins_mrecall_list += batch_ins_mrecall    # recall
            part_recall.append(batch_part_recall)
            loss = criterion(logits, target)   ##############第2步
        else:
            iou_array = []
            for ib in range(batch_size):
                sl = data['cls'][ib][0]
                iou = get_ins_mious(preds[ib:ib + 1], target[ib:ib + 1], sl.unsqueeze(0), cfg.cls2parts, multihead=True)
                iou_array.append(iou)
            ins_miou_list += iou_array

        loss_meter.update(loss.item(), n=batch_size)    ##############第3步
        # print(loss.item())
        # record_iter_val['loss'].append(loss.item())
        record_iter_val['loss'].append(loss_meter.avg)
        record_iter_val['epoch'].append(epoch)

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

    part_iou = np.array(part_iou)  # part_iou = 每次迭代中每个类别的iou      # [total/batch, 2]
    part_iou = np.mean(part_iou, axis=0)  # part_iou = 总的每个类别的iou    # [1, 2]
    part_acc = np.array(part_acc)  # part_acc = 每次迭代中每个类别的acc      # [total/batch, 2]
    part_acc = np.mean(part_acc, axis=0)  # part_acc = 总的每个类别的acc    # [1, 2]
    part_pre = np.array(part_pre)  # part_pre = 每次迭代中每个类别的pre      # [total/batch, 2]
    part_pre  = np.mean(part_pre, axis=0)  # part_pre = 总的每个类别的pre   # [1, 2]
    part_recall = np.array(part_recall)  # part_recall = 每次迭代中每个类别的recall     # [total/batch, 2]
    part_recall = np.mean(part_recall, axis=0)  # part_recall = 总的每个类别的recall   # [1, 2]
    ins_miou = ins_mious_sum/count
    cls_miou = torch.mean(cls_mious)

    with np.printoptions(precision=2, suppress=True):
        logging.info(f'Validation Epoch [{cfg.epoch}/{cfg.epochs}],'
                     f'Loss {loss_meter.avg:.3f} '   ##############第4步
                     f'Instance mIoU {ins_miou:.2f}, '
                     f'Class mIoU {cls_miou:.2f}, '
                     f'part IoUs {part_iou},'            # 每个类别的iou
                     f'part ACC {part_acc},'             # 每个类别的acc
                     f'part Pre {part_pre},'             # 每个类别的pre
                     f'part Recall {part_recall},'       # 每个类别的recall
                     f'\n Class mIoUs {cls_mious}')

    # 绘制并保存损失函数图
    record_epoch['val'].append(loss_meter.avg)  # 每次epoch的loss
    record_all['IoU'].append(np.mean(part_iou))
    record_all['IoU spike'].append(part_iou[1])
    record_all['IoU leaf'].append(part_iou[0])
    record_all['test acc'].append(np.mean(part_acc))
    record_all['spike test pre'].append(part_pre[1])
    record_all['leaf test pre'].append(part_pre[0])
    record_all['test pre'].append(np.mean(part_pre))
    record_all['spike test recall'].append(part_recall[1])
    record_all['leaf test recall'].append(part_recall[0])
    record_all['test recall'].append(np.mean(part_recall))
    # print(record_all)

    return ins_miou, cls_miou, cls_mious


@torch.no_grad()
def testmetric(model, val_loader, cfg, num_votes=0, data_transform=None):
    model.eval()  # set model to eval mode
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    cls_mious = torch.zeros(cfg.shape_classes, dtype=torch.float32).cuda(non_blocking=True)
    cls_nums = torch.zeros(cfg.shape_classes, dtype=torch.int32).cuda(non_blocking=True)
    ins_miou_list, ins_macc_list, ins_mpre_list, ins_mrecall_list = [], [], [], []
    part_iou, part_acc, part_pre, part_recall = [], [], [], []
    ### 用测试实现可视化：第一步
    file_visual = {'plot_name': [], 'plot': []}  # 创建一个字典，用于存储文件名前缀和对应的文件内容
    datapath = val_loader.dataset.datapath
    index = 0

    # label_size: b, means each sample has one corresponding class
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']
        cls = data['cls']
       # data['x'] = get_features_by_keys(data, cfg.feature_keys)
        batch_size, num_point, _ = data['pos'].size()
        ### 用测试实现可视化：第二步
        ori_data = data['ori_data']

        data['pos'] = data['pos'].transpose(2, 1)      # 8, 4096, 3 --> 8, 3, 4096
        data['x'] = data['x'].transpose(2, 1)          # 8, 4096, 3
        # cls = ( 8, 1 )      y = ( 8, 4096 )
        label_one_hot = np.zeros((data['cls'].shape[0], 1))   # label_one_hot = ( 8, 1 )
        for idx in range(data['cls'].shape[0]):
            label_one_hot[idx, data['cls'][idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        label_one_hot = label_one_hot.cuda()
        logits = 0
        for v in range(num_votes+1):
            set_random_seed(v)
            # if v > 0:
            #     data['pos'] = data_transform(data['pos'])
            logits += model(data['pos'], data['x'],label_one_hot)
        logits /= (num_votes + 1)      # logits = （8，2，4096）
        preds = logits.max(dim=1)[1]   #preds即为所求类别  preds = （8，4096）
        #print("preds",preds.shape)
        #print(preds)
        # if cfg.get('refine', False):
        #     part_seg_refinement(preds, data['pos'], data['cls'], cfg.cls2parts, cfg.get('refine_n', 10))

        if cfg.criterion_args.NAME != 'MultiShapeCrossEntropy':
            batch_ins_mious,batch_part_iou = get_part_mious(preds, target, data['cls'], cfg.cls2parts)
            batch_ins_macc,batch_part_acc = get_part_acc(preds, target, data['cls'], cfg.cls2parts)
            batch_ins_mpre,batch_part_pre = get_part_pre(preds, target, data['cls'], cfg.cls2parts)
            batch_ins_mrecall,batch_part_recall = get_part_recall(preds, target, data['cls'], cfg.cls2parts)
            ins_miou_list += batch_ins_mious        # IoU
            part_iou.append(batch_part_iou)
            ins_macc_list += batch_ins_macc          # acc
            part_acc.append(batch_part_acc)
            ins_mpre_list += batch_ins_mpre          # pre
            part_pre.append(batch_part_pre)
            ins_mrecall_list += batch_ins_mrecall    # recall
            part_recall.append(batch_part_recall)
        else:
            iou_array = []
            for ib in range(batch_size):
                sl = data['cls'][ib][0]
                iou = get_part_mious(preds[ib:ib + 1], target[ib:ib + 1], sl.unsqueeze(0), cfg.cls2parts, multihead=True)
                iou_array.append(iou)
            ins_miou_list += iou_array

        # per category iou at each batch_size:
        for shape_idx in range(batch_size):  # sample_idx
            cur_gt_label = int(cls[shape_idx].cpu().numpy())
            # add the iou belongs to this cat
            cls_mious[cur_gt_label] += batch_ins_mious[shape_idx]   # batch_ins_mious列表中所有值相加
            cls_nums[cur_gt_label] += 1

        ### 第三步
        cur_pred_val = np.expand_dims(preds.cpu().numpy(), axis=2)  # 扩展维度 (B,N)->(B,N,1)
        seg_points = np.concatenate((ori_data.cpu().numpy(), cur_pred_val), axis=2)  # 拼接后得到有分割标签的原始数据，进行分类保存，
        bsize, _, _ = seg_points.shape
        leafarr = []
        disarr = []
        predictedpath = "/data/home/whs/Pointnext/examples/shapenetpart/log/visual/pointmlp/rice_pointmlp_super_80/"  # 保存可视化预测结果的txt的路径
        if not os.path.exists(predictedpath):
            os.makedirs(predictedpath)

        # # 对batch中每个点云保存
        for item in range(bsize):
            datapath_1 = datapath[index]
            datapath_2 = datapath_1[1].split("/")[-1]
            singlepoints = seg_points[item, :, :]  # 单个点云数据
            output_path = os.path.join(predictedpath, datapath_2)
            np.savetxt(output_path, singlepoints, fmt='%.6f')  # 将拼接后的数组内容写入文件
            print("完成小区可视化预测：" + output_path)
            index = index + 1

    # ins_miou_list = 每个txt文件的miou
    ins_mious_sum, count = torch.sum(torch.stack(ins_miou_list)), torch.tensor(len(ins_miou_list)).cuda()
    if cfg.distributed:
        dist.all_reduce(cls_mious), dist.all_reduce(cls_nums), dist.all_reduce(ins_mious_sum), dist.all_reduce(count)

    for cat_idx in range(cfg.shape_classes):
        # indicating this cat is included during previous iou appending
        if cls_nums[cat_idx] > 0:
            cls_mious[cat_idx] = cls_mious[cat_idx] / cls_nums[cat_idx]

    part_iou = np.array(part_iou)               # part_iou = 每次迭代中每个类别的iou
    part_iou = np.mean(part_iou, axis=0)        # part_iou = 总的每个类别的iou
    part_acc = np.array(part_acc)               # part_acc = 每次迭代中每个类别的acc      # [total/batch, 2]
    part_acc = np.mean(part_acc, axis=0)        # part_acc = 总的每个类别的acc    # [1, 2]
    part_pre = np.array(part_pre)               # part_pre = 每次迭代中每个类别的pre      # [total/batch, 2]
    part_pre  = np.mean(part_pre, axis=0)       # part_pre = 总的每个类别的pre   # [1, 2]
    part_recall = np.array(part_recall)         # part_recall = 每次迭代中每个类别的recall     # [total/batch, 2]
    part_recall = np.mean(part_recall, axis=0)  # part_recall = 总的每个类别的recall   # [1, 2]
    ins_miou = ins_mious_sum/count
    cls_miou = torch.mean(cls_mious)
    with np.printoptions(precision=2, suppress=True):
        logging.info(f'Test Epoch [{cfg.epoch}/{cfg.epochs}],'
                        f'Instance mIoU {ins_miou:.2f}, '
                        f'Class mIoU {cls_miou:.2f}, '
                        f'part IoUs {part_iou},'            # 每个类别的iou
                        f'part ACC {part_acc},'             # 每个类别的acc
                        f'part Pre {part_pre},'             # 每个类别的pre
                        f'part Recall {part_recall},'       # 每个类别的recall
                        f'\n Class mIoUs {cls_mious}')
    # return ins_miou, cls_miou, part_iou, cls_mious
    return ins_miou, cls_miou, part_iou, part_acc, part_pre, part_recall, cls_mious


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser('ShapeNetPart Part segmentation training')  #pointnext_default.yaml.yaml
    parser.add_argument('--cfg', type=str, default="../../cfgs/shapenetpart/pointmlp_default.yaml", help='config file')
    args, opts = parser.parse_known_args()
   #args.cfg = "D:/pythonproject/Pointnext/cfgs/shapenetpart/default.yaml"
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    if cfg.seed is None:
        # cfg.seed = np.random.randint(1, 10000)
        cfg.seed = np.random.randint(1, 100)
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
