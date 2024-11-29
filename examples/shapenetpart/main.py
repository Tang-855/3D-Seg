"""
Distributed training script for scene segmentation with S3DIS dataset
"""
import argparse
import copy
from openpoints.models.pointmlp.pointMLP import pointMLP
import yaml
from datautils.forafterpointDataLoader import ShapeNetPartNormal
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
#from sklearn.metrics import confusion_matrix
from collections import defaultdict, Counter



torch.backends.cudnn.benchmark = False
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

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

global record
record  = {'loss':[],}
global record_all
record_all = {'IoU spike': [], 'IoU leaf': [], 'IoU': [],
                 'spike test acc': [], 'leaf test acc': [], 'test acc': [],
                 'spike test pre': [], 'leaf test pre': [], 'test pre': [],
                 'spike test recall': [], 'leaf test recall': [], 'test recall': [],
                 'train loss': [], 'test loss': []}

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
                    knn_idx = knn_point(n + 1, torch.unsqueeze(less_pos, axis=0),
                                        torch.unsqueeze(pos[shape_idx], axis=0))[1]
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
    ins_mious = []
    final_part_ious = []

    batchpart_iou = []
    for shape_idx in range(pred.shape[0]):  # sample_idx
        part_ious = []
        mypart_iou = []
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

            mypart_iou.append(iou.cpu().float())  #自加


        batchpart_iou.append(mypart_iou)


        ins_mious.append(torch.mean(torch.stack(part_ious)))

    batchpart_iou = np.array(batchpart_iou)
    batchpart_iou = np.mean(batchpart_iou,axis=0)
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
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
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
    logging.info(f"length of training dataset: {len(val_loader.dataset)}")

    test_loader = build_dataloader_from_cfg(cfg.batch_size, cfg.dataset, cfg.dataloader, datatransforms_cfg=cfg.datatransforms, split='test', distributed=cfg.distributed)
    logging.info(f"length of validation dataset: {len(test_loader.dataset)}")

    num_classes = val_loader.dataset.num_classes if hasattr(
        val_loader.dataset, 'num_classes') else None
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

   # 加载预训练模型
   #  try:
   #      # # state_dict = torch.load(pretrained_path, map_location='cpu')
   #      pretrained_model = torch.load(r"./log/shapenetpart/pointmlp/checkpoint/_ckpt_latest.pth")   #加载预训练模型
   #      # pretrained_weights = pretrained_model['model_state_dict']   #加载预训练模型中的参数模型
   #      # del pretrained_weights['module.conv9.weight']   #删除不匹配的模型参数
   #      # del pretrained_weights['module.conv1.0.weight']
   #      pretrained_epoch = pretrained_model['epoch']   #加载当前的预训练epoch
   #      # current_weights = model.state_dict()     #参看当前模型的参数模型
   #      model.load_state_dict(pretrained_model, strict=False)  # False，将预训练模型中的参数加载到当前模型中
   #      logging.info('**************************Use pretrain model**************************')
   #  except:
   #      logging.info('No existing model, starting training from scratch...')
    # cfg.start_epoch = pretrained_epoch

   # 加载自监督参数
   #  selfprepath = r"D:\pythonproject\Pointnext\examples\shapenetpart\log\shapenetpart\815newself\checkpoint\ckpt_best.pth"
   #  logging.info('load self-pretrained params')
   #  load_checkpoint(model, pretrained_path=selfprepath)

    # temppath = r"D:\pythonproject\Pointnext\examples\shapenetpart\log\shapenetpart\810selfbest\checkpoint\ckpt_best.pth"
    # myselfsupervisepath = "D:/pythonproject/Pointnext/examples/shapenetpart/256_16_mybestself.pth"
    # model.load_state_dict(torch.load(myselfsupervisepath))
    #
    # logging.info('load self-pretrained params')
   # print(torch.load(myselfsupervisepath))

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


    train_loader = build_dataloader_from_cfg(cfg.batch_size, cfg.dataset, cfg.dataloader, datatransforms_cfg=cfg.datatransforms, split='train', distributed=cfg.distributed,)
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")

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
        cfg.epoch = epoch
        train_loss = \
            train_one_epoch(model, train_loader, criterion,segcriterion, optimizer, scheduler, epoch, cfg)
        is_best = False
        if epoch % cfg.val_freq == 0:
            val_ins_miou, val_cls_miou, val_cls_mious = validate_fn(model, val_loader, cfg, criterion)  ###############第6步  加criterion
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
            writer.add_scalar('best_val_instance_miou',
                              best_ins_miou, epoch)
            writer.add_scalar('val_class_miou_when_best', cls_miou_when_best, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('lr', lr, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)

        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'ins_miou': best_ins_miou,
                                             'cls_miou': cls_miou_when_best}, is_best=is_best)
    # if writer is not None:
    #     Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    # Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.logname}_ckpt_latest.pth'))
    with np.printoptions(precision=2, suppress=True):
        logging.info(f'Best Epoch {best_epoch},'
                     f'Instance mIoU {best_ins_miou:.2f}, '
                     f'Class mIoU {cls_miou_when_best:.2f}, '
                     f'\n Class mIoUs {cls_mious_when_best}')

    if cfg.get('num_votes', 0) > 0:
        load_checkpoint(model, pretrained_path=os.path.join(
            cfg.ckpt_dir, 'ckpt_best.pth'))
        set_random_seed(cfg.seed)
        test_ins_miou, test_cls_miou, test_part_iou, test_cls_mious  = testmetric(model, test_loader, cfg, num_votes=cfg.get('num_votes', 0), data_transform=voting_transform)
        # with np.printoptions(precision=2, suppress=True):
        #     logging.info(f'Best Epoch {best_epoch},'
        #                 f'Voting Instance mIoU {test_ins_miou:.2f}, '
        #                 f'Voting Paty mIoU {test_part_iou:.2f}, '
        #                 f'\n Voting Class mIoUs {test_cls_mious}')

        # if writer is not None:
        #     writer.add_scalar('test_ins_miou_voting', test_ins_miou, epoch)
        #     writer.add_scalar('test_class_miou_voting', test_cls_miou, epoch)
    torch.cuda.synchronize()
    if writer is not None:
        writer.close()
    #dist.destroy_process_group()
    wandb.finish(exit_code=True)


def train_one_epoch(model, train_loader, criterion, segcriterion, optimizer, scheduler, epoch, cfg):
    cls_mious = torch.zeros(cfg.shape_classes, dtype=torch.float32).cuda(non_blocking=True)
    cls_nums = torch.zeros(cfg.shape_classes, dtype=torch.int32).cuda(non_blocking=True)
    ins_miou_list = []
    part_iou = []
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

    paramcout = count_param(model)
    print("网络参数量:", paramcout)

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
        cls = data['cls']

        label_one_hot = np.zeros((data['cls'].shape[0], 1))
        for idx in range(data['cls'].shape[0]):
            label_one_hot[idx, data['cls'][idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        label_one_hot = label_one_hot.cuda()

        logits = model(data['pos'], data['x'], label_one_hot)
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
            loss = criterion(logits, target)
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
        preds = logits.max(dim=1)[1]  # preds即为所求类别
        if cfg.criterion_args.NAME != 'MultiShapeCrossEntropy':
            batch_ins_mious,batch_part_iou = get_part_mious(preds, target, data['cls'], cfg.cls2parts)
            ins_miou_list += batch_ins_mious
            part_iou.append(batch_part_iou)
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
            cls_mious[cur_gt_label] += batch_ins_mious[shape_idx]
            cls_nums[cur_gt_label] += 1

    ins_mious_sum, count = torch.sum(torch.stack(ins_miou_list)), torch.tensor(len(ins_miou_list)).cuda()
    if cfg.distributed:
        dist.all_reduce(cls_mious), dist.all_reduce(cls_nums), dist.all_reduce(ins_mious_sum), dist.all_reduce(count)

    for cat_idx in range(cfg.shape_classes):
        # indicating this cat is included during previous iou appending
        if cls_nums[cat_idx] > 0:
            cls_mious[cat_idx] = cls_mious[cat_idx] / cls_nums[cat_idx]

    part_iou = np.array(part_iou)
    part_iou = np.mean(part_iou, axis=0)
    ins_miou = ins_mious_sum / count
    cls_miou = torch.mean(cls_mious)
    if idx % cfg.print_freq:
        pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                             f"Loss {loss_meter.avg:.3f} "
                             f'Instance mIoU {ins_miou:.2f}, '
                             f'Class mIoU {cls_miou:.2f}, '
                             f'part IoUs {part_iou},'       # 每个类别的
                             )
    train_loss = loss_meter.avg
    return train_loss


@torch.no_grad()
def validate(model, val_loader, cfg, criterion, num_votes=0, data_transform=None):  ##############第5步

    loss_meter = AverageMeter()      ##############第一步

    model.eval()  # set model to eval mode    不再更新模型权重
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    cls_mious = torch.zeros(cfg.shape_classes, dtype=torch.float32).cuda(non_blocking=True)
    cls_nums = torch.zeros(cfg.shape_classes, dtype=torch.int32).cuda(non_blocking=True)
    ins_miou_list = []
    part_iou = []

    # label_size: b, means each sample has one corresponding class
    for idx, data in pbar:

        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)

        target = data['y']
        cls = data['cls']
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
        preds = logits.max(dim=1)[1]   #preds即为所求类别
        # if cfg.get('refine', False):
        #     part_seg_refinement(preds, data['pos'], data['cls'], cfg.cls2parts, cfg.get('refine_n', 3))

        if cfg.criterion_args.NAME != 'MultiShapeCrossEntropy':
            batch_ins_mious,batch_part_iou = get_ins_mious(preds, target, data['cls'], cfg.cls2parts)
            ins_miou_list += batch_ins_mious
            part_iou.append(batch_part_iou)
            loss = criterion(logits, target)   ##############第2步

        else:
            iou_array = []
            for ib in range(batch_size):
                sl = data['cls'][ib][0]
                iou = get_ins_mious(preds[ib:ib + 1], target[ib:ib + 1], sl.unsqueeze(0), cfg.cls2parts,
                                    multihead=True)
                iou_array.append(iou)
            ins_miou_list += iou_array

        loss_meter.update(loss.item(), n=batch_size)    ##############第3步

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

    part_iou = np.array(part_iou)
    part_iou = np.mean(part_iou, axis=0)
    ins_miou = ins_mious_sum/count
    cls_miou = torch.mean(cls_mious)
    with np.printoptions(precision=2, suppress=True):
        logging.info(f'Test Epoch [{cfg.epoch}/{cfg.epochs}],'
                     f'Loss {loss_meter.avg:.3f} '   ##############第4步
                        f'Instance mIoU {ins_miou:.2f}, '
                        f'Class mIoU {cls_miou:.2f}, '
                        f'\n Class mIoUs {cls_mious}'
                        f'part IoUs {part_iou},')       # 每个类别的
    return ins_miou, cls_miou, cls_mious


@torch.no_grad()
def testmetric(model, val_loader, cfg, num_votes=0, data_transform=None):
    model.eval()  # set model to eval mode
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    cls_mious = torch.zeros(cfg.shape_classes, dtype=torch.float32).cuda(non_blocking=True)
    cls_nums = torch.zeros(cfg.shape_classes, dtype=torch.int32).cuda(non_blocking=True)
    ins_miou_list = []
    part_iou = []

    # label_size: b, means each sample has one corresponding class
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']
        cls = data['cls']
       # data['x'] = get_features_by_keys(data, cfg.feature_keys)
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
            # if v > 0:
            #     data['pos'] = data_transform(data['pos'])
            logits += model(data['pos'], data['x'],label_one_hot)
        logits /= (num_votes + 1)
        preds = logits.max(dim=1)[1]   #preds即为所求类别
        #print("preds",preds.shape)
        #print(preds)
        # if cfg.get('refine', False):
        #     part_seg_refinement(preds, data['pos'], data['cls'], cfg.cls2parts, cfg.get('refine_n', 10))

        if cfg.criterion_args.NAME != 'MultiShapeCrossEntropy':
            batch_ins_mious,batch_part_iou = get_part_mious(preds, target, data['cls'], cfg.cls2parts)
            ins_miou_list += batch_ins_mious
            part_iou.append(batch_part_iou)
        else:
            iou_array = []
            for ib in range(batch_size):
                sl = data['cls'][ib][0]
                iou = get_part_mious(preds[ib:ib + 1], target[ib:ib + 1], sl.unsqueeze(0), cfg.cls2parts,
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

    part_iou = np.array(part_iou)
    part_iou = np.mean(part_iou, axis=0)
    ins_miou = ins_mious_sum/count
    cls_miou = torch.mean(cls_mious)
    with np.printoptions(precision=2, suppress=True):
        logging.info(f'Test Epoch [{cfg.epoch}/{cfg.epochs}],'
                        f'Instance mIoU {ins_miou:.2f}, '
                        f'Class mIoU {cls_miou:.2f}, '
                        f'part IoUs {part_iou},'       # 每个类别的
                        f'\n Class mIoUs {cls_mious}')
    return ins_miou, cls_miou, part_iou, cls_mious


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
