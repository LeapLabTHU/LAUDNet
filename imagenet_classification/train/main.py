import os
import sys
import math
import time
import random
import argparse
import warnings
import pandas as pd
sys.path.append('.')
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as pytorchmodels
import torchvision.transforms as transforms
import numpy as np

import models
from models.op_counter import measure_model
from utils.utils import *
from utils.config import Config
from utils.optimizer import get_optimizer
from utils.criterion import get_criterion
from utils.scheduler import get_scheduler
from utils.transform import get_transform
from utils.hyperparams import get_hyperparams
from utils.sparsity_loss_unify import SparsityCriterion_bounds

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--config', help='train config file path', default=None)
parser.add_argument('--data_url', type=str, metavar='DIR', default='/home/data/ImageNet/', help='path to dataset')
parser.add_argument('--train_url', type=str, metavar='PATH', default='./log/',
                    help='path to save result and checkpoint (default: results/savedir)')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet', choices=['imagenet'],
                    help='dataset')
parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--lr_mult', default=0.1, type=float)
parser.add_argument('--scheduler', default='cosine', type=str, metavar='T',
                    help='learning rate strategy (default: multistep)',
                    choices=['cosine', 'multistep', 'linear'])
parser.add_argument('--labelsmooth', default=0., type=float)
parser.add_argument('--warmup_epoch', default=None, type=int, metavar='N',
                    help='number of epochs to warm up')
parser.add_argument('--warmup_lr', default=0.1, type=float,
                    metavar='LR', help='initial warm up learning rate (default: 0.1)')
parser.add_argument('--weigh_decay_apply_on_all', default=True, type=str)
parser.add_argument('--nesterov', default=True, type=str)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', action='store_true',
                    help='evaluate model on validation set (default: false)')
parser.add_argument('--evaluate_from', default=None, type=str, metavar='PATH',
                    help='path to saved checkpoint (default: none)')
# hyperparameter
parser.add_argument('--hyperparams_set_index', default=1, type=int,
                    help='choose which hyperparameter set to use')
parser.add_argument('--init_method', type=str, default='',
                    help='an argument needed in huawei cloud, but i do not know its usage')
parser.add_argument('--test_code', default=0, type=int,
                    help='whether to test the code')
# multiprocess
parser.add_argument('--world_size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:29500', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--visible_gpus', type=str, default='0',
                    help='visible gpus')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
# training
parser.add_argument('--t0', default=1.0, type=float, metavar='M')
parser.add_argument('--t_last', default=0.01, type=float, metavar='M')
parser.add_argument('--target_rate', default=1.0, type=float, metavar='M')
parser.add_argument('--lambda_act', default=0.1, type=float, metavar='M')
parser.add_argument('--temp', default=0.1, type=float, metavar='M')
# structure
parser.add_argument('--arch', default='resnet56', type=str)
parser.add_argument('--width', default=1, type=int)
parser.add_argument('--sparse', action='store_true', default=False)
parser.add_argument('--mask_channel_group', default='1-1-1-1', type=str)
parser.add_argument('--mask_spatial_granularity', default='1-1-1-1', type=str)
parser.add_argument('--channel_dyn_granularity', default='1-1-1-1', type=str)
parser.add_argument('--dyn_mode', default='both-both-both-both', type=str)
parser.add_argument('--channel_masker', default='MLP-MLP-MLP-MLP', type=str)
parser.add_argument('--channel_masker_layers', default='2-2-2-2', type=str)
parser.add_argument('--channel_masker_reduction', default='16-16-16-16', type=str)

parser.add_argument('--finetune_from', default=None, type=str, metavar='PATH', help='path to finetune checkpoint (default: none)')
parser.add_argument('--teacher_path', default=None, type=str, metavar='PATH', help='path to teacher checkpoint')

parser.add_argument('--temp_scheduler', default='exp', type=str)
parser.add_argument('--target_begin_epoch', default=0, type=int)
parser.add_argument('--start_eval_epoch', default=0, type=int)
parser.add_argument('--round', default=0, type=int)
parser.add_argument('--IMAGE_SIZE', default=224, type=int)

parser.add_argument('--T_kd', default=1.0, type=float, metavar='M', help='T for kd loss')
parser.add_argument('--alpha_kd', default=0.5, type=float, metavar='M', help='alpha for kd loss')
parser.add_argument('--channel_target', default=None, type=float)

args = parser.parse_args()

args.use_amp = False
args.sparse = True

best_acc1 = 0
best_acc1_corresponding_acc5 = 0
val_acc_top1 = []
val_acc_top5 = []
tr_acc_top1 = []
tr_acc_top5 = []
train_loss = []
valid_loss = []
lr_log = []
epoch_log = []
val_act_rate = []
val_FLOPs = []
args.temp = args.t0


def main():
    assert args.dataset == 'imagenet'
    args.num_classes = 1000
    args.multiprocessing_distributed = True

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global best_acc1_corresponding_acc5
    global val_acc_top1
    global val_acc_top5
    global tr_acc_top1
    global tr_acc_top5
    global train_loss
    global valid_loss
    global lr_log
    global epoch_log
    global val_FLOPs
    global val_act_rate
    args.gpu = gpu
    args.cfg = Config.fromfile(args.config)
    args.ngpus_per_node = ngpus_per_node
    
    print(args.cfg)
    args.hyperparams_set_index = args.cfg['train_cfg']['hyperparams_set_index']
    args = get_hyperparams(args, test_code=args.test_code)
    
    args.autoaugment = False
    args.colorjitter = False
    args.change_light = False
    args.labelsmooth = 0.0
    args.mixup = 0.0
    
    args.t_last_epoch = args.epochs
    str_t0 = str(args.t0).replace('.', 'x')
    str_lambda = str(args.lambda_act).replace('.', 'x')
    str_ta = str(args.target_rate).replace('.', 'x')
    str_t_last = str(args.t_last).replace('.', 'x')
    str_lr = str(args.lr).replace('.', 'x')
    
    args.list_mask_channel_group = list(args.mask_channel_group.split('-'))
    args.list_mask_channel_group = [int(x) for x in args.list_mask_channel_group]
    args.list_mask_spatial_granularity = list(args.mask_spatial_granularity.split('-'))
    args.list_mask_spatial_granularity = [int(x) for x in args.list_mask_spatial_granularity]
    args.list_channel_dyn_granularity = list(args.channel_dyn_granularity.split('-'))
    args.list_channel_dyn_granularity = [int(x) for x in args.list_channel_dyn_granularity]
    
    args.list_dyn_mode = list(args.dyn_mode.split('-'))
    
    args.list_channel_masker = list(args.channel_masker.split('-'))
    args.list_channel_masker_layers = list(args.channel_masker_layers.split('-'))
    args.list_channel_masker_layers = [int(x) for x in args.list_channel_masker_layers]
    args.list_channel_masker_reduction = list(args.channel_masker_reduction.split('-'))
    args.list_channel_masker_reduction = [int(x) for x in args.list_channel_masker_reduction]
    
    args.train_url = f'{args.train_url}/{args.arch}/dyn_mode_{args.dyn_mode}/channel_masker_{args.channel_masker}_layer_{args.list_channel_masker_layers[0]}_reduction_{args.list_channel_masker_reduction[0]}/channel_dyn_granularity_{args.channel_dyn_granularity}/mask_channel_group_{args.mask_channel_group}/spatial_granularity_{args.mask_spatial_granularity}/' \
        + f'epochs_{args.epochs}_bs_{args.batch_size}_lr{str_lr}_lr_mult{args.lr_mult}_t0_{str_t0}_tLast{str_t_last}_tempScheduler_{args.temp_scheduler}_tKD_{args.T_kd}_alphaKD_{args.alpha_kd}_lambda{str_lambda}_ta_begin_epoch{args.target_begin_epoch}_target{str_ta}_channel_target{args.channel_target}/'
    os.makedirs(args.train_url, exist_ok=True)
    logger = Logger(args.train_url + 'screen_output.txt') if not args.resume else Logger(args.train_url + 'screen_output_resume.txt')
    args.print_custom = logger.log

    with open(args.train_url+'train_configs.txt', "w") as f:
        f.write(str(args))

    if args.gpu is not None:
        args.print_custom("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    ### Create model
    args.print_custom(args.arch)
    model = eval(f'models.{args.arch}')(input_size=args.IMAGE_SIZE,
                                        channel_dyn_granularity=args.list_channel_dyn_granularity,
                                        spatial_mask_channel_group=args.list_mask_channel_group, 
                                        mask_spatial_granularity=args.list_mask_spatial_granularity,
                                        dyn_mode=args.list_dyn_mode,
                                        lr_mult=args.lr_mult,
                                        channel_masker=args.list_channel_masker,
                                        channel_masker_layers=args.list_channel_masker_layers,
                                        reduction_ratio=args.list_channel_masker_reduction)
    
    ### Load Pretrained
    if args.finetune_from is not None:
        args.teacher_path = args.finetune_from
        
        args.print_custom(f'loading static ckpt from {args.finetune_from}')
        model.load_state_dict(torch.load(args.finetune_from), strict=False)
        args.print_custom(f'loaded static ckpt from {args.finetune_from}')
    args.print_custom('Model Struture:', str(model))
    with open(args.train_url+'model_arch.txt', "w") as f:
        f.write(str(model))

    ### Calculate FLOPs & Param
    model.eval()
    
    model_teacher = eval(f'models.{args.arch[4:]}')()
    model_teacher.eval()
    
    args.print_custom(f'loading static ckpt from teacher_path: {args.teacher_path}')
    model_teacher.load_state_dict(torch.load(args.teacher_path), strict=True)
    args.print_custom(f'loaded static ckpt from teacher_path: {args.teacher_path}')
    
    args.full_flops, _ = measure_model(model_teacher, 224, 224)
    args.full_flops /= 1e9
    args.print_custom(f'FULL FLOPs: {args.full_flops} x 1e9')

    args.target_flops = args.full_flops * args.target_rate

    ### Optionally evaluate from a model
    if args.evaluate_from is not None:
        args.evaluate = True
        state_dict = torch.load(args.evaluate_from, map_location='cpu')['state_dict']
        model.load_state_dict(state_dict)

    ### Define criterion and optimizer
    criterion = get_criterion(args).to(args.gpu)
    sparsity_criterion = SparsityCriterion_bounds(args.target_rate, args.epochs, args.full_flops)
    scheduler = get_scheduler(args)
    policies = model.get_optim_policies()
    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    
    # DistributedDataParallel
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model_teacher.cuda(args.gpu)
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_teacher = torch.nn.parallel.DistributedDataParallel(model_teacher, device_ids=[args.gpu])

    # auto resume
    auto_resume_dir = os.path.join(args.train_url, "checkpoint.pth.tar")
    if os.path.exists(auto_resume_dir):
        args.print_custom("=> loading checkpoint '{}'".format(auto_resume_dir))
        if args.gpu is None:
            checkpoint = torch.load(auto_resume_dir)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(auto_resume_dir, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        best_acc1_corresponding_acc5 = checkpoint['best_acc1_corresponding_acc5']

        model.module.load_state_dict(checkpoint['state_dict'])
        if not args.evaluate:
            optimizer.load_state_dict(checkpoint['optimizer'])
        val_acc_top1 = checkpoint['val_acc_top1']
        val_acc_top5 = checkpoint['val_acc_top5']
        tr_acc_top1 = checkpoint['tr_acc_top1']
        tr_acc_top5 = checkpoint['tr_acc_top5']
        train_loss = checkpoint['train_loss']
        valid_loss = checkpoint['valid_loss']
        lr_log = checkpoint['lr_log']
        val_act_rate = checkpoint['val_act_rate']
        val_FLOPs = checkpoint['val_FLOPs']
        args.temp = checkpoint['temp']
        try:
            epoch_log = checkpoint['epoch_log']
        except:
            args.print_custom('There is no epoch_log in checkpoint!')
        args.print_custom("=> loaded checkpoint '{}' (epoch {})"
                .format(auto_resume_dir, checkpoint['epoch']))
    else:
        args.print_custom("=> no checkpoint found at '{}'".format(auto_resume_dir))
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            args.print_custom("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_acc1_corresponding_acc5 = checkpoint['best_acc1_corresponding_acc5']

            model.module.load_state_dict(checkpoint['state_dict'])
            if not args.evaluate:
                optimizer.load_state_dict(checkpoint['optimizer'])
            val_acc_top1 = checkpoint['val_acc_top1']
            val_acc_top5 = checkpoint['val_acc_top5']
            tr_acc_top1 = checkpoint['tr_acc_top1']
            tr_acc_top5 = checkpoint['tr_acc_top5']
            train_loss = checkpoint['train_loss']
            valid_loss = checkpoint['valid_loss']
            lr_log = checkpoint['lr_log']
            val_act_rate = checkpoint['val_act_rate']
            val_FLOPs = checkpoint['val_FLOPs']
            args.temp = checkpoint['temp']
            try:
                epoch_log = checkpoint['epoch_log']
            except:
                args.print_custom('There is no epoch_log in checkpoint!')
            args.print_custom("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            args.print_custom("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    ### Data loading
    args.print_custom('Train data augmentaion:', get_transform(args, is_train_set=True))
    args.print_custom('Valid data augmentaion:', get_transform(args, is_train_set=False))

    traindir = args.data_url + 'train/'
    valdir = args.data_url + 'val/'

    train_dataset = datasets.ImageFolder(
        traindir,
        get_transform(args, is_train_set=True)
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        get_transform(args, is_train_set=False)
    )
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(args.batch_size*1.5), sampler=val_sampler, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, sparsity_criterion, args, epoch)
        return

    epoch_time = AverageMeter('Epoch Time', ':6.3f')
    start_time = time.time()
    is_best = True
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        ### Train for one epoch
        target_rate = 1.0 if epoch < args.target_begin_epoch else args.target_rate
        args.print_custom(f'Epoch {epoch}, Target rate: {target_rate}')
        args.print_custom(f'Temperature: {args.temp}')
       
        tr_acc1, tr_acc5, tr_loss, lr = \
            train(train_loader, model, model_teacher, criterion, sparsity_criterion, optimizer, scheduler, epoch, args)

        val_acc1, val_acc5, val_loss, val_rate, val_flops, all_density = validate(val_loader, model, criterion, sparsity_criterion, args, epoch)
        ### Record best Acc@1 and save checkpoint
        np.savetxt(os.path.join(args.train_url, 'all_density_latest.txt'), all_density)
        
        is_best = val_acc1 > best_acc1
        if is_best:
            best_acc1_corresponding_acc5 = val_acc5
            np.savetxt(os.path.join(args.train_url, 'all_density_best.txt'), all_density)
        best_acc1 = max(val_acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            val_acc_top1.append(val_acc1)
            val_acc_top5.append(val_acc5)
            tr_acc_top1.append(tr_acc1)
            tr_acc_top5.append(tr_acc5)
            val_act_rate.append(val_rate)
            val_FLOPs.append(val_flops)
            train_loss.append(tr_loss)
            valid_loss.append(val_loss)
            lr_log.append(lr)
            epoch_log.append(epoch+1)

            df = pd.DataFrame({'epoch_log': epoch_log, 'lr_log': lr_log,
                                'train_loss': train_loss, 'tr_acc_top1': tr_acc_top1, 'tr_acc_top5': tr_acc_top5, 
                                'valid_loss': valid_loss, 'val_acc_top1': val_acc_top1, 'val_acc_top5': val_acc_top5, 
                                'val_act_rate': val_act_rate, 'val_FLOPs': val_FLOPs, 
                                })
            log_file = args.train_url + 'log.txt'
            with open(log_file, "w") as f:
                df.to_csv(f)
        
            ckpt_name = f'checkpoint.pth.tar'
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.arch,
                'hyper_set': str(args),
                'state_dict': model.module.state_dict(),
                'best_acc1': best_acc1,
                'best_acc1_corresponding_acc5': best_acc1_corresponding_acc5,
                'optimizer': optimizer.state_dict(),
                'val_acc_top1': val_acc_top1,
                'val_acc_top5': val_acc_top5,
                'val_act_rate': val_act_rate,
                'val_FLOPs': val_FLOPs,
                'tr_acc_top1': tr_acc_top1,
                'tr_acc_top5': tr_acc_top5,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'lr_log': lr_log,
                'epoch_log': epoch_log,
                'temp': args.temp,
            }, args, is_best, filename=ckpt_name)

        epoch_time.update(time.time() - start_time, 1)
        start_time = time.time()
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            args.print_custom('Duration: %4f H, Left Time: %4f H' % (epoch_time.sum / 3600, epoch_time.avg * (args.epochs - epoch - 1) / 3600))

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        args.print_custom(' * Best Acc@1 {best_acc1:.3f} Acc@5 {best_acc1_corresponding_acc5:.3f}'
            .format(best_acc1=best_acc1, best_acc1_corresponding_acc5=best_acc1_corresponding_acc5))
        
        log_file = args.train_url + 'log.txt'
        file1 = pd.read_csv(log_file)
        acc1 = np.array(file1['val_acc_top1'])
        rate1 = np.array(file1['val_act_rate'])
        flops1 = np.array(file1['val_FLOPs'])
        loc = np.argmax(acc1)
        max_acc = acc1[loc]
        acc_rate = rate1[loc]
        acc_flops = flops1[loc]
        with open(os.path.join(args.train_url, 'best_result.txt'), 'w') as f:
            f.write("%.6f\t%.6f\t%.6f" % (max_acc, acc_rate, acc_flops))
    return

def train(train_loader, model, model_teacher, criterion, sparsity_criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_cls = AverageMeter('Loss_cls', ':.4e')
    losses_flops = AverageMeter('loss_flops', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    act_rates = AverageMeter('FLOPS percentage', ':.2e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    FLOPs = AverageMeter('FLOPs', ':.4e')
    train_batches_num = len(train_loader)

    train_progress = ProgressMeter(
        train_batches_num,
        [batch_time, data_time,act_rates, losses, losses_cls, losses_flops,FLOPs, top1, top5],
        args=args, prefix="Epoch: [{}/{}]".format(epoch+1, args.epochs))

    model.train()

    end = time.time()
    num_samples = 0
    for i, (images, target) in enumerate(train_loader):
        ### Adjust learning rate
        lr = scheduler.step(optimizer, epoch, batch=i, nBatch=len(train_loader))

        images = images.to(args.gpu, non_blocking=True)
        target = target.to(args.gpu, non_blocking=True)
        batch_size = images.size(0)
        num_samples += batch_size
        
        ### Measure data loading time
        data_time.update(time.time() - end)

        ### Compute output
        adjust_gs_temperature(epoch, i, train_batches_num, args)
        output, _, _, _, channel_sparsity_list, flops_perc_list, flops = model(images, temperature=args.temp)
        loss_cls = criterion(output, target)
        flops /= 1e9
        ### Measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
        
        act_rate = torch.mean(flops_perc_list)
        loss_flops = sparsity_criterion(epoch, flops_perc_list, flops)

        with torch.no_grad():
            out_teacher = model_teacher(images)
        
        kd_loss = F.kl_div(
            F.log_softmax(output/args.T_kd, dim=1),
            F.softmax(out_teacher.detach()/args.T_kd, dim=1),
            reduction='batchmean'
        ) * args.T_kd**2
        loss =  args.lambda_act * loss_flops + loss_cls + args.alpha_kd * kd_loss

        act_rates.update(act_rate.item(), images.size(0))
        losses_flops.update(loss_flops.item(), images.size(0))
        FLOPs.update(flops.item(), images.size(0))
        
        losses_cls.update(loss_cls.item(), images.size(0))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        ### Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % args.print_freq == 0) and (args.rank % args.ngpus_per_node == 0):
            train_progress.display(i)
            args.print_custom('LR: %6.8f' % (lr))
            args.print_custom('act_rate: %6.4f' % (act_rates.avg))
            args.print_custom('FLOPs: %6.4f' % (FLOPs.avg))
    
    return top1.avg, top5.avg, losses.avg, lr


def validate(val_loader, model, criterion, sparsity_criterion, args, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses_cls = AverageMeter('Loss_cls', ':.4e')
    losses_flops = AverageMeter('loss_flops', ':.4e')
    FLOPs = AverageMeter('FLOPs', ':.2e')
    losses = AverageMeter('Loss', ':.4e')
    act_rates = AverageMeter('FLOPs percentage', ':.2e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time,FLOPs, act_rates,losses, losses_cls, losses_flops, top1, top5],
        args=args, prefix='Test: ')

    model.eval()

    end = time.time()
    
    all_spatial_sparsity_conv3_list, all_spatial_sparsity_conv2_list, all_spatial_sparsity_conv1_list, all_channel_sparsity_list = [], [], [], []
    
    with torch.no_grad():
        num_samples = 0
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            batch_size = images.size(0)
            num_samples += batch_size
            
            ### Compute output single crop
            output, spatial_sparsity_conv3_list, spatial_sparsity_conv2_list, spatial_sparsity_conv1_list, channel_sparsity_list, flops_perc_list, flops = model(images, temperature=args.t_last)
            
            # conv3, 4 stages
            all_spatial_sparsity_conv3_stage1 = spatial_sparsity_conv3_list[0]*batch_size if i==0 else all_spatial_sparsity_conv3_stage1 + spatial_sparsity_conv3_list[0]*batch_size
            all_spatial_sparsity_conv3_stage2 = spatial_sparsity_conv3_list[1]*batch_size if i==0 else all_spatial_sparsity_conv3_stage2 + spatial_sparsity_conv3_list[1]*batch_size
            all_spatial_sparsity_conv3_stage3 = spatial_sparsity_conv3_list[2]*batch_size if i==0 else all_spatial_sparsity_conv3_stage3 + spatial_sparsity_conv3_list[2]*batch_size
            all_spatial_sparsity_conv3_stage4 = spatial_sparsity_conv3_list[3]*batch_size if i==0 else all_spatial_sparsity_conv3_stage4 + spatial_sparsity_conv3_list[3]*batch_size
            
            # conv2, 4 stages
            all_spatial_sparsity_conv2_stage1 = spatial_sparsity_conv2_list[0]*batch_size if i==0 else all_spatial_sparsity_conv2_stage1 + spatial_sparsity_conv2_list[0]*batch_size
            all_spatial_sparsity_conv2_stage2 = spatial_sparsity_conv2_list[1]*batch_size if i==0 else all_spatial_sparsity_conv2_stage2 + spatial_sparsity_conv2_list[1]*batch_size
            all_spatial_sparsity_conv2_stage3 = spatial_sparsity_conv2_list[2]*batch_size if i==0 else all_spatial_sparsity_conv2_stage3 + spatial_sparsity_conv2_list[2]*batch_size
            all_spatial_sparsity_conv2_stage4 = spatial_sparsity_conv2_list[3]*batch_size if i==0 else all_spatial_sparsity_conv2_stage4 + spatial_sparsity_conv2_list[3]*batch_size

            # conv1, 4 stages
            all_spatial_sparsity_conv1_stage1 = spatial_sparsity_conv1_list[0]*batch_size if i==0 else all_spatial_sparsity_conv1_stage1 + spatial_sparsity_conv1_list[0]*batch_size
            all_spatial_sparsity_conv1_stage2 = spatial_sparsity_conv1_list[1]*batch_size if i==0 else all_spatial_sparsity_conv1_stage2 + spatial_sparsity_conv1_list[1]*batch_size
            all_spatial_sparsity_conv1_stage3 = spatial_sparsity_conv1_list[2]*batch_size if i==0 else all_spatial_sparsity_conv1_stage3 + spatial_sparsity_conv1_list[2]*batch_size
            all_spatial_sparsity_conv1_stage4 = spatial_sparsity_conv1_list[3]*batch_size if i==0 else all_spatial_sparsity_conv1_stage4 + spatial_sparsity_conv1_list[3]*batch_size
            
            # channel, 4 stages
            all_channel_sparsity_stage1 = channel_sparsity_list[0]*batch_size if i==0 else all_channel_sparsity_stage1 + channel_sparsity_list[0]*batch_size
            all_channel_sparsity_stage2 = channel_sparsity_list[1]*batch_size if i==0 else all_channel_sparsity_stage2 + channel_sparsity_list[1]*batch_size
            all_channel_sparsity_stage3 = channel_sparsity_list[2]*batch_size if i==0 else all_channel_sparsity_stage3 + channel_sparsity_list[2]*batch_size
            all_channel_sparsity_stage4 = channel_sparsity_list[3]*batch_size if i==0 else all_channel_sparsity_stage4 + channel_sparsity_list[3]*batch_size
            
            flops /= 1e9

            loss_cls= criterion(output, target)
            dist.all_reduce(loss_cls)
            loss_cls /= args.world_size
            
            if args.sparse:
                act_rate = torch.mean(flops_perc_list)
                loss_flops = sparsity_criterion(epoch, flops_perc_list, flops)
                loss = loss_cls + args.lambda_act * loss_flops
                
                dist.all_reduce(act_rate)
                act_rate /= args.world_size
                act_rates.update(act_rate.item(), images.size(0))
                
                dist.all_reduce(loss_flops)
                loss_flops /= args.world_size
                losses_flops.update(loss_flops.item(),images.size(0))
                
                dist.all_reduce(flops)
                flops /= args.world_size
                FLOPs.update(flops.item(), images.size(0))
            else:
                loss = loss_cls
                act_rate = 1.0
                loss_flops = 0.0
                act_rates.update(act_rate, images.size(0))
                losses_flops.update(loss_flops, images.size(0))
                FLOPs.update(flops, images.size(0))

            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
            dist.all_reduce(acc1)
            acc1 /= args.world_size
            dist.all_reduce(acc5)
            acc5 /= args.world_size
            dist.all_reduce(loss)
            loss /= args.world_size
            losses_cls.update(loss_cls.item(), images.size(0))
            
            losses.update(loss.data.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            ### Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)
            
    all_spatial_sparsity_conv3_list = [all_spatial_sparsity_conv3_stage1, all_spatial_sparsity_conv3_stage2, all_spatial_sparsity_conv3_stage3, all_spatial_sparsity_conv3_stage4]
    all_spatial_sparsity_conv2_list = [all_spatial_sparsity_conv2_stage1, all_spatial_sparsity_conv2_stage2, all_spatial_sparsity_conv2_stage3, all_spatial_sparsity_conv2_stage4]
    all_spatial_sparsity_conv1_list = [all_spatial_sparsity_conv1_stage1, all_spatial_sparsity_conv1_stage2, all_spatial_sparsity_conv1_stage3, all_spatial_sparsity_conv1_stage4]
    all_channel_sparsity_list = [all_channel_sparsity_stage1, all_channel_sparsity_stage2, all_channel_sparsity_stage3, all_channel_sparsity_stage4]
    
    for i in range(4):
        if args.list_dyn_mode in ['channel', 'both']:
            dist.all_reduce(all_channel_sparsity_list[i])
            all_channel_sparsity_list[i] /= args.world_size
            
        if args.list_dyn_mode in ['spatial', 'both']:
            dist.all_reduce(all_spatial_sparsity_conv1_list[i])
            all_spatial_sparsity_conv1_list[i] /= args.world_size
            
            dist.all_reduce(all_spatial_sparsity_conv2_list[i])
            all_spatial_sparsity_conv2_list[i] /= args.world_size
            
            dist.all_reduce(all_spatial_sparsity_conv3_list[i])
            all_spatial_sparsity_conv3_list[i] /= args.world_size
        
    all_spatial_sparsity_conv3_list = torch.cat(all_spatial_sparsity_conv3_list, 0)
    all_spatial_sparsity_conv2_list = torch.cat(all_spatial_sparsity_conv2_list, 0)
    all_spatial_sparsity_conv1_list = torch.cat(all_spatial_sparsity_conv1_list, 0)
    all_channel_sparsity_list = torch.cat(all_channel_sparsity_list, 0)
    
    all_spatial_sparsity_conv3_list /= num_samples
    all_spatial_sparsity_conv2_list /= num_samples
    all_spatial_sparsity_conv1_list /= num_samples
    all_channel_sparsity_list /= num_samples
    
    
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % args.ngpus_per_node == 0):
        args.print_custom(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        args.print_custom(f'* Conv3 spatial sparsity: {all_spatial_sparsity_conv3_list}')
        args.print_custom(f'* Conv2 spatial sparsity: {all_spatial_sparsity_conv2_list}')
        args.print_custom(f'* Conv1 spatial sparsity: {all_spatial_sparsity_conv1_list}')
        args.print_custom(f'* channel sparsity: {all_channel_sparsity_list}')
        args.print_custom(f'*****************\n')
    
    all_density = torch.cat((all_spatial_sparsity_conv3_list.unsqueeze(0),
                             all_spatial_sparsity_conv2_list.unsqueeze(0),
                             all_spatial_sparsity_conv1_list.unsqueeze(0), 
                             all_channel_sparsity_list.unsqueeze(0)), 0)
    
    return top1.avg, top5.avg, losses.avg, act_rates.avg, FLOPs.avg, all_density.cpu().numpy()


def adjust_gs_temperature(epoch, step, len_epoch, args):
    if epoch >= args.t_last_epoch:
        return args.t_last
    else:
        T_total = args.t_last_epoch * len_epoch
        T_cur = epoch * len_epoch + step
        if args.temp_scheduler == 'exp':
            alpha = math.pow(args.t_last / args.t0, 1 / T_total)
            args.temp = math.pow(alpha, T_cur) * args.t0
        elif args.temp_scheduler == 'linear':
            args.temp = (args.t0 - args.t_last) * (1 - T_cur / T_total) + args.t_last
        else:
            args.temp = 0.5 * (args.t0-args.t_last) * (1 + math.cos(math.pi * T_cur / (T_total))) + args.t_last


class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, 'w') as f:
            f.write('==================== start running ===================='+'\n')
    def log(self, string, isprint=True):
        if isprint:
            print(string)
        with open(self.filename, 'a') as f:
            f.write(str(string)+'\n')


if __name__ == '__main__':
    main()
