import argparse
import torch
from torch.utils.data import DataLoader
import dataset.UAV_EOD.samplers as samplers
from dataset import build_dataset
from models import build_model, build_criterion, build_postprocessor
from util.optim.ema import build_ema
from dataset.UAV_EOD.collate_fn import BatchImageCollateFuncion
from util.optim.optim import build_optim
from util.misc import dist_utils
from util.misc import target_to_coco_format as TTC
from engine import Detection
from util.optim.warmup import LinearWarmup


def get_args_parser():
    parser = argparse.ArgumentParser('Event-based Streaming Object Detection', add_help=False)

    # 训练参数
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--batch_size_val', default=1, type=int)
    parser.add_argument('--epoches', default=72, type=int)
    parser.add_argument('--lr_drop_list', default=[70])
    parser.add_argument('--test_only', default=False)
    parser.add_argument('--resume', default=r'', type=str)

    # 数据集
    parser.add_argument('--dataset', default='UAV-EOD', type=str)
    parser.add_argument('--dataset_path', default=r'path to dataset', type=str)
    parser.add_argument('--scales',
                        default=[480, 512, 544, 576, 608, 640, 640, 640, 672, 704, 736, 768, 800])

    # 模型
    parser.add_argument('--model', default='ESVT', type=str, choices=['ESVT', ])
    parser.add_argument('--model_type', default='event', type=str)
    parser.add_argument('--event_rep', default='voxel', type=str, choices=['voxel', ])

    # backbone
    parser.add_argument('--backbone', default='resnet18', type=str, choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--backbone_pretrained', default=False)

    # encoder和decoder
    parser.add_argument('--transformer_scale', default='hybrid_transformer_L', type=str,
                        choices=['hybrid_transformer_L', 'hybrid_transformer_X', 'hybrid_transformer_H'])
    parser.add_argument('--streaming_type', default='lstm', type=str, choices=['stc', 'lstm'])
    parser.add_argument('--num_top_queries', default=300, type=int)

    # 训练设备
    parser.add_argument('--output_dir', default='outputs/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--checkpoint_freq', default=10, type=int)

    # 优化器
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lr_drop', default=71, type=int)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--amp', default=True)
    parser.add_argument('--optimizer', default='AdamW', choices=['AdamW', 'Adam', 'SGD'])

    # 损失函数
    parser.add_argument('--use_focal_loss', default=True)
    parser.add_argument('--matcher_weight_dict', default={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2})
    parser.add_argument('--criterion_weight_dict', default={'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2})
    parser.add_argument('--criterion_losses', default=['vfl', 'boxes', ])
    parser.add_argument('--clip_max_norm', default=0.1, type=float)

    # 分布式训练参数设置
    parser.add_argument('--distributed', default=False)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--print-method', default='builtin', type=str)
    parser.add_argument('--print-rank', default=0, type=int)
    parser.add_argument('--local-rank', type=int)
    parser.add_argument('--print_freq', default=10, type=int)
    return parser


def main(args):
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    device = torch.device(args.device)

    dataset_train = build_dataset(mode='train', args=args)
    dataset_val = build_dataset(mode='val', args=args)

    if args.distributed:
        sampler_train = samplers.DistributedSampler(dataset_train, args.batch_size)
        sampler_val = samplers.DistributedSampler(dataset_val, args.batch_size)
    else:
        sampler_train = samplers.StreamingSampler(dataset_train, args.batch_size)
        sampler_val = samplers.StreamingSampler(dataset_val, args.batch_size)

    data_loader_train = DataLoader(dataset_train, args.batch_size, sampler=sampler_train, drop_last=True,
                                   num_workers=args.num_workers, pin_memory=True,
                                   collate_fn=BatchImageCollateFuncion(scales=args.scales, stop_epoch=args.epoches))
    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size_val, sampler=sampler_val, drop_last=False,
                                 num_workers=args.num_workers, collate_fn=BatchImageCollateFuncion())

    base_ds_val = TTC.target_to_coco_format(data_loader_val)

    model = build_model(args).to(device)
    ema = build_ema(model)
    criterion = build_criterion(args).to(device)
    postprocessor = build_postprocessor(args).to(device)
    print('number of params:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = build_optim(model, args)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.lr_drop_list, gamma=0.1)
    lr_warmup_scheduler = LinearWarmup(lr_scheduler=lr_scheduler, warmup_duration=1000)

    det = Detection(model,
                    criterion,
                    postprocessor,
                    ema,
                    optimizer,
                    lr_scheduler,
                    lr_warmup_scheduler,
                    data_loader_train,
                    data_loader_val,
                    base_ds_val,
                    device,
                    args)

    if args.test_only:
        det.val()
    else:
        det.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ESOD training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
