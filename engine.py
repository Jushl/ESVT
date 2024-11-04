import os.path
import time
from torch.cuda.amp.grad_scaler import GradScaler
from util.optim.ema import ModelEMA
from util.optim.warmup import Warmup
from util.misc.logger import MetricLogger, SmoothedValue
import torch
from util.misc import dist_utils
import math
import sys
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import json
from dataset.UAV_EOD.val import DVSEvaluator
from models.ESVT.utils import check_target, check_empty_target
from models.ESVT.postprocessor.box_revert import box_revert
from util.misc.box_ops import box_cxcywh_to_xyxy


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    import cv2
    w, h, _ = img.shape
    c1, c2 = (int(x[0] * w), int(x[1] * h)), (int(x[2] * w), int(x[3] * h))
    cv2.rectangle(img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)


def vis_save(images, targets):
    import cv2
    import numpy as np
    from models.ESVT.box_ops import box_cxcywh_to_xyxy
    for batch in range(len(images)):
        image = np.array(images[batch].to('cpu'))
        image = np.transpose(image, (1, 2, 0)) * 255
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = targets[batch]['boxes']
        target_xyxy = box_cxcywh_to_xyxy(target)
        for i in range(len(target_xyxy)):
            xyxy = np.array(target_xyxy[i].to('cpu'))
            plot_one_box(xyxy, image, label='', color=(255, 0, 0), line_thickness=1)
        cv2.imshow('detect', image)
        cv2.waitKey(0)


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, max_norm, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    writer: SummaryWriter = kwargs.get('writer', None)
    ema: ModelEMA = kwargs.get('ema', None)
    scaler: GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler: Warmup = kwargs.get('lr_warmup_scheduler', None)

    for i, ((images, events, targets), indexes) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = images.to(device)
        events = events.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if indexes[-1][-1] % 100 == 0:
            pre_status = None
        else:
            pre_status = [(state[0].detach(), state[1].detach()) for state in status]

        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step)
        outputs, targets, status = model(events, targets=targets, pre_status=pre_status)
        if not check_empty_target(targets):
            continue
        else:
            loss_dict = criterion(outputs, targets, **metas)
            loss: torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            if ema is not None:
                ema.update(model)
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()
            loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
            loss_value = sum(loss_dict_reduced.values())

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)
            metric_logger.update(loss=loss_value, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            if writer and dist_utils.is_main_process():
                writer.add_scalar('Loss/total', loss_value.item(), global_step)
                for j, pg in enumerate(optimizer.param_groups):
                    writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
                for k, v in loss_dict_reduced.items():
                    writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessor, data_loader, base_ds, device, iou_types):
    model.eval()
    criterion.eval()

    dvs_evaluator = DVSEvaluator(iou_types, base_ds)
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    for (images, events, targets), indexes in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device)
        events = events.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if indexes[-1][-1] % 100 == 0:
            pre_status = None
        else:
            pre_status = [(state[0].detach(), state[1].detach()) for state in status]

        outputs, _, status = model(events, targets=targets, pre_status=pre_status)

        if not check_empty_target(targets):
            continue
        else:
            targets = check_target(targets)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessor(outputs, orig_target_sizes)

            for target in targets:
                boxes = box_cxcywh_to_xyxy(target['boxes'])
                orig_target_size = target['orig_size'].repeat(2)
                target['boxes'] = boxes * orig_target_size

            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            if dvs_evaluator is not None:
                dvs_evaluator.update(res)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if dvs_evaluator is not None:
        dvs_evaluator.synchronize_between_processes()

    if dvs_evaluator is not None:
        dvs_evaluator.accumulate()
        dvs_evaluator.summarize()

    stats = {}

    if dvs_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = dvs_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = dvs_evaluator.coco_eval['segm'].stats.tolist()

    return stats, dvs_evaluator


class Detection(object):
    def __init__(self, model, criterion, postprocessor, ema, optimizer, lr_scheduler, lr_warmup_scheduler,
                 data_loader_train, data_loader_val, base_ds, device, args):

        self.model = dist_utils.warp_model(model.to(device), sync_bn=True, find_unused_parameters=False)

        self.train_dataloader = data_loader_train
        self.val_dataloader = data_loader_val

        self.criterion = criterion
        self.postprocessor = postprocessor
        self.ema = ema
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_scheduler = lr_warmup_scheduler
        self.last_epoch = -1

        self.base_ds = base_ds
        self.device = device
        self.output_dir = args.output_dir
        self.args = args
        self.scaler = None

        output_dir = os.path.join(args.output_dir, args.model + '_' + args.backbone + '_' + args.streaming_type)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = self.build_writer()

        if args.resume:
            print(f'Resume checkpoint from {args.resume}')
            self.load_resume_state(args.resume)

    def load_state_dict(self, state):
        if 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            print('Load last_epoch')
        for k, v in self.__dict__.items():
            if hasattr(v, 'load_state_dict') and k in state:
                v = dist_utils.de_parallel(v)
                v.load_state_dict(state[k])
                print(f'Load {k}.state_dict')

            if hasattr(v, 'load_state_dict') and k not in state:
                print(f'Not load {k}.state_dict')

    def load_resume_state(self, path: str):
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state)

    def state_dict(self):
        state = {}
        state['date'] = datetime.now().isoformat()
        state['last_epoch'] = self.last_epoch
        for k, v in self.__dict__.items():
            if hasattr(v, 'state_dict'):
                v = dist_utils.de_parallel(v)
                state[k] = v.state_dict()
        return state

    def build_writer(self) -> SummaryWriter:
        if self.output_dir:
            writer = SummaryWriter(Path(self.output_dir) / 'summary')
        return writer

    def train(self):
        print("Start training")
        args = self.args
        best_stat = {'epoch': -1, }
        start_time = time.time()
        start_epcoch = self.last_epoch + 1

        for epoch in range(start_epcoch, args.epoches):
            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.base_ds,
                self.device,
                iou_types=['bbox']
            )

            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
                if best_stat['epoch'] == epoch and self.output_dir:
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')

            print(f'best_stat: {best_stat}')

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                    if coco_evaluator is not None:
                        (self.output_dir / 'eval').mkdir(exist_ok=True)
                        if "bbox" in coco_evaluator.coco_eval:
                            filenames = ['latest.pth']
                            if epoch % 50 == 0:
                                filenames.append(f'{epoch:03}.pth')
                            for name in filenames:
                                torch.save(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def val(self):
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.base_ds,
            self.device,
            iou_types=['bbox']
        )
        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        return








