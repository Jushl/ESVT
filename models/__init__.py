from models.ESVT import build_ESVT
from models.ESVT import build_ESVT_criterion
from models.ESVT import build_ESVT_postprocessor


def build_model(args):
    if args.model == 'ESVT':
        return build_ESVT(args)
    else:
        assert args.model in ['ESVT', 'RTDETR', 'RTDETRv2', 'YOLOv10', 'YOLOv8', 'YOLOv6', 'YOLOv5', 'YOLOv3']


def build_criterion(args):
    if args.model == 'ESVT':
        return build_ESVT_criterion(args)
    else:
        assert args.model in ['ESVT', 'RTDETR', 'RTDETRv2', 'YOLOv10', 'YOLOv8', 'YOLOv6', 'YOLOv5', 'YOLOv3']


def build_postprocessor(args):
    if args.model == 'ESVT':
        return build_ESVT_postprocessor(args)
    else:
        assert args.model in ['ESVT', 'RTDETR', 'RTDETRv2', 'YOLOv10', 'YOLOv8', 'YOLOv6', 'YOLOv5', 'YOLOv3']


