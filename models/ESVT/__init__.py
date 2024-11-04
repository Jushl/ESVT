from models.ESVT.esvt import ESVT
from models.ESVT.backbone.hgnetv2 import HGNetv2
from models.ESVT.backbone.resnet import ResNet
from models.ESVT.encoder.hybrid_encoder import HybridEncoder
from models.ESVT.decoder.rtdetrv2_decoder import RTDETRTransformerv2
from models.ESVT.criterion.rtdetrv2_criterion import RTDETRCriterionv2
from models.ESVT.criterion.matcher import HungarianMatcher
from models.ESVT.postprocessor.rtdetr_postprocessor import RTDETRPostProcessor


def build_ESVT(args):
    if args.model_type == 'event':

        if args.backbone[:-1] == 'hgnetv2':
            return ESVT(
                backbone=HGNetv2(name=args.backbone[-1], pretrained=args.backbone_pretrained),
                encoder=HybridEncoder(name=args.transformer_scale[-1], streaming_type=args.streaming_type),
                decoder=RTDETRTransformerv2(name=args.transformer_scale[-1], dataset=args.dataset)
            )

        elif args.backbone[:-2] == 'resnet':
            return ESVT(
                backbone=ResNet(name=args.backbone[-2:], pretrained=args.backbone_pretrained),
                encoder=HybridEncoder(backbone_name=args.backbone[-2:], name=args.transformer_scale[-1],
                                      streaming_type=args.streaming_type),
                decoder=RTDETRTransformerv2(name=args.transformer_scale[-1], dataset=args.dataset)
            )

    # TODO 还未实现其他模态, 稍后实现
    elif args.model_type == 'image':
        pass
    elif args.model_type == 'multimodel':
        pass


def build_ESVT_criterion(args):
    return RTDETRCriterionv2(
        matcher=HungarianMatcher(weight_dict=args.matcher_weight_dict, use_focal_loss=args.use_focal_loss),
        weight_dict=args.criterion_weight_dict,
        losses=args.criterion_losses,
        dataset=args.dataset
    )


def build_ESVT_postprocessor(args):
    return RTDETRPostProcessor(dataset=args.dataset,
                               use_focal_loss=args.use_focal_loss,
                               num_top_queries=args.num_top_queries)
