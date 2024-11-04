import torch
from models.ESVT.backbone.utils import IntermediateLayerGetter


class ResNet(torch.nn.Module):
    def __init__(self,
                 name,
                 return_layers=['layer2', 'layer3', 'layer4'],
                 pretrained=False,
                 exportable=True,
                 features_only=True,
                 **kwargs) -> None:

        super().__init__()

        name = 'resnet' + name

        import timm
        model = timm.create_model(
            name,
            pretrained=pretrained, 
            exportable=exportable, 
            features_only=features_only,
            **kwargs
        )

        assert set(return_layers).issubset(model.feature_info.module_name()), \
            f'return_layers should be a subset of {model.feature_info.module_name()}'
        
        # self.model = model
        self.model = IntermediateLayerGetter(model, return_layers)

        return_idx = [model.feature_info.module_name().index(name) for name in return_layers]
        self.strides = [model.feature_info.reduction()[i] for i in return_idx]
        self.channels = [model.feature_info.channels()[i] for i in return_idx]
        self.return_idx = return_idx
        self.return_layers = return_layers

    def forward(self, x: torch.Tensor): 
        outputs = self.model(x)
        return outputs
