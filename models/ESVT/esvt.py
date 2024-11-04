import torch.nn as nn
from models.ESVT.utils import check_x_target, check_empty_target


class ESVT(nn.Module):
    def __init__(self, backbone: nn.Module, encoder: nn.Module,  decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None, pre_status=None):
        x = self.backbone(x)
        x, status = self.encoder(x, pre_status)
        if not check_empty_target(targets):
            return x, targets, status
        x, targets = check_x_target(x, targets)
        x, targets = self.decoder(x, targets)
        return x, targets, status


    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
