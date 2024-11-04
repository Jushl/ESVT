from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torchvision.transforms.functional as F


class RevNorm(nn.Module):
    def __init__(self, num_features: int = 256, eps=1e-5, affine=True, subtract_last=False):
        super(RevNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class DWSConvLSTM2d(nn.Module):
    def __init__(self,
                 dim: int = 256,
                 dws_conv: bool = True,
                 dws_conv_only_hidden: bool = True,
                 dws_conv_kernel_size: int = 3,
                 cell_update_dropout: float = 0.):
        super().__init__()

        self.dim = dim
        xh_dim = dim * 2
        gates_dim = dim * 4
        conv3x3_dws_dim = dim if dws_conv_only_hidden else xh_dim
        self.conv3x3_dws = nn.Conv2d(in_channels=conv3x3_dws_dim,
                                     out_channels=conv3x3_dws_dim,
                                     kernel_size=dws_conv_kernel_size,
                                     padding=dws_conv_kernel_size // 2,
                                     groups=conv3x3_dws_dim) if dws_conv else nn.Identity()
        self.conv1x1 = nn.Conv2d(in_channels=xh_dim,
                                 out_channels=gates_dim,
                                 kernel_size=1)
        self.conv_only_hidden = dws_conv_only_hidden
        self.cell_update_dropout = nn.Dropout(p=cell_update_dropout)
        self.revnorm = RevNorm(num_features=dim, affine=True, subtract_last=False)

    def forward(self, x, hc_previous) -> Tuple[Tensor, Tensor]:
        if hc_previous is None:
            hidden = x
            cell = x
            hc_previous = (hidden, cell)

        h_t0, c_t0 = hc_previous
        B, C, H, W = x.shape
        c_t0, h_t0 = F.resize(c_t0, [H, W]), F.resize(h_t0, [H, W])

        B, C, H, W = h_t0.shape
        h_t0 = h_t0.flatten(2)
        h_t0 = h_t0.permute(0, 2, 1)
        h_t0 = self.revnorm(h_t0, 'norm')
        h_t0 = h_t0.permute(0, 2, 1)
        h_t0 = h_t0.reshape(B, C, H, W).contiguous()

        if self.conv_only_hidden:
            h_t0 = self.conv3x3_dws(h_t0)

        xh = torch.cat((x, h_t0), dim=1)

        if not self.conv_only_hidden:
            xh = self.conv3x3_dws(xh)

        mix = self.conv1x1(xh)
        cell_input, gates = torch.tensor_split(mix, [self.dim], dim=1)
        gates = torch.sigmoid(gates)
        forget_gate, input_gate, output_gate = torch.tensor_split(gates, 3, dim=1)

        cell_input = self.cell_update_dropout(torch.tanh(cell_input))

        c_t1 = forget_gate * c_t0 + input_gate * cell_input
        h_t1 = output_gate * torch.tanh(c_t1)

        B, C, H, W = h_t1.shape
        h_t1 = h_t1.flatten(2)
        h_t1 = h_t1.permute(0, 2, 1)
        h_t1 = self.revnorm(h_t1, 'denorm')
        h_t1 = h_t1.permute(0, 2, 1)
        h_t1 = h_t1.reshape(B, C, H, W).contiguous()

        return h_t1, c_t1

