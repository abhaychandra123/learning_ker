"""Layer-specific correction weights for a controlled ConvLISTA capacity study."""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from combined_lista import SharedConvLISTA
from torch_inverse import shrink


class UntiedConvLISTA(nn.Module):
    def __init__(self, lam, layers=16, kernel_size=9, initial='blurred', checkpoint_layers=False):
        super().__init__()
        if (layers < 1 or kernel_size < 1 or kernel_size % 2 != 1
                or not math.isfinite(lam) or lam < 0 or initial not in ('blurred','zero')):
            raise ValueError('Invalid model configuration')
        self.layers, self.initial, self.checkpoint_layers = layers, initial, checkpoint_layers
        self.input_corrections = nn.ModuleList([
            nn.Conv2d(1,1,kernel_size,padding=kernel_size//2,bias=False) for _ in range(layers)])
        self.state_corrections = nn.ModuleList([
            nn.Conv2d(1,1,kernel_size,padding=kernel_size//2,bias=False) for _ in range(layers)])
        self.thresholds = nn.Parameter(torch.full((layers,),lam))
        for layer in (*self.input_corrections, *self.state_corrections):
            nn.init.zeros_(layer.weight)

    def forward(self, b, operator):
        x = b.clone() if self.initial=='blurred' else torch.zeros_like(b)
        step = .99/operator.bound
        for k in range(self.layers):
            # Bind k now: backward checkpoint recomputation must use this layer.
            def layer(x, b, index=k):
                v = (x-step*operator.adjoint(operator(x)-b)
                     + self.input_corrections[index](b)+self.state_corrections[index](x))
                return shrink(v,step*self.thresholds[index].clamp_min(0))
            if self.checkpoint_layers and self.training and torch.is_grad_enabled():
                x = checkpoint(layer,x,b,use_reentrant=False)
            else:
                x = layer(x,b)
        return x

    @torch.no_grad()
    def project_parameters(self):
        self.thresholds.clamp_(min=0)


def build_model(config):
    options = dict(config)
    sharing = options.pop('sharing')
    if sharing not in ('tied','untied'):
        raise ValueError('Unknown parameter sharing')
    return (SharedConvLISTA if sharing=='tied' else UntiedConvLISTA)(**options)


def load_model(path, device='cpu'):
    saved = torch.load(path,map_location='cpu',weights_only=True)
    model = build_model(saved['model_config'])
    model.load_state_dict(saved['model_state'])
    return model.to(device).eval(), saved
