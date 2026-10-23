"""One tied ConvLISTA correction model with an explicit operator per pair."""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from torch_inverse import shrink


class SharedConvLISTA(nn.Module):
    """x+ = shrink(x - s A*(Ax-b) + C_b b + C_x x, s * theta).

    s=.99/||H||_inf^2 varies with the known pair operator. The learned scalar
    theta is in globally normalized field units. With zero convolutions and
    theta=lambda_uT/scale this exactly reproduces ISTA for EVERY pair.
    All 163 parameters (9x9 case) are shared across pairs and layers.
    """
    def __init__(self, lam, layers=16, kernel_size=9, initial='blurred',
                 checkpoint_layers=False, dtype=torch.float32):
        super().__init__()
        if (layers < 1 or kernel_size < 1 or kernel_size % 2 != 1
                or not math.isfinite(lam) or lam < 0 or initial not in ('blurred', 'zero')):
            raise ValueError('Invalid shared LISTA configuration')
        self.layers, self.initial = layers, initial
        self.checkpoint_layers = checkpoint_layers
        self.input_correction = nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2,
                                         bias=False, dtype=dtype)
        self.state_correction = nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2,
                                         bias=False, dtype=dtype)
        nn.init.zeros_(self.input_correction.weight)
        nn.init.zeros_(self.state_correction.weight)
        self.threshold = nn.Parameter(torch.tensor(lam, dtype=dtype))

    def forward(self, b, operator):
        x = b.clone() if self.initial == 'blurred' else torch.zeros_like(b)
        correction_b = self.input_correction(b)
        step = .99/operator.bound

        def layer(x, b, correction_b):
            v = (x - step*operator.adjoint(operator(x)-b)
                 + correction_b + self.state_correction(x))
            return shrink(v, step*self.threshold.clamp_min(0))

        for _ in range(self.layers):
            if self.checkpoint_layers and self.training and torch.is_grad_enabled():
                x = checkpoint(layer, x, b, correction_b, use_reentrant=False)
            else:
                x = layer(x, b, correction_b)
        return x

    @torch.no_grad()
    def project_parameters(self):
        self.threshold.clamp_(min=0)


def load_shared_checkpoint(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = SharedConvLISTA(**saved['model_config'])
    model.load_state_dict(saved['model_state'])
    return model.to(device).eval(), saved
