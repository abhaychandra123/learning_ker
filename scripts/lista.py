"""Physics-preserving convolutional LISTA with tied learned corrections.

x_(k+1) = shrink(x_k - step A*(A x_k-b) + C_b b + C_x x_k, theta).
Thus W_b = step A* + C_b and W_x = I-step A*A+C_x are learned linear
maps. C_b/C_x are zero-padded bias-free Conv2d maps, tied across layers.
This supervised reconstruction variant is not the original solver-distillation
training protocol. There is no convergence guarantee after weights are learned.
"""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from torch_inverse import PaddedFFT, shrink


class ConvLISTA(nn.Module):
    def __init__(self, transfer, shape, pad, lam, *, layers=8, kernel_size=9,
                 initial="blurred", dtype=torch.float32, checkpoint_layers=False):
        super().__init__()
        if not isinstance(layers, int) or layers < 1:
            raise ValueError("layers must be a positive integer")
        if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("kernel_size must be a positive odd integer")
        if initial not in ("blurred", "zero") or not math.isfinite(lam) or lam < 0:
            raise ValueError("invalid initialization or lambda")
        self.operator = PaddedFFT(transfer, shape, pad, dtype)
        self.layers, self.initial = layers, initial
        self.kernel_size, self.checkpoint_layers = kernel_size, checkpoint_layers
        self.step = .99/self.operator.bound
        self.input_correction = nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2,
                                          bias=False, dtype=dtype)
        self.state_correction = nn.Conv2d(1, 1, kernel_size, padding=kernel_size//2,
                                          bias=False, dtype=dtype)
        nn.init.zeros_(self.input_correction.weight)
        nn.init.zeros_(self.state_correction.weight)
        # Projected nonnegative parameter, rather than softplus of a tiny
        # initialization whose derivative would nearly freeze its learning.
        self.threshold = nn.Parameter(torch.tensor(lam*self.step, dtype=dtype))

    def layer(self, x, b, correction_b):
        v = (x - self.step*self.operator.adjoint(self.operator(x)-b)
             + correction_b + self.state_correction(x))
        return shrink(v, self.threshold.clamp_min(0))

    def iterates(self, b):
        """Yield x_0 ... x_layers; no extrapolation beyond trained depth."""
        x = b.clone() if self.initial == "blurred" else torch.zeros_like(b)
        correction_b = self.input_correction(b)
        yield x
        for _ in range(self.layers):
            if self.checkpoint_layers and self.training and torch.is_grad_enabled():
                x = checkpoint(self.layer, x, b, correction_b, use_reentrant=False)
            else:
                x = self.layer(x, b, correction_b)
            yield x

    def forward(self, b):
        x = None
        for x in self.iterates(b):
            pass
        return x

    @torch.no_grad()
    def project_parameters(self):
        self.threshold.clamp_(min=0)


def load_checkpoint(path, device="cpu"):
    """Load a local trusted state-dict checkpoint with all forward parameters."""
    import numpy as np
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state = saved["state_dict"]
    transfer = (state["operator.transfer_real"].numpy().astype(np.float64)
                + 1j*state["operator.transfer_imag"].numpy().astype(np.float64))
    model = ConvLISTA(transfer, saved["shape"], saved["pad"],
                      saved["lambda_normalized"], layers=saved["layers"],
                      kernel_size=saved["kernel_size"], initial=saved["initial"])
    model.load_state_dict(state, strict=True)
    # The original bound was evaluated from the float64 fit, before the
    # transfer was stored as float32 buffers. Preserve its exact step size.
    bound = saved["norm_bound"]
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError("invalid saved norm bound")
    model.operator.bound = bound
    model.step = .99/bound
    model.to(device).eval()
    return model, saved


if __name__ == '__main__':
    # The model remains independently importable; this is also a standalone CLI.
    from compare_lista import main
    main()
