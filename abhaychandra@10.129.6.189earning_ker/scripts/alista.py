"""ALISTA, Eq. (15), Liu et al., ICLR 2019; batch-major PyTorch implementation.

x[k+1] = soft_theta[k](x[k] + gamma[k] W.T (b - D x[k])).
Fixed D/W buffers, zero initialization, exactly two learned scalars per layer.
This is the dense sparse-code model, not a replacement for ConvLISTA on QDM.
"""
import math

import numpy as np
import torch
from torch import nn

from alista_weights import validate_dictionary


def soft_threshold(x, theta):
    return x.sign() * torch.relu(x.abs()-theta)


def support_threshold(x, theta, percent):
    """Authors' shrink_ss: strict percentile and threshold comparisons.

    TF 1.12 rounds an index into DESCENDING sorted values, unlike modern
    torch.quantile/NumPy at halfway ranks. Preserve its ties-to-even rule.
    Ties at the cutoff are not selected; q=0 is ordinary soft thresholding.
    Selection is per example, over coefficients, never across the batch.
    """
    if not math.isfinite(percent) or not 0 <= percent <= 100:
        raise ValueError("support percentage must be between 0 and 100")
    if percent == 0:
        return soft_threshold(x, theta)
    q = 100.-percent
    rank = round((x.shape[-1]-1)*(1.-q/100.))
    cutoff = torch.sort(x.detach().abs(), dim=-1, descending=True).values[..., rank:rank+1]
    selected = (x.abs() > cutoff) & (x.abs() > theta)
    return torch.where(selected, x, soft_threshold(x, theta))


class ALISTA(nn.Module):
    def __init__(self, dictionary, weights, *, layers=16, initial_lambda=.4,
                 support_percent=0., max_support_percent=13.,
                 mode="alista", dtype=torch.float32):
        super().__init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be float32 or float64")
        d = validate_dictionary(dictionary)
        if np.iscomplexobj(weights):
            raise ValueError("W must be real")
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != d.shape or not np.isfinite(w).all():
            raise ValueError("W must be finite and have D's shape")
        if not isinstance(layers, int) or layers < 1:
            raise ValueError("layers must be a positive integer")
        if not math.isfinite(initial_lambda) or initial_lambda < 0:
            raise ValueError("initial_lambda must be finite and nonnegative")
        if (not all(math.isfinite(p) and 0 <= p <= 100 for p in (support_percent, max_support_percent))):
            raise ValueError("support percentages must be between 0 and 100")
        if mode not in ("alista", "learned-ista"):
            raise ValueError("unknown model mode")
        if mode == "alista" and not np.allclose(np.sum(d*w, axis=0), 1., rtol=1e-5, atol=1e-7):
            raise ValueError("ALISTA W must satisfy diag(W.T D)=1")
        if mode == "learned-ista" and not np.array_equal(d, w):
            raise ValueError("learned-ista control requires W=D")
        self.layers, self.mode = layers, mode
        self.support_percent, self.max_support_percent = support_percent, max_support_percent
        self.initial_lambda = initial_lambda
        scale = 1.001 * float(np.linalg.norm(d, ord=2)**2)
        self.register_buffer("D", torch.tensor(d, dtype=dtype))
        self.register_buffer("W", torch.tensor(w, dtype=dtype))
        # Scalar ParameterLists allow genuine layerwise freezing during training.
        gamma = 1. if mode == "alista" else 1./scale
        self.steps = nn.ParameterList([nn.Parameter(torch.tensor(gamma, dtype=dtype)) for _ in range(layers)])
        self.thresholds = nn.ParameterList([nn.Parameter(torch.tensor(initial_lambda/scale, dtype=dtype)) for _ in range(layers)])

    def iterates(self, b, *, depth=None):
        depth = self.layers if depth is None else depth
        if not isinstance(depth, int) or not 0 <= depth <= self.layers:
            raise ValueError("depth must lie between zero and the trained layer count")
        if b.ndim != 2 or b.shape[1] != self.D.shape[0] or b.shape[0] < 1:
            raise ValueError("expected nonempty b [batch, measurements]")
        if b.dtype != self.D.dtype or b.device != self.D.device:
            raise ValueError("b and model must share dtype and device")
        x = b.new_zeros((b.shape[0], self.D.shape[1]))
        yield x
        for k in range(depth):
            residual = b - x @ self.D.T
            z = x + self.steps[k] * (residual @ self.W)
            theta = self.thresholds[k].clamp_min(0.)
            percent = min((k+1)*self.support_percent, self.max_support_percent)
            x = support_threshold(z, theta, percent)
            yield x

    def forward(self, b, *, depth=None):
        for x in self.iterates(b, depth=depth):
            pass
        return x

    @torch.no_grad()
    def project_parameters(self):
        for theta in self.thresholds:
            theta.clamp_(min=0.)

    def configuration(self):
        return dict(layers=self.layers, initial_lambda=self.initial_lambda,
                    support_percent=self.support_percent,
                    max_support_percent=self.max_support_percent, mode=self.mode)


def save_checkpoint(path, model, metadata):
    """Self-contained inference checkpoint; no pickle objects beyond tensors/primitives."""
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save(dict(format="alista-checkpoint-v1", config=model.configuration(),
                    state_dict=state, metadata=metadata), path)


def load_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != "alista-checkpoint-v1":
        raise ValueError("unsupported ALISTA checkpoint")
    state = saved["state_dict"]
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("checkpoint contains nonfinite tensors")
    model = ALISTA(state["D"].numpy(), state["W"].numpy(),
                   dtype=state["D"].dtype, **saved["config"])
    model.load_state_dict(state, strict=True)
    if any(float(t.detach()) < 0 for t in model.thresholds):
        raise ValueError("checkpoint contains a negative threshold")
    return model.to(device).eval(), saved["metadata"]
