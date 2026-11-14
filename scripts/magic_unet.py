"""MAGIC-UNet (Reed et al. 2024, arXiv 2407.14553) in PyTorch, plus the helpers to use it here.

The layer graph copies the authors' released trained model,
MAGIC_UNet_64x64_50um (Harvard Dataverse doi:10.7910/DVN/SD6PVP, keras_metadata.pb),
recorded in tests/data/magic_unet_released_layers.json:

    encoder   3 x [conv3x3 -> BN -> ReLU] at 64, 128, 256 channels, 2x2 average pooling
              (stride 2) between levels, then 3 x [conv3x3 -> BN -> ReLU] at 512.
    branches  two independent decoders, one for Jx and one for Jy. Each level:
              nearest 2x upsampling -> conv3x3 (no BN) -> concatenate [skip, up]
              -> BN -> ReLU -> 2 x [conv3x3 -> BN -> ReLU]. Skips come from the 256 and 128
              levels; the 64-level skip is omitted, as the paper says. Then
              conv3x3 (2 channels) -> conv3x3 (1 channel), both linear.

Keras defaults reproduced: He-normal truncated initialisation (Keras VarianceScaling,
truncated at 2 standard deviations), zero biases, BatchNormalization momentum 0.99
(PyTorch momentum 0.01) and epsilon 1e-3, zero 'same' padding.

Units here: B in uT, sheet current J in A/m, lengths in um. The network predicts the
equivalent sheet current at the zero-offset plane (see README_MAGIC_UNET.md).
"""
import math

import numpy as np
import torch
from torch import nn

from simulate_fields import MU0_UT

KERAS_TRUNCATED_STD = 0.87962566103423978   # std of a unit normal truncated at +-2
BN_MOMENTUM, BN_EPS = 0.01, 1e-3            # Keras momentum 0.99, epsilon 1e-3


def conv3x3(cin, cout):
    return nn.Conv2d(cin, cout, 3, padding=1, bias=True)


def bn(c):
    return nn.BatchNorm2d(c, eps=BN_EPS, momentum=BN_MOMENTUM)


class CBR(nn.Sequential):
    """conv3x3 -> BatchNorm -> ReLU."""

    def __init__(self, cin, cout):
        super().__init__(conv3x3(cin, cout), bn(cout), nn.ReLU(inplace=True))


class UpLevel(nn.Module):
    """Upsample, conv, optional skip concatenation, BN+ReLU, then two CBR blocks."""

    def __init__(self, cin, cout, skip):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = conv3x3(cin, cout)
        merged = cout + skip
        self.norm = nn.Sequential(bn(merged), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(CBR(merged, cout), CBR(cout, cout))
        self.skip = skip

    def forward(self, x, skip=None):
        x = self.conv(self.up(x))
        if self.skip:
            x = torch.cat([skip, x], dim=1)     # Keras order: [encoder, upsampled]
        return self.blocks(self.norm(x))


class Branch(nn.Module):
    """One decoder; outputs a single current component."""

    def __init__(self):
        super().__init__()
        self.up3 = UpLevel(512, 256, skip=256)
        self.up2 = UpLevel(256, 128, skip=128)
        self.up1 = UpLevel(128, 64, skip=0)
        self.head = nn.Sequential(conv3x3(64, 2), conv3x3(2, 1))

    def forward(self, bottom, s2, s3):
        x = self.up3(bottom, s3)
        x = self.up2(x, s2)
        return self.head(self.up1(x))


class MAGICUNet(nn.Module):
    """Input [N, 3, H, W] (Bx, By, Bz), output [N, 2, H, W] (Jx, Jy); H, W divisible by 8."""

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(CBR(3, 64), CBR(64, 64), CBR(64, 64))
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128), CBR(128, 128))
        self.enc3 = nn.Sequential(CBR(128, 256), CBR(256, 256), CBR(256, 256))
        self.bottom = nn.Sequential(CBR(256, 512), CBR(512, 512), CBR(512, 512))
        self.pool = nn.AvgPool2d(2, stride=2)
        self.jx, self.jy = Branch(), Branch()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                std = math.sqrt(2.0 / fan_in) / KERAS_TRUNCATED_STD
                nn.init.trunc_normal_(m.weight, 0.0, std, -2 * std, 2 * std)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, b):
        s1 = self.enc1(b)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        bottom = self.bottom(self.pool(s3))
        return torch.cat([self.jx(bottom, s2, s3), self.jy(bottom, s2, s3)], dim=1)


# ----------------------------------------------------------------- normalisation

def alpha(b):
    """Per-image scale of the paper: max over pixels of |B| (vector norm). b: [..., 3, H, W]."""
    return b.pow(2).sum(-3).sqrt().amax((-2, -1))


# ----------------------------------------------------------------- physics

def solenoidal_sheet(kx, ky, dx, depth=0.0):
    """Divergence-free part of a sheet current, upward-continued by `depth` um.

    With depth = d this is the equivalent sheet at the zero-offset plane: it gives the
    same field above that plane as the true sheet d below it. Same projection as
    simulate_fields.fields_from_current.
    """
    ny, nx = kx.shape
    wx = 2 * np.pi * np.fft.rfftfreq(nx, dx)[None, :]
    wy = 2 * np.pi * np.fft.fftfreq(ny, dx)[:, None]
    k = np.hypot(wx, wy)
    k2 = k ** 2
    k2[0, 0] = 1.0
    fx, fy = np.fft.rfft2(kx), np.fft.rfft2(ky)
    longitudinal = (wx * fx + wy * fy) / k2
    decay = np.exp(-k * depth)
    fx, fy = (fx - wx * longitudinal) * decay, (fy - wy * longitudinal) * decay
    fx[0, 0] = fy[0, 0] = 0
    return np.fft.irfft2(fx, s=(ny, nx)), np.fft.irfft2(fy, s=(ny, nx))


def biot_savart_sheet(jx, jy, dx, height):
    """Thin-sheet Biot-Savart for ANY sheet current (no divergence-free projection).

    Same formulas and COMSOL component frame as simulate_fields.fields_from_current,
    which additionally projects the current; for a divergence-free current both agree.
    Without the projection the field is linear in the current, so a current cropped to
    the image gives exactly the field of the in-image current.
    """
    ny, nx = jx.shape
    wx = 2 * np.pi * np.fft.rfftfreq(nx, dx)[None, :]
    wy = 2 * np.pi * np.fft.fftfreq(ny, dx)[:, None]
    k = np.hypot(wx, wy)
    safe = k.copy()
    safe[0, 0] = 1.0
    fx, fy = np.fft.rfft2(jx), np.fft.rfft2(jy)
    fz = 1j * (wy * fx - wx * fy) / safe
    decay = 0.5 * MU0_UT * np.exp(-k * height)
    return [np.fft.irfft2(f * decay, s=(ny, nx)) for f in (fx, fy, fz)]


def field_from_sheet(j, height, dx=0.1, pad=700):
    """[3, H, W] field in uT at `height` um above a [2, H, W] sheet current in A/m.

    Zero padding outside the image, then thin-sheet Biot-Savart in the simulator's
    component frame, so signs match the training data and COMSOL.
    """
    jx = np.pad(np.asarray(j[0], np.float64), pad)
    jy = np.pad(np.asarray(j[1], np.float64), pad)
    f = biot_savart_sheet(jx, jy, dx, height)
    crop = (slice(pad, pad + j.shape[-2]), slice(pad, pad + j.shape[-1]))
    return np.stack([c[crop] for c in f])


# ----------------------------------------------------------------- tiled inference

def tile_starts(size, tile, stride):
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def tile_owner(size, tile, starts):
    """For each pixel, the start of the covering tile whose centre is nearest.

    This divides every overlap equally between neighbouring tiles and discards the
    duplicate predictions, as in Fig. 5c of the paper.
    """
    pos = np.arange(size)[:, None]
    s = np.asarray(starts)[None, :]
    covers = (pos >= s) & (pos < s + tile)
    dist = np.where(covers, np.abs(pos - (s + (tile - 1) / 2)), np.inf)
    return np.asarray(starts)[dist.argmin(1)]


@torch.no_grad()
def predict_tiled(model, b, beta, tile=128, stride=64, batch=32, device='cpu'):
    """Sheet current [2, H, W] (A/m) from a full field b [3, H, W] (uT).

    Each tile is normalised by its own alpha, as in the authors' inference code; the
    prediction is multiplied back by alpha / beta.
    """
    model.eval()
    _, h, w = b.shape
    rows, cols = tile_starts(h, tile, stride), tile_starts(w, tile, stride)
    own_r, own_c = tile_owner(h, tile, rows), tile_owner(w, tile, cols)
    full = torch.as_tensor(np.asarray(b, np.float32))
    out = np.zeros((2, h, w), np.float64)
    jobs = [(r, c) for r in rows for c in cols]
    for i in range(0, len(jobs), batch):
        part = jobs[i:i + batch]
        x = torch.stack([full[:, r:r + tile, c:c + tile] for r, c in part])
        a = alpha(x).clamp_min(1e-30)
        y = model((x / a[:, None, None, None]).to(device)).cpu().double()
        y = y * a.double()[:, None, None, None] / beta
        for (r, c), yt in zip(part, y.numpy()):
            mr = own_r[r:r + tile] == r
            mc = own_c[c:c + tile] == c
            out[:, r:r + tile, c:c + tile][:, mr[:, None] & mc[None, :]] = yt[:, mr[:, None] & mc[None, :]]
    return out


# ----------------------------------------------------------------- metrics

def relative_l2(pred, truth):
    return float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))


def _gaussian_window(size=11, sigma=1.5):
    x = np.arange(size) - (size - 1) / 2
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


def ssim(pred, truth, data_range=None):
    """Mean SSIM (Wang et al. 2004): 11x11 Gaussian window, sigma 1.5, K1 0.01, K2 0.03.

    Valid-region mean; data_range defaults to the truth's max - min.
    """
    pred, truth = np.asarray(pred, np.float64), np.asarray(truth, np.float64)
    if data_range is None:
        data_range = float(truth.max() - truth.min())
    w = _gaussian_window()
    m = w.shape[0]
    shape = (pred.shape[0] + m - 1, pred.shape[1] + m - 1)
    fw = np.fft.rfft2(w, shape)

    def f(a):   # 'valid' convolution with the window, NumPy only
        full = np.fft.irfft2(np.fft.rfft2(a, shape) * fw, shape)
        return full[m - 1:a.shape[0], m - 1:a.shape[1]]
    mu_x, mu_y = f(pred), f(truth)
    sxx = f(pred * pred) - mu_x ** 2
    syy = f(truth * truth) - mu_y ** 2
    sxy = f(pred * truth) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return float(s.mean())


def psnr(pred, truth, data_range=None):
    if data_range is None:
        data_range = float(truth.max() - truth.min())
    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    return float(20 * np.log10(data_range / rmse))
