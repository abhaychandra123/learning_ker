"""FISTA-Net for standoff deblurring.

Xiang, Dong and Yang, "FISTA-Net: Learning A Fast Iterative Shrinkage Thresholding
Network for Inverse Problems in Imaging". Equations cited are from that paper.

    r_k = y_k - mu_k G(y_k)                     gradient module (8a)
    x_k = r_k + F~(soft(F(r_k), theta_k))       proximal module (8b)
    y_(k+1) = x_k + rho_k (x_k - x_(k-1))       momentum module (8c)

Taken from the authors' M5FISTANet.py, by choice: the gradient step and the
10-layer convolutional block with bias, which gives the paper's 74,599 parameters.

    G(y) = (A^T A + 0.001 L^T L)^-1 (A^T A y - A^T b - 0.001 L^T L y)

Taken from the paper: Eq. 15 schedules, Eq. 16 initial image and Eq. 14 loss. Start
values come from either source; `code` restates the file's values for k counted from
1, which gives the same schedules. The code's final ReLU is left out because magnetic
fields are signed.
A is the fixed exp(-kz) filter and L the row and column difference operator; both
are diagonal in Fourier space on the padded canvas.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from csc_lista import transfer

LAMBDA_0 = 0.001
# Section III-C values, and those of M5FISTANet.py with c = b - w for k counted from 1.
START = dict(paper=dict(w1=-.5, w2=-.2, w3=1., c1=-2., c2=-1., c3=0.),
             code=dict(w1=-.2, w2=-.5, w3=.5, c1=.3, c2=-1.5, c3=-.5))


def difference_power(n, dtype, device):
    """Fourier symbol of L^T L, with L the row and column differences."""
    fy = torch.fft.fftfreq(n, dtype=dtype, device=device)[:, None]
    fx = torch.fft.rfftfreq(n, dtype=dtype, device=device)[None, :]
    return 4 * (torch.sin(torch.pi * fx) ** 2 + torch.sin(torch.pi * fy) ** 2)


class ProximalBlock(nn.Module):
    """conv_D, four forward convs, soft threshold, four backward convs, conv_G."""

    def __init__(self, features=32):
        super().__init__()
        conv = lambda a, b: nn.Conv2d(a, b, 3, padding=1)
        self.conv_d = conv(1, features)
        self.forward_layers = nn.ModuleList(conv(features, features) for _ in range(4))
        self.backward_layers = nn.ModuleList(conv(features, features) for _ in range(4))
        self.conv_g = conv(features, 1)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def transform(self, x):
        for i, layer in enumerate(self.forward_layers):
            x = layer(x)
            if i < 3:
                x = F.relu(x)
        return x

    def inverse(self, x):
        for i, layer in enumerate(self.backward_layers):
            x = layer(x)
            if i < 3:
                x = F.relu(x)
        return x

    def forward(self, r, theta):
        start = self.conv_d(r[:, None])
        coded = self.transform(start)
        kept = torch.sign(coded) * F.relu(torch.abs(coded) - theta)
        output = r + self.conv_g(self.inverse(kept))[:, 0]
        symmetry = ((self.inverse(coded) - start) ** 2).mean()
        return output, symmetry, coded.abs().mean()


class FISTANet(nn.Module):
    def __init__(self, dx, dz, layers=7, use_checkpoint=True, start='paper'):
        super().__init__()
        self.dx, self.dz, self.layers, self.use_checkpoint = dx, dz, layers, use_checkpoint
        self.block = ProximalBlock()
        for name, value in START[start].items():
            setattr(self, name, nn.Parameter(torch.tensor([value])))

    def schedules(self, k):
        """Eq. 15 for iteration k, counted from 1."""
        mu = F.softplus(self.w1 * k + self.c1)
        theta = F.softplus(self.w2 * k + self.c2)
        top = F.softplus(self.w3 * k + self.c3)
        return mu, theta, (top - F.softplus(self.w3 + self.c3)) / top

    def spectra(self, n, reference):
        h = transfer(n, self.dx, self.dz, reference.dtype, reference.device)
        return h, LAMBDA_0 * difference_power(n, reference.dtype, reference.device)

    def initial(self, b):
        """Eq. 16: x0 = (A^T A + lambda0 L^T L)^-1 A^T b."""
        n = b.shape[-1]
        h, smooth = self.spectra(n, b)
        return torch.fft.irfft2(torch.fft.rfft2(b) * h / (h * h + smooth), s=(n, n))

    def stage(self, y, b_hat, h, smooth, mu, theta):
        n = y.shape[-1]
        y_hat = torch.fft.rfft2(y)
        step = (h * h * y_hat - h * b_hat - smooth * y_hat) / (h * h + smooth)
        r = y - mu * torch.fft.irfft2(step, s=(n, n))
        return self.block(r, theta)

    def forward(self, b):
        """b is [B, N, N], zero-padded. Returns field, symmetry and sparsity losses."""
        h, smooth = self.spectra(b.shape[-1], b)
        b_hat = torch.fft.rfft2(b)
        old = y = self.initial(b)
        symmetry = sparsity = 0
        for k in range(1, self.layers + 1):
            mu, theta, rho = self.schedules(k)
            if self.use_checkpoint and self.training:
                x, sym, spa = checkpoint(self.stage, y, b_hat, h, smooth, mu, theta,
                                         use_reentrant=False)
            else:
                x, sym, spa = self.stage(y, b_hat, h, smooth, mu, theta)
            y = x + rho * (x - old)
            old = x
            symmetry, sparsity = symmetry + sym, sparsity + spa
        return x, symmetry, sparsity


def total_loss(output, target, symmetry, sparsity):
    """Eq. 14 with lambda1 = 0.01 and lambda2 = 0.001."""
    return ((output - target) ** 2).mean() + .01 * symmetry + .001 * sparsity
