"""CSC-LISTA: convolutional sparse coding LISTA for standoff deblurring.

This is Eq. 20 (convolutional LISTA) of the ALISTA paper: Liu, Chen, Wang and Yin,
"ALISTA: Analytic Weights Are As Good As Learned Weights in LISTA", ICLR 2019. It is
not the old in-house ConvLISTA in lista.py:

    x_m <- shrink(x_m - w_m' * (sum_m d_m * x_m - b), theta)

Each layer has its own learned filters w_m and threshold. Our addition, which the
paper does not cover: the paper's experiment is denoising, so here the dictionary
is d_m = h * g_m, with h the fixed exp(-kz) standoff kernel and g_m a convolutional
dictionary learned on sharp simulated fields. The output field is sum_m g_m * x_m.
The data term is measured inside the image window only.

With momentum=True each layer acts on an extrapolated point, as in FISTA:

    y = x_k + beta_k (x_k - x_(k-1))

beta_k is learned per layer and starts at FISTA's own sequence. Learned FISTA
networks of this kind appear in Moreau and Bruna, ICLR 2017, and Xiang et al.,
FISTA-Net. This convolutional deblurring form is ours.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def shrink(x, theta):
    return torch.sign(x) * torch.relu(torch.abs(x) - theta)


def centred_rfft(filters, n):
    """FFT of [..., D, D] real filters on an n-square canvas, centred at the origin."""
    d = filters.shape[-1]
    canvas = torch.zeros(*filters.shape[:-2], n, n, dtype=filters.dtype,
                         device=filters.device)
    canvas[..., :d, :d] = filters
    return torch.fft.rfft2(torch.roll(canvas, (-(d // 2), -(d // 2)), (-2, -1)))


def transfer(n, dx, dz, dtype=torch.float32, device='cpu'):
    """The exact upward-continuation filter exp(-2 pi |f| dz) in rfft layout."""
    fy = torch.fft.fftfreq(n, dx, dtype=dtype, device=device)[:, None]
    fx = torch.fft.rfftfreq(n, dx, dtype=dtype, device=device)[None, :]
    return torch.exp(-2 * torch.pi * dz * torch.hypot(fx, fy))


class Operator:
    """A convolutional dictionary, optionally followed by the standoff filter h.

    Spectra are built on demand and `chunk` filters at a time, so a full image
    never holds every filter spectrum in memory at once.
    """

    def __init__(self, filters, h=None, chunk=None):
        self.filters, self.h = filters, h
        self.chunk = chunk or filters.shape[0]

    def spectra(self, n):
        for first in range(0, self.filters.shape[0], self.chunk):
            part = centred_rfft(self.filters[first:first + self.chunk], n)
            yield first, part if self.h is None else part * self.h

    def forward(self, x):
        """sum_m filter_m * x_m for maps [B, M, N, N]; returns [B, N, N]."""
        n = x.shape[-1]
        total = 0
        for first, part in self.spectra(n):
            total = total + (torch.fft.rfft2(x[:, first:first + part.shape[0]]) * part).sum(1)
        return torch.fft.irfft2(total, s=(n, n))

    def adjoint(self, residual):
        """filter_m' * residual, the transpose of forward, for [B, N, N] input."""
        n = residual.shape[-1]
        spectrum = torch.fft.rfft2(residual)[:, None]
        return torch.cat([torch.fft.irfft2(spectrum * part.conj(), s=(n, n))
                          for _, part in self.spectra(n)], 1)

    def bound(self, n):
        """Largest eigenvalue of D D^T on an n-square canvas."""
        return float(sum((part.abs() ** 2).sum(0) for _, part in self.spectra(n)).max())


def unit_norm(g):
    return g / torch.linalg.vector_norm(g, dim=(-2, -1), keepdim=True).clamp_min(1e-12)


def window_mask(n, pad, device='cpu'):
    mask = torch.zeros(n, n, device=device)
    mask[pad:n - pad, pad:n - pad] = 1
    return mask


def code_ista(b, operator, lam, steps, pad=0, fast=True, bound=None):
    """FISTA or ISTA for 0.5 ||mask(Dx - b)||^2 + lam ||x||_1."""
    n = b.shape[-1]
    mask = window_mask(n, pad, b.device) if pad else 1.0
    step = 1 / (bound or operator.bound(n))
    x = torch.zeros(b.shape[0], operator.filters.shape[0], n, n, dtype=b.dtype,
                    device=b.device)
    y, t = x, 1.0
    for _ in range(steps):
        residual = mask * (operator.forward(y) - b)
        update = shrink(y - step * operator.adjoint(residual), lam * step)
        if fast:
            t_next = (1 + (1 + 4 * t * t) ** .5) / 2
            y = update + (t - 1) / t_next * (update - x)
            t = t_next
        else:
            y = update
        x = update
    return x


def random_crops(fields, count, size, generator, minimum=0.0):
    """Crops of [count, H, W] fields; retries pieces with almost no signal."""
    height = fields.shape[-1]
    crops = []
    for _ in range(count):
        for _ in range(30):
            index = int(torch.randint(0, len(fields), (1,), generator=generator))
            row = int(torch.randint(0, height - size + 1, (1,), generator=generator))
            col = int(torch.randint(0, height - size + 1, (1,), generator=generator))
            crop = fields[index, row:row + size, col:col + size]
            if float(crop.abs().max()) >= minimum:
                break
        crops.append(crop)
    return torch.stack(crops)


def learn_dictionary(fields, atoms, size, lam, steps, batch, crop, sparse_steps=30,
                     lr=1e-3, seed=0, device='cpu', log=print):
    """Alternate sparse coding and Adam on g, with unit-norm filters, on crops."""
    generator = torch.Generator().manual_seed(seed)
    pad = size // 2 + 1
    n = crop + 2 * pad
    start = random_crops(fields, atoms, size, generator, .02).to(device)
    noise = torch.randn(start.shape, generator=generator).to(device, start.dtype)
    g = nn.Parameter(unit_norm(start - start.mean((1, 2), keepdim=True) + 1e-3 * noise))
    optimiser = torch.optim.Adam([g], lr=lr)
    mask = window_mask(n, pad, device)
    history = []
    for step in range(steps):
        pieces = random_crops(fields, batch, crop, generator, .02).to(device)
        target = torch.nn.functional.pad(pieces, (pad, pad, pad, pad))
        with torch.no_grad():
            x = code_ista(target, Operator(g.data), lam, sparse_steps, pad)
        error = mask * (Operator(g).forward(x) - target)
        loss = .5 * (error ** 2).sum() / batch
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            g.copy_(unit_norm(g))
        if step % 100 == 0 or step == steps - 1:
            relative = float(error.norm() / (mask * target).norm())
            density = float((x != 0).float().mean())
            history.append(dict(step=step, relative_l2=relative, density=density))
            log(f'dictionary step {step}: relative L2 {relative:.4f}, density {density:.4f}')
    return g.detach(), history


class CSCLISTA(nn.Module):
    def __init__(self, atoms, dx, dz, canvas, pad, lam, layers=16, width=155,
                 use_checkpoint=True, chunk=None, momentum=False):
        """atoms is [M, D, D]; canvas is the padded training size in pixels."""
        super().__init__()
        self.dx, self.dz, self.pad, self.layers = dx, dz, pad, layers
        self.use_checkpoint, self.chunk = use_checkpoint, chunk
        self.register_buffer('atoms', atoms)
        h = transfer(canvas, dx, dz, atoms.dtype, atoms.device)
        operator = Operator(atoms, h, chunk)
        step = 1 / operator.bound(canvas)
        self.bound_value = 1 / step
        # ISTA start: w = d / L, truncated to the filter support.
        rows = []
        for first, part in operator.spectra(canvas):
            full = torch.roll(torch.fft.irfft2(part, s=(canvas, canvas)),
                              (width // 2, width // 2), (-2, -1))
            rows.append(full[..., :width, :width] * step)
        self.w = nn.Parameter(torch.cat(rows)[None].repeat(layers, 1, 1, 1))
        self.theta = nn.Parameter(torch.full((layers,), lam * step, dtype=atoms.dtype,
                                             device=atoms.device))
        self.beta = None
        if momentum:
            t, start_beta = 1.0, []
            for _ in range(layers):
                t_next = (1 + (1 + 4 * t * t) ** .5) / 2
                start_beta.append((t - 1) / t_next)
                t = t_next
            self.beta = nn.Parameter(torch.tensor(start_beta, dtype=atoms.dtype,
                                                  device=atoms.device))

    def physics(self, n):
        h = transfer(n, self.dx, self.dz, self.atoms.dtype, self.atoms.device)
        return Operator(self.atoms, h, self.chunk), window_mask(n, self.pad, h.device)

    def layer(self, x, b, k, operator, mask):
        residual = mask * (operator.forward(x) - b)
        update = x - Operator(self.w[k], chunk=self.chunk).adjoint(residual)
        return shrink(update, self.theta[k].clamp_min(0))

    def codes(self, b):
        n = b.shape[-1]
        operator, mask = self.physics(n)
        x = torch.zeros(b.shape[0], self.w.shape[1], n, n, dtype=b.dtype, device=b.device)
        y = x
        for k in range(self.layers):
            if self.use_checkpoint and self.training:
                update = checkpoint(self.layer, y, b, k, operator, mask, use_reentrant=False)
            else:
                update = self.layer(y, b, k, operator, mask)
            y = update if self.beta is None else update + self.beta[k] * (update - x)
            x = update
        return x

    def lasso_objective(self, b, lam):
        """Mean of 0.5 ||mask(Dz - b)||^2 + lam ||z||_1 per pixel, as in LFISTA training."""
        z = self.codes(b)
        operator, mask = self.physics(b.shape[-1])
        residual = mask * (operator.forward(z) - b)
        value = .5 * (residual ** 2).sum((1, 2)) + lam * z.abs().sum((1, 2, 3))
        return value.mean() / b.shape[-1] ** 2

    def forward(self, b):
        """b is [B, N, N], zero-padded; returns the padded output field."""
        return Operator(self.atoms, chunk=self.chunk).forward(self.codes(b))


class Dataless(nn.Module):
    """Tikhonov/Wiener baseline with the same exp(-kz) operator."""

    def __init__(self, dx, dz, mu):
        super().__init__()
        self.dx, self.dz, self.mu = dx, dz, mu

    def forward(self, b):
        n = b.shape[-1]
        h = transfer(n, self.dx, self.dz, b.dtype, b.device)
        return torch.fft.irfft2(torch.fft.rfft2(b) * h / (h * h + self.mu), s=(n, n))
