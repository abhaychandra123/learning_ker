"""Checks for the MAGIC-UNet port, its data and its tiled inference (needs torch)."""
from collections import Counter
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import torch  # noqa: E402

from magic_unet import (MAGICUNet, alpha, field_from_sheet, predict_tiled, solenoidal_sheet,  # noqa: E402
                        ssim, tile_owner, tile_starts)
from simulate_fields import fields_from_current  # noqa: E402

SPEC = json.loads((ROOT / 'tests' / 'data' / 'magic_unet_released_layers.json').read_text())


def released(cls):
    return [l for l in SPEC['layers'] if l['cls'] == cls]


class Architecture(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = MAGICUNet()

    def test_convolutions_match_released_model(self):
        ours = Counter((m.in_channels, m.out_channels, m.kernel_size, m.padding, m.bias is not None)
                       for m in self.model.modules() if isinstance(m, torch.nn.Conv2d))
        theirs = Counter((l['in_channels'], l['filters'], tuple(l['kernel']), (1, 1), l['use_bias'])
                         for l in released('Conv2D'))
        self.assertEqual(ours, theirs)
        self.assertTrue(all(l['padding'] == 'same' and l['activation'] == 'linear' and
                            l['strides'] == [1, 1] for l in released('Conv2D')))

    def test_norms_pools_upsamples_match(self):
        bns = [m for m in self.model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        self.assertEqual(Counter(m.num_features for m in bns),
                         Counter(l['channels'] for l in released('BatchNormalization')))
        for m, l in zip(bns, released('BatchNormalization')):
            self.assertAlmostEqual(m.eps, l['epsilon'])
            self.assertAlmostEqual(1 - m.momentum, l['momentum'])
        self.assertEqual(len(released('AveragePooling2D')), 3)
        self.assertTrue(all(l['strides'] == [2, 2] and l['pool_size'] == [2, 2]
                            for l in released('AveragePooling2D')))
        ups = [m for m in self.model.modules() if isinstance(m, torch.nn.Upsample)]
        self.assertEqual(len(ups), len(released('UpSampling2D')))
        self.assertTrue(all(l['interpolation'] == 'nearest' for l in released('UpSampling2D')))
        concats = released('Concatenate')
        self.assertEqual(len(concats), 4)       # two levels x two branches, no 64-level skip
        self.assertEqual(sorted(l['channels'] for l in concats), [256, 256, 512, 512])

    def test_parameter_count_and_checkpoint_size(self):
        trainable = sum(p.numel() for p in self.model.parameters())
        conv = sum(l['kernel'][0] * l['kernel'][1] * l['in_channels'] * l['filters'] + l['filters']
                   for l in released('Conv2D'))
        bn = sum(l['channels'] for l in released('BatchNormalization'))
        self.assertEqual(trainable, conv + 2 * bn)
        self.assertEqual(trainable, 15505194)
        stats = sum(b.numel() for n, b in self.model.named_buffers() if 'running' in n)
        bytes_all = 4 * (trainable + stats)
        self.assertLess(abs(bytes_all - SPEC['variables_data_bytes']) / SPEC['variables_data_bytes'], 0.002)

    def test_output_shapes(self):
        self.model.eval()
        for size in (64, 128):
            y = self.model(torch.randn(2, 3, size, size))
            self.assertEqual(tuple(y.shape), (2, 2, size, size))

    def test_branches_are_independent(self):
        y = self.model(torch.randn(2, 3, 64, 64))
        y[:, 0].sum().backward()
        self.assertTrue(all(p.grad is None or float(p.grad.abs().max()) == 0
                            for p in self.model.jy.parameters()))
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().max()) > 0
                            for p in self.model.jx.parameters()))

    def test_keras_he_normal_initialisation(self):
        conv = self.model.bottom[1][0]          # 512 -> 512, 3x3
        fan_in = 512 * 9
        w = conv.weight.detach()
        self.assertAlmostEqual(float(w.std()), np.sqrt(2 / fan_in), delta=0.02 * np.sqrt(2 / fan_in))
        self.assertLessEqual(float(w.abs().max()), 2 * np.sqrt(2 / fan_in) / 0.8796256610342398 + 1e-7)
        self.assertEqual(float(conv.bias.detach().abs().max()), 0.0)


class Inference(unittest.TestCase):
    def test_tile_ownership_covers_and_splits_overlaps(self):
        for size, tile, stride in ((1400, 128, 64), (1400, 128, 96), (300, 128, 32)):
            starts = tile_starts(size, tile, stride)
            own = tile_owner(size, tile, starts)
            self.assertEqual(starts[-1], size - tile)
            self.assertTrue(np.all((np.arange(size) >= own) & (np.arange(size) < own + tile)))

    def test_stitching_returns_every_pixel(self):
        class Copy(torch.nn.Module):
            def forward(self, x):
                return x[:, :2] * 1.0
        rng = np.random.default_rng(0)
        b = rng.standard_normal((3, 300, 260)).astype(np.float32)
        for stride in (32, 64, 128):
            j = predict_tiled(Copy(), b, beta=10.0, tile=128, stride=stride, batch=7)
            np.testing.assert_allclose(j, b[:2] / 10.0, rtol=1e-5, atol=1e-6)

    def test_equivalent_sheet_reproduces_the_offset_field(self):
        n, dx, d, h = 256, 0.1, 0.15, 0.5
        rng = np.random.default_rng(1)
        kx = np.zeros((n, n))
        ky = np.zeros((n, n))
        kx[100:110, 60:200] = 300.0
        ky[60:200, 150:160] = -300.0
        kx += rng.normal(0, 1, (n, n))
        (truth,) = fields_from_current(kx, ky, dx, [h + d])
        jx, jy = solenoidal_sheet(kx, ky, dx, depth=d)
        pred = field_from_sheet(np.stack([jx, jy]), h, dx, pad=0)
        np.testing.assert_allclose(pred, np.stack(truth), atol=1e-9 * np.abs(truth).max())

    def test_biot_savart_is_linear_in_cropped_current(self):
        n, dx = 200, 0.1
        rng = np.random.default_rng(3)
        j = rng.normal(0, 1, (2, n, n))
        inside = np.zeros_like(j)
        inside[:, 50:150, 50:150] = j[:, 50:150, 50:150]
        outside = j - inside
        total = field_from_sheet(j, 0.7, dx, pad=100)
        parts = field_from_sheet(inside, 0.7, dx, pad=100) + field_from_sheet(outside, 0.7, dx, pad=100)
        np.testing.assert_allclose(total, parts, atol=1e-10 * np.abs(total).max())

    def test_alpha_is_max_vector_norm(self):
        b = torch.zeros(1, 3, 4, 4)
        b[0, :, 1, 2] = torch.tensor([3.0, 4.0, 12.0])
        self.assertAlmostEqual(float(alpha(b)[0]), 13.0, places=5)

    def test_ssim_identity(self):
        x = np.random.default_rng(2).standard_normal((40, 40))
        self.assertAlmostEqual(ssim(x, x), 1.0, places=6)


class Data(unittest.TestCase):
    def test_one_layout(self):
        from magic_data import layout_crops
        from simulate_fields import DEFAULTS
        cfg = dict(crop=64, bin=2, standoffs=2, crops_per_standoff=3, z_mean=3.0, z_std=0.6,
                   z_trunc=3.0, empty_fraction=0.0)
        b, j, z, sigma, meta = layout_crops(('magic_test_000000', 0, dict(DEFAULTS), cfg))
        self.assertEqual(b.shape, (6, 3, 64, 64))
        self.assertEqual(j.shape, (6, 2, 64, 64))
        self.assertTrue(np.all(np.abs(z - 3.0) <= 1.8))
        self.assertTrue(np.all(sigma > 0))
        self.assertEqual(len(set(z.round(6))), 2)
        self.assertTrue(np.all(np.hypot(j[:, 0], j[:, 1]).max((1, 2)) > 0))
        # Noise is relative to each crop's own field (run 1's bug: up to 165 x max|B|).
        a = np.sqrt((b ** 2).sum(1)).max((1, 2))
        top = max(DEFAULTS['noise_percent']) / 100
        self.assertTrue(np.all(sigma / a <= 2 * top + 1e-9), sigma / a)

    def test_uniform_standoff(self):
        from magic_data import layout_crops
        from simulate_fields import DEFAULTS
        cfg = dict(crop=64, bin=2, standoffs=4, crops_per_standoff=1, z_mean=3.0, z_std=0.6,
                   z_trunc=3.0, empty_fraction=0.0, z_uniform=[1.0, 10.0])
        b, j, z, sigma, meta = layout_crops(('magic_test_000001', 0, dict(DEFAULTS), cfg))
        self.assertEqual(b.shape, (4, 3, 64, 64))
        self.assertTrue(np.all((z >= 1.0) & (z <= 10.0)), z)
        a = np.sqrt((b ** 2).sum(1)).max((1, 2))
        self.assertTrue(np.all(sigma / a <= 2 * max(DEFAULTS['noise_percent']) / 100 + 1e-9))


if __name__ == '__main__':
    unittest.main()
