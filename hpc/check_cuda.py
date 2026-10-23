"""Small GPU FFT/autograd check: Slurm allocation or explicit direct-server mode."""
import argparse
import os
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--allow-direct', action='store_true',
                    help='Run on a directly accessible GPU server without Slurm')
args = parser.parse_args()
if not os.environ.get('SLURM_JOB_ID') and not args.allow_direct:
    raise SystemExit('Run inside a Slurm GPU allocation, or use --allow-direct on a direct GPU server.')
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable to this Python environment.')
print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
x = torch.randn(1, 1, 64, 64, device='cuda', requires_grad=True)
y = torch.fft.ifft2(torch.fft.fft2(x)).real
torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)
y.square().mean().backward()
assert torch.isfinite(x.grad).all()
torch.cuda.synchronize()
print('CUDA FFT and gradients passed.')
