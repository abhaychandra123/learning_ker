"""Build the portable Kaggle notebook and minimal z=3 data/code bundle."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def notebook():
    cells = []

    def md(text):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(True)})

    def code(text):
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                      'outputs': [], 'source': text.splitlines(True)})

    md('''# ISTA, supervised convolutional LISTA, and FISTA

**Experiment:** same coil, z=3 → 0.5 µm. All three full component images are used to fit the physical kernel and train LISTA. Results are **training-sample reconstruction**, not an independent test.

1. Upload this `.ipynb` to Kaggle.
2. Add `kaggle_lista_bundle.zip` as a dataset/input. Kaggle may unpack it automatically; both forms are supported below.
3. Select the two T4 GPUs. Run the setup/tests, inspect the configuration, then run training.
4. Download the final results ZIP from the last cell.

No internet download of raw data or external model weights is required. The bundle contains only the six needed cached arrays, scripts and tests. It retains the original metadata, which lists additional heights that are not included in the bundle.

The original `deblur.py` is unchanged. The separate scripts can also run outside this notebook. Full equations and limitations are in `LISTA_FISTA.md` inside the bundle.''')

    md('''## Mathematical contract

All methods use the same padded operator `A` and the objective `J(x)=0.5*||Ax-b||² + λ||x||₁` for optimization diagnostics. ISTA and FISTA solve this objective; LISTA is trained to minimise sharp-field MSE instead.

- ISTA: `x_next=soft(x-α A*(Ax-b), αλ)`.
- FISTA: take that step at the extrapolated point, then use Beck–Teboulle momentum. Return the proximal iterate, not the extrapolated point.
- Structured LISTA: `x_next=soft(x-α A*(Ax-b)+C_b(b)+C_x(x), θ)`.

Here `α=0.99/max|H|²`. `C_b`, `C_x` are learned zero-padded, bias-free convolutional corrections, tied across layers; θ is a learned nonnegative scalar. Initially corrections are zero and θ=αλ, giving exactly ISTA. The full physical operator and intermediate crop remain present.

This is the agreed physics-preserving, supervised variant of [LISTA](https://icml.cc/Conferences/2010/papers/449.pdf); it differs from the original paper's sparse-solver targets. [FISTA source, §4](https://www.ceremade.dauphine.fr/~carlier/FISTA). Learned LISTA is not guaranteed to minimise J or converge if repeatedly extended.''')

    code('''from pathlib import Path
import os, json, hashlib, shutil, zipfile, sys, subprocess
from datetime import datetime, timezone

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLCONFIGDIR'] = '/kaggle/working/matplotlib_config'

# Leave None for automatic discovery. If several bundles are attached, set
# this to the ZIP file OR to the extracted folder containing BUNDLE_MANIFEST.json.
INPUT_BUNDLE = None

if INPUT_BUNDLE is None:
    candidates = list(Path('/kaggle/input').rglob('kaggle_lista_bundle.zip'))
    candidates += [p.parent for p in Path('/kaggle/input').rglob('BUNDLE_MANIFEST.json')]
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one bundle; found {candidates}. Set INPUT_BUNDLE explicitly.')
    INPUT_BUNDLE = candidates[0]
INPUT_BUNDLE = Path(INPUT_BUNDLE)
WORK = Path('/kaggle/working') / ('lista_project_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
if INPUT_BUNDLE.is_file():
    WORK.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(INPUT_BUNDLE) as archive:
        for name in archive.namelist():
            if not (WORK / name).resolve().is_relative_to(WORK.resolve()):
                raise ValueError(f'Invalid archive member: {name}')
        archive.extractall(WORK)
else:
    shutil.copytree(INPUT_BUNDLE, WORK)
manifest = json.loads((WORK / 'BUNDLE_MANIFEST.json').read_text())
for relative, expected in manifest['sha256'].items():
    actual = hashlib.sha256((WORK / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f'Bundle hash mismatch: {relative}')
print('Verified bundle:', WORK)
sys.path.insert(0, str(WORK / 'scripts'))
''')

    code('''import numpy as np
import pandas as pd
import matplotlib
import torch

print('Python:', sys.version)
print('NumPy:', np.__version__, 'pandas:', pd.__version__, 'Matplotlib:', matplotlib.__version__)
print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('Available GPUs:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory / 2**30, 'GiB')
if not torch.cuda.is_available():
    raise RuntimeError('Enable GPU acceleration for the full-image training run.')
# Keep Kaggle's installed CUDA-compatible torch; do not replace it with a CPU wheel.
''')

    md('''## Run mathematical checks first

Tests check adjoints, dense reference updates, a known L1 solution, ISTA equivalence, LISTA gradients, normalization, training updates, checkpoint reloads, and the uneven 2+1 two-GPU batch gradients. The CPU-only local development run skipped the last check; this notebook runs it when both GPUs are available.''')
    code("""subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
               cwd=WORK, check=True)
""")

    md('''## Explicit experiment configuration

These are editable starting settings, not claimed optimal parameters. Train whole images to preserve the exact forward operator. Shared 9×9 corrections and one shared threshold give 163 learned parameters. Gradient checkpointing reduces activation memory by recomputing layers during backpropagation.

Training uses two GPUs, while **all inference timing uses GPU 0** so hardware is identical across methods. Each latency is a warmed, synchronized median for the batch of three images. Separate training time is also recorded. No half-precision FFT or TF32 is used.

The global field scale is recorded, and the L1 parameter is divided by that scale internally. This preserves the physical-unit ISTA/FISTA objective. `INITIAL='blurred'` matches the existing repository; use `'zero'` to change all methods together.''')

    code('''Z_TARGET = 3.0
PAD = 32
LAMBDA_UT = 1e-4
LAYERS = 8
CORRECTION_KERNEL_SIZE = 9
EPOCHS = 200
LEARNING_RATE = 1e-4
BASELINE_ITERATIONS = 200
INITIAL = 'blurred'
NUM_GPUS = 2
SEED = 0
RECORD_EVERY = 10
BENCHMARK_REPEATS = 3

if torch.cuda.device_count() < NUM_GPUS:
    raise RuntimeError(f'Requested {NUM_GPUS} GPUs; enable both or explicitly change NUM_GPUS.')
RUN_DIR = WORK / 'runs' / ('z3_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
print('Output:', RUN_DIR)
''')

    md('''## Train and compare

This is the computationally intensive cell. It fits H once, checks untrained LISTA against ISTA, evaluates baselines, trains LISTA against all three sharp images, restores the best **training-loss** weights, and saves the comparisons. It refuses to overwrite an existing run directory. Increase epochs/layers only as an explicit new experiment.''')
    code('''command = [sys.executable, '-u', 'scripts/lista.py',
    '--data-root', str(WORK), '--output', str(RUN_DIR),
    '--z', str(Z_TARGET), '--pad', str(PAD), '--lambda-ista', str(LAMBDA_UT),
    '--layers', str(LAYERS), '--kernel-size', str(CORRECTION_KERNEL_SIZE),
    '--epochs', str(EPOCHS), '--lr', str(LEARNING_RATE),
    '--baseline-iterations', str(BASELINE_ITERATIONS), '--initial', INITIAL,
    '--device', 'cuda', '--gpus', str(NUM_GPUS), '--seed', str(SEED),
    '--record-every', str(RECORD_EVERY), '--benchmark-repeats', str(BENCHMARK_REPEATS),
    '--checkpoint-layers']
print(' '.join(command))
subprocess.run(command, cwd=WORK, check=True)
''')

    code('''results = json.loads((RUN_DIR / 'metrics.json').read_text())
print(results['evaluation'])
print('Best training epoch:', results['best_training_epoch'])
print('Training seconds:', results['training_seconds_including_post_update_evaluation'])
print('Parameter changes:', results['learned_parameter_change_l2'])
rows = []
for method, entry in results['comparisons'].items():
    for component, score in zip(results['components'], entry['metrics']):
        rows.append({'method': method, 'component': component,
            'range NRMSE (%)': None if score['nrmse_range'] is None else 100*score['nrmse_range'],
            'relative L2 (%)': None if score['relative_l2'] is None else 100*score['relative_l2'],
            'RMSE (uT)': score['rmse_uT'], 'L1 objective': score['objective'],
            'stationarity RMS (uT)': score['pg_rms_uT'],
            'median batch inference (s)': entry['inference']['median_seconds']})
table = pd.DataFrame(rows)
display(table)
table.to_csv(RUN_DIR / 'comparison_table.csv', index=False)
''')

    md('''## Inspect results before interpreting speed

Compare equal-depth entries (`ista_8`, `fista_8`, `lista` with defaults) and the longer runs separately. A learned layer has more operations than an ISTA step. Low supervised reconstruction error does not imply low L1 objective, and low objective does not necessarily imply better agreement with the noisy reference. The plotted intermediate LISTA layers were not individually supervised. All scores here are in-sample.''')
    code('''from IPython.display import Image, display
for filename in ('reconstructions.png', 'convergence.png', 'training_and_latency.png'):
    display(Image(filename=str(RUN_DIR / filename)))
''')

    md('''## Verify saved-model inference

This reloads the selected checkpoint without retraining or fitting H. It checks the restored output against the saved physical-unit reconstruction.''')
    code('''from lista import load_checkpoint
model, saved = load_checkpoint(RUN_DIR / 'lista_best_training.pt', device='cuda:0')
with np.load(RUN_DIR / 'reconstructions.npz', allow_pickle=False) as arrays:
    # Rebuild normalization in float64 before converting to float32, matching training.
    observation = torch.as_tensor(arrays['blurred'].astype(np.float64) / saved['scale_uT'],
                                  dtype=torch.float32, device='cuda:0').unsqueeze(1)
    expected = arrays['lista'].copy()
with torch.no_grad():
    restored = model(observation).cpu().numpy()[:, 0] * saved['scale_uT']
np.testing.assert_allclose(restored, expected, rtol=3e-5, atol=3e-4)
print('Checkpoint round-trip verified; reconstruction units are microtesla.')
''')

    code('''from IPython.display import FileLink
archive_path = shutil.make_archive(str(RUN_DIR), 'zip', root_dir=RUN_DIR)
print('Download the ZIP below. It includes numerical arrays, weights, metrics and plots.')
display(FileLink(archive_path))
''')
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
            'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.11'}},
            'nbformat': 4, 'nbformat_minor': 4}


def main():
    output = ROOT/'notebooks'/'ISTA_LISTA_FISTA_Kaggle.ipynb'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(notebook(), indent=1, ensure_ascii=False)+'\n')
    names = ['deblur.py', 'fields.py', 'propagator.py', 'iterative.py', 'fista.py',
             'torch_inverse.py', 'lista.py', 'compare_lista.py', 'plot_comparison.py']
    paths = [ROOT/'scripts'/name for name in names]
    paths += sorted((ROOT/'tests').glob('test_*.py'))
    paths += [ROOT/'cache'/'grid.json', ROOT/'LISTA_FISTA.md', ROOT/'requirements-lista.txt', output]
    paths += [ROOT/'cache'/f'z{z:04.1f}_{c}.npy' for z in (.5, 3.) for c in ('Bx','By','Bz')]
    manifest = {'experiment': 'same coil z=3 to 0.5, supervised physics-preserving ConvLISTA',
                'packaged_heights': [.5, 3.], 'packaged_components': ['Bx','By','Bz'],
                'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    bundle = ROOT/'kaggle_lista_bundle.zip'
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in paths:
            z.write(p, str(p.relative_to(ROOT)))
        z.writestr('BUNDLE_MANIFEST.json', json.dumps(manifest, indent=2))
    print(f'Wrote {output}')
    print(f'Wrote {bundle} ({bundle.stat().st_size/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
