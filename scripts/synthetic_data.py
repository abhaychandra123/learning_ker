"""Portable, reproducible recipes for synthetic magnetic-field reconstruction.

NumPy only. Physical arrays are in microtesla, coordinates in micrometres.
The existing COMSOL caches and comparison runners are never modified.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import re

import numpy as np

from propagator import analytic_transfer, k_grid

COMPONENTS = ('Bx', 'By', 'Bz')
KIND = 'synthetic_upward_continuation'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def inside(root, relative):
    root = Path(root).resolve()
    p = (root / relative).resolve()
    if Path(relative).is_absolute() or not p.is_relative_to(root):
        raise ValueError('Dataset path escapes its root')
    return p


def valid_grid(grid):
    for k in ('nx', 'ny'):
        if isinstance(grid[k], bool) or int(grid[k]) != grid[k] or grid[k] < 2:
            raise ValueError('Invalid grid shape')
    if not all(math.isfinite(grid[k]) for k in ('x0', 'y0', 'dx', 'dy')):
        raise ValueError('Nonfinite grid')
    if min(grid['dx'], grid['dy']) <= 0:
        raise ValueError('Pixel spacing must be positive')


def validate_fields(fields, grid):
    valid_grid(grid)
    if fields.shape != (3, grid['ny'], grid['nx']) or not np.isfinite(fields).all():
        raise ValueError('Expected finite [Bx, By, Bz] fields matching the grid')


def transfer(shape, dx, dy, dz, pad):
    if (not math.isfinite(dz) or dz < 0 or min(dx, dy) <= 0
            or not np.isfinite([dx, dy]).all() or int(pad) != pad or pad < 0):
        raise ValueError('Invalid propagation distance, spacing or padding')
    return analytic_transfer(k_grid(tuple(n + 2*pad for n in shape), dx, dy), dz)


def propagate(field, dz, dx, dy, pad=256):
    """Crop * inverse FFT * exp(-2 pi |f| dz) * FFT * zero-pad.

    pad=0 is explicitly periodic. Positive padding assumes zero exterior field;
    it reduces periodic-image interaction but cannot recover an unknown exterior.
    """
    if field.ndim != 2 or not np.isfinite(field).all():
        raise ValueError('Expected a finite two-dimensional component')
    h = transfer(field.shape, dx, dy, dz, pad)
    padded = np.pad(np.asarray(field, dtype=np.float64), pad) if pad else field
    result = np.fft.ifft2(np.fft.fft2(padded) * h).real
    return result[pad:pad+field.shape[0], pad:pad+field.shape[1]].copy()


def rotation_component(component, degrees):
    """Active CCW rotation about +z, with array row direction = physical +y."""
    if degrees not in (0, 90, 180, 270) or component not in (0, 1, 2):
        raise ValueError('Expected a signed component and a quarter-turn rotation')
    mappings = (((0, 1), (1, 1), (2, 1)),
                ((1, -1), (0, 1), (2, 1)),
                ((0, -1), (1, -1), (2, 1)),
                ((1, 1), (0, -1), (2, 1)))
    return mappings[degrees//90][component]


def rotate_component(field, degrees, sign=1):
    # np.rot90(k=+1) is clockwise in physical coordinates when origin='lower'.
    return (sign * np.rot90(field, k=-degrees//90)).copy()


def rotated_grid(grid, degrees):
    rotation_component(0, degrees)
    out = dict(grid)
    if degrees in (90, 270):
        out.update(nx=grid['ny'], ny=grid['nx'], dx=grid['dy'], dy=grid['dx'])
    cx = grid['x0'] + (grid['nx']-1)*grid['dx']/2
    cy = grid['y0'] + (grid['ny']-1)*grid['dy']/2
    out['x0'] = cx - (out['nx']-1)*out['dx']/2
    out['y0'] = cy - (out['ny']-1)*out['dy']/2
    return out


def stable_seed(*parts):
    encoded = json.dumps(parts, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:16], 'little')


def add_noise(clean, percent, seed):
    if not math.isfinite(percent) or percent < 0:
        raise ValueError('Noise percentage must be finite and nonnegative')
    clean = np.asarray(clean, dtype=np.float32)
    sigma = percent/100 * float(np.ptp(clean.astype(np.float64)))
    rng = np.random.Generator(np.random.PCG64(seed))
    noisy = (clean.astype(np.float64) + rng.normal(0, sigma, clean.shape)).astype(np.float32)
    return noisy, sigma


def error_metrics(pred, truth, margin=0):
    if margin:
        pred, truth = pred[margin:-margin, margin:-margin], truth[margin:-margin, margin:-margin]
    residual = np.asarray(pred, dtype=np.float64) - truth
    rmse = float(np.sqrt(np.mean(residual**2)))
    span = float(np.ptp(truth))
    return {'rmse_uT': rmse, 'range_nrmse_percent': 100*rmse/span if span else None}


def source_inventory(source_manifest=None, source_run=None):
    """Read metadata only; each geometry is loaded separately during preparation."""
    if bool(source_manifest) == bool(source_run):
        raise ValueError('Choose exactly one source manifest or completed comparison run')
    source = Path(source_manifest or Path(source_run)/'setup.json').resolve()
    data = json.loads(source.read_text())
    manifest = data['manifest'] if source_run else data
    if (manifest.get('version') != 1 or manifest.get('components') != list(COMPONENTS)
            or manifest.get('field_unit') != 'uT' or manifest.get('length_unit') != 'um'):
        raise ValueError('Unsupported source schema, components or units')
    records = {}
    for geometry, entry in sorted(manifest['geometries'].items()):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', geometry):
            raise ValueError('Invalid geometry identifier')
        valid_grid(entry['grid'])
        planes = {float(z): v for z, v in entry['planes'].items()}
        if len(planes) < 2 or not all(math.isfinite(z) and z >= 0 for z in planes):
            raise ValueError('Need at least two finite nonnegative source heights')
        z0, zmax = min(planes), max(planes)
        record = dict(geometry=geometry, grid=entry['grid'], reference_z_um=z0,
                      max_existing_z_um=zmax)
        if source_run:
            candidates = [p for p in manifest['pairs'] if p['geometry'] == geometry
                          and p['sharp_z_um'] == z0 and p['blur_z_um'] == zmax]
            if len(candidates) != 1:
                raise ValueError('Comparison source must contain the lowest-to-highest pair')
            pair = candidates[0]
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', pair['id']):
                raise ValueError('Unsafe source pair identifier')
            record['archive'] = str(inside(source.parent, f"comparison/{pair['id']}.npz"))
        else:
            for role, z in [('reference', z0), ('actual', zmax)]:
                record[role] = str(inside(source.parent, planes[z]['path']))
                record[role+'_sha256'] = planes[z]['sha256']
        records[geometry] = record
    return records, {'metadata_file': str(source), 'metadata_sha256': digest(source),
                     'kind': 'comparison_run' if source_run else 'prepared_comsol'}


def load_source(record):
    if 'archive' in record:
        with np.load(record['archive'], allow_pickle=False) as data:
            sharp, actual = data['sharp'].copy(), data['blurred'].copy()
        hashes = {'archive_sha256': digest(record['archive'])}
    else:
        hashes = {}
        for role in ('reference', 'actual'):
            hashes[role+'_sha256'] = digest(record[role])
            if hashes[role+'_sha256'] != record[role+'_sha256']:
                raise ValueError('Source checksum mismatch: ' + record[role])
        sharp = np.load(record['reference'], allow_pickle=False)
        actual = np.load(record['actual'], allow_pickle=False)
    validate_fields(sharp, record['grid'])
    validate_fields(actual, record['grid'])
    return sharp.astype(np.float32), actual.astype(np.float32), hashes


def prepare(output, *, source_manifest=None, source_run=None, height_count=10,
            min_dz_um=.5, noise_percent=(0., 1., 1.5), rotations=(0,),
            realizations=1, pad=256, interior_margin=128, seed=0,
            validation_geometry=(), test_geometry=(), geometries=None):
    records, source = source_inventory(source_manifest, source_run)
    if geometries is not None:
        if not set(geometries) <= records.keys() or not geometries:
            raise ValueError('Unknown or empty geometry selection')
        records = {g: records[g] for g in sorted(set(geometries))}
    numbers = (height_count, realizations, pad, interior_margin, seed)
    if (any(isinstance(v, bool) or int(v) != v for v in numbers)
            or min(height_count, realizations) < 1 or min(pad, interior_margin, seed) < 0
            or not math.isfinite(min_dz_um) or min_dz_um <= 0):
        raise ValueError('Invalid generation settings')
    noise_percent, rotations = sorted(set(noise_percent)), sorted(set(rotations))
    if (not noise_percent or any(not math.isfinite(x) or x < 0 for x in noise_percent)
            or not rotations or any(x not in (0, 90, 180, 270) for x in rotations)):
        raise ValueError('Invalid noise levels or rotations')
    validation, test = set(validation_geometry), set(test_geometry)
    if validation & test or not (validation | test) <= records.keys():
        raise ValueError('Unknown or overlapping geometry splits')
    if not records.keys() - (validation | test):
        raise ValueError('At least one training geometry is required')
    for record in records.values():
        if min_dz_um > record['max_existing_z_um']-record['reference_z_um']:
            raise ValueError('Minimum dz exceeds the available height span')
        if interior_margin*2 >= min(record['grid']['nx'], record['grid']['ny']):
            raise ValueError('Interior margin leaves no pixels')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'references').mkdir()
    write_json(output/'status.json', {'status': 'preparing'})
    config = dict(height_count=height_count, min_dz_um=min_dz_um,
                  noise_percent=noise_percent, rotations_degrees=rotations,
                  realizations=realizations, pad_pixels=pad,
                  interior_margin_pixels=interior_margin, seed=seed)
    manifest = dict(version=1, dataset_kind=KIND, components=list(COMPONENTS),
                    field_unit='uT', length_unit='um', config=config, source=source,
                    geometries={}, samples=[], environment={'python': platform.python_version(),
                    'numpy': np.__version__, 'rng': 'numpy.random.PCG64'},
                    code_sha256={name: digest(Path(__file__).with_name(name))
                                 for name in ('synthetic_data.py', 'propagator.py')},
                    noise_definition='sigma_uT = noise_percent / 100 * ptp(clean propagated component)',
                    reference_definition='Lowest available COMSOL plane; no added noise, not a mesh-free field',
                    boundary={'mode': 'zero_extension' if pad else 'periodic', 'pad_pixels': pad},
                    evaluation='geometry-label holdout' if validation or test else 'same-geometry synthetic training only')
    audit = []
    scale = 0.
    try:
        for geometry, record in records.items():
            print(f'Preparing {geometry}: reference {record["reference_z_um"]:g} µm', flush=True)
            sharp, actual, hashes = load_source(record)
            ref_path = output/'references'/f'{geometry}.npy'
            np.save(ref_path, sharp, allow_pickle=False)
            split = 'test' if geometry in test else 'validation' if geometry in validation else 'train'
            if split == 'train':
                scale = max(scale, float(np.abs(sharp).max()))
            manifest['geometries'][geometry] = dict(
                grid=record['grid'], reference_z_um=record['reference_z_um'],
                max_existing_z_um=record['max_existing_z_um'], split=split,
                reference_path=str(ref_path.relative_to(output)), reference_sha256=digest(ref_path),
                source={**record, **hashes})
            max_dz = record['max_existing_z_um']-record['reference_z_um']
            grid = record['grid']
            # A bounded audit at the most distant available COMSOL plane. These
            # are propagation discrepancies, not estimates of Gaussian noise.
            for c, component in enumerate(COMPONENTS):
                baseline = propagate(sharp[c], max_dz, grid['dx'], grid['dy'], pad)
                for check_pad in sorted(set((0, pad, max(1, 2*pad)))):
                    prediction = baseline if check_pad == pad else propagate(
                        sharp[c], max_dz, grid['dx'], grid['dy'], check_pad)
                    row = dict(geometry=geometry, split=split, component=component,
                               reference_z_um=record['reference_z_um'], target_z_um=record['max_existing_z_um'],
                               pad_pixels=check_pad, interior_margin_pixels=interior_margin,
                               full=error_metrics(prediction, actual[c]),
                               interior=error_metrics(prediction, actual[c], interior_margin),
                               difference_from_generation_padding=error_metrics(prediction, baseline))
                    audit.append(row)
            for dz in sorted(set(float(v) for v in np.linspace(min_dz_um, max_dz, height_count))):
                for rotation in rotations:
                    for percent in noise_percent:
                        for repeat in range(realizations if percent else 1):
                            recipe = dict(geometry=geometry, split=split,
                                          reference_z_um=record['reference_z_um'], dz_um=dz,
                                          target_z_um=record['reference_z_um']+dz, rotation_degrees=rotation,
                                          noise_percent=percent, realization=repeat)
                            token = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:20]
                            recipe['id'] = geometry+'_'+token
                            recipe['noise_seed'] = stable_seed(seed, recipe['id'])
                            manifest['samples'].append(recipe)
            print(f'  audit complete; {sum(s["geometry"] == geometry for s in manifest["samples"])} vector recipes', flush=True)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('Training references have zero or invalid normalization scale')
        manifest['normalization'] = dict(scale_uT=scale, definition='max absolute training reference value before augmentation',
                                         applied_to_saved_references=False, clip=False)
        counts = Counter(s['split'] for s in manifest['samples'])
        raw_bytes = sum(3*3*4*records[s['geometry']]['grid']['nx']*records[s['geometry']]['grid']['ny']
                        for s in manifest['samples'])
        manifest['inventory'] = dict(source_geometries=len(records), vector_pairs=len(manifest['samples']),
                                     component_pairs=3*len(manifest['samples']), vector_pairs_by_split=dict(counts),
                                     materialized_three_array_bytes_uncompressed=raw_bytes,
                                     independent_source_note='Variants share reference geometries and are not independent source configurations')
        write_json(output/'boundary_audit.json', {'interpretation': 'Forward discrepancies at each maximum existing height. No mesh-noise or Gaussian-noise attribution. Holdout rows are evaluation-only; do not tune on test rows.', 'rows': audit})
        write_json(output/'manifest.json', manifest)
        with (output/'inventory.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(manifest['samples'][0]))
            writer.writeheader(); writer.writerows(manifest['samples'])
        write_json(output/'status.json', {'status': 'complete', **manifest['inventory']})
        print(json.dumps(manifest['inventory'], indent=2), flush=True)
        return manifest
    except Exception as error:
        write_json(output/'status.json', {'status': 'failed', 'error': str(error)})
        raise


class SyntheticDataset:
    """One-component-at-a-time loader with deterministic, optionally fresh noise.

    Returned fields are float32; forward FFTs use float64. No fitted kernels.
    Loading needs only this dataset directory, not its original COMSOL source.
    """
    def __init__(self, root, verify=True):
        self.root = Path(root).resolve()
        self.manifest = m = json.loads((self.root/'manifest.json').read_text())
        if (m.get('version') != 1 or m.get('dataset_kind') != KIND
                or m.get('components') != list(COMPONENTS) or m.get('field_unit') != 'uT'
                or m.get('length_unit') != 'um'):
            raise ValueError('Unsupported synthetic dataset')
        if json.loads((self.root/'status.json').read_text())['status'] != 'complete':
            raise ValueError('Dataset preparation is incomplete')
        self.scale = m['normalization']['scale_uT']
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('Invalid normalization scale')
        self.references = {}
        for geometry, entry in m['geometries'].items():
            path = inside(self.root, entry['reference_path'])
            if verify and digest(path) != entry['reference_sha256']:
                raise ValueError('Reference checksum mismatch')
            a = np.load(path, mmap_mode='r', allow_pickle=False)
            validate_fields(a, entry['grid'])
            if entry['split'] not in ('train', 'validation', 'test'):
                raise ValueError('Invalid geometry split')
            self.references[geometry] = a
        self.samples = m['samples']
        seen = set()
        for s in self.samples:
            g = m['geometries'][s['geometry']]
            if s['id'] in seen or not re.fullmatch(r'[A-Za-z0-9_-]+', s['id']):
                raise ValueError('Invalid/duplicate sample ID')
            seen.add(s['id'])
            if s['split'] != g['split']:
                raise ValueError('A geometry cannot cross dataset splits')
            if (not math.isfinite(s['dz_um']) or s['dz_um'] <= 0
                    or s['reference_z_um'] != g['reference_z_um']
                    or not math.isclose(s['target_z_um'], s['reference_z_um']+s['dz_um'])
                    or s['target_z_um'] > g['max_existing_z_um']+1e-9
                    or not math.isfinite(s['noise_percent']) or s['noise_percent'] < 0):
                raise ValueError('Invalid sample height or noise setting')
            rotation_component(0, s['rotation_degrees'])
        if not self.samples:
            raise ValueError('Empty synthetic dataset')
        self._clean_cache = None

    def component(self, sample_index, component, *, epoch=0, normalized=False):
        if isinstance(component, str):
            component = COMPONENTS.index(component)
        sample = self.samples[sample_index]
        if int(epoch) != epoch or epoch < 0 or (epoch and sample['split'] != 'train'):
            raise ValueError('Only training noise may change with epoch')
        geometry = sample['geometry']; entry = self.manifest['geometries'][geometry]
        grid = entry['grid']; rotation = sample['rotation_degrees']
        source_component, sign = rotation_component(component, rotation)
        source = self.references[geometry][source_component]
        key = (geometry, source_component, sample['dz_um'])
        if self._clean_cache is None or self._clean_cache[0] != key:
            clean = propagate(source, sample['dz_um'], grid['dx'], grid['dy'],
                              self.manifest['boundary']['pad_pixels']).astype(np.float32)
            self._clean_cache = (key, clean)
        clean = rotate_component(self._clean_cache[1], rotation, sign)
        sharp = rotate_component(source, rotation, sign)
        noise_seed = stable_seed(sample['noise_seed'], component, epoch)
        blurred, sigma = add_noise(clean, sample['noise_percent'], noise_seed)
        out_grid = rotated_grid(grid, rotation)
        factor = self.scale if normalized else 1.
        return dict(sharp=sharp/factor, clean=clean/factor, blurred=blurred/factor,
                    metadata={**sample, 'component': COMPONENTS[component], 'epoch': epoch,
                              'noise_seed_component': noise_seed, 'sigma_uT': sigma, 'grid': out_grid,
                              'normalized': normalized, 'scale_uT': self.scale,
                              'operator': {'kind': 'analytic', 'dz_um': sample['dz_um'],
                                           'pad_pixels': self.manifest['boundary']['pad_pixels'],
                                           'boundary': self.manifest['boundary']['mode']}})

    def iter_components(self, split='train', *, epoch=0, normalized=False):
        if split not in ('train', 'validation', 'test', 'all'):
            raise ValueError('Unknown split')
        for i, sample in enumerate(self.samples):
            if split in ('all', sample['split']):
                for component in COMPONENTS:
                    yield self.component(i, component, epoch=epoch, normalized=normalized)


def export(root, output, split='train', limit=None):
    if limit is not None and limit < 1:
        raise ValueError('Export limit must be positive')
    dataset = SyntheticDataset(root)
    if split not in ('train', 'validation', 'test', 'all'):
        raise ValueError('Unknown split')
    if not any(split in ('all', s['split']) for s in dataset.samples):
        raise ValueError('Selected split is empty')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    records = []
    write_json(output/'status.json', {'status': 'exporting'})
    for item in dataset.iter_components(split):
        meta = item['metadata']; name = meta['id']+'_'+meta['component']
        path = output/(name+'.npz')
        np.savez_compressed(path, sharp=item['sharp'], clean=item['clean'], blurred=item['blurred'])
        records.append({'path': path.name, 'sha256': digest(path), **meta})
        print(f'Exported {len(records)}: {name}', flush=True)
        if limit is not None and len(records) >= limit:
            break
    write_json(output/'manifest.json', {'dataset_kind': KIND+'_materialized', 'version': 1,
                                       'field_unit': 'uT', 'source_manifest_sha256': digest(dataset.root/'manifest.json'),
                                       'component_pairs': records})
    write_json(output/'status.json', {'status': 'complete', 'component_pairs': len(records)})
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare', help='Copy references, audit boundaries and write compact recipes')
    sources = p.add_mutually_exclusive_group(required=True)
    sources.add_argument('--source-manifest', type=Path)
    sources.add_argument('--source-run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--height-count', type=int, default=10)
    p.add_argument('--min-dz-um', type=float, default=.5)
    p.add_argument('--noise-percent', type=float, nargs='+', default=[0., 1., 1.5])
    p.add_argument('--rotations', type=int, nargs='+', default=[0])
    p.add_argument('--realizations', type=int, default=1)
    p.add_argument('--pad', type=int, default=256)
    p.add_argument('--interior-margin', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--validation-geometry', nargs='*', default=[])
    p.add_argument('--test-geometry', nargs='*', default=[])
    p.add_argument('--geometries', nargs='+')
    p = commands.add_parser('inspect', help='Verify reference checksums and print inventory')
    p.add_argument('--dataset', type=Path, required=True)
    p = commands.add_parser('export', help='Materialize component NPZ files in physical units')
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'validation', 'test', 'all'], default='train')
    p.add_argument('--limit', type=int, help='Maximum component pairs, for a bounded smoke export')
    args = vars(parser.parse_args(argv)); command = args.pop('command')
    if command == 'prepare':
        args['interior_margin'] = args.pop('interior_margin')
        prepare(**args)
    elif command == 'inspect':
        dataset = SyntheticDataset(args['dataset'])
        print(json.dumps(dataset.manifest['inventory'], indent=2))
    else:
        export(args.pop('dataset'), **args)


if __name__ == '__main__':
    main()
