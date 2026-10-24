"""NumPy-only experiment selection, provenance checks and physical metrics."""
import json
import math
import numpy as np
from synthetic_data import COMPONENTS, digest, inside, validate_fields


def select_samples(dataset, actual_manifest, subset):
    """Match original heights without using actual input values or scores."""
    if any(s['split'] != 'train' for s in dataset.samples):
        raise ValueError('This same-geometry augmentation experiment requires an all-train dataset')
    if subset == 'full':
        return list(range(len(dataset.samples)))
    indices = []
    for pair in actual_manifest['pairs']:
        matches = [i for i, s in enumerate(dataset.samples)
                   if s['geometry'] == pair['geometry']
                   and math.isclose(s['reference_z_um'], pair['sharp_z_um'], abs_tol=1e-9)
                   and math.isclose(s['target_z_um'], pair['blur_z_um'], abs_tol=1e-9)
                   and s['rotation_degrees'] == 0 and s['noise_percent'] == 0]
        if len(matches) != 1:
            raise ValueError(f"Expected one noiseless unrotated recipe for {pair['id']}; got {len(matches)}")
        indices.append(matches[0])
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('Empty or duplicate matched-original recipes')
    return indices


def verify_actual_run(root, dataset):
    """Check every archived input/target before spending GPU time."""
    setup = json.loads((root/'setup.json').read_text())
    manifest = setup['manifest']
    if (manifest.get('components') != list(COMPONENTS)
            or manifest.get('field_unit') != 'uT' or manifest.get('length_unit') != 'um'):
        raise ValueError('Actual COMSOL units/components mismatch')
    if set(manifest['geometries']) != set(dataset.references):
        raise ValueError('Actual and synthetic geometry sets must match')
    hashes = {'setup.json': digest(root/'setup.json')}
    seen = set()
    for pair in manifest['pairs']:
        if pair['id'] in seen:
            raise ValueError('Duplicate actual pair ID')
        seen.add(pair['id'])
        g = pair['geometry']
        grid = manifest['geometries'][g]['grid']
        entry = dataset.manifest['geometries'][g]
        if grid != entry['grid'] or pair['sharp_z_um'] != entry['reference_z_um']:
            raise ValueError('Actual grid/reference height does not match synthetic data')
        if not math.isclose(pair['dz_um'], pair['blur_z_um']-pair['sharp_z_um']) or pair['dz_um'] <= 0:
            raise ValueError('Invalid actual propagation distance')
        name = f"comparison/{pair['id']}.npz"
        path = inside(root, name)
        hashes[name] = digest(path)
        with np.load(path, allow_pickle=False) as a:
            for field in ('sharp', 'blurred', 'lista'):
                validate_fields(a[field], grid)
            if not np.array_equal(a['sharp'], dataset.references[g]):
                raise ValueError('Actual and synthetic reference fields differ')
    if not seen:
        raise ValueError('No actual COMSOL evaluation pairs')
    return manifest, hashes


def field_metrics(prediction, truth):
    x, y = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((x-y)**2)))
    span, rms = float(np.ptp(y)), float(np.sqrt(np.mean(y*y)))
    return dict(rmse_uT=rmse, nrmse_range=rmse/span if span else None,
                relative_l2=rmse/rms if rms else None,
                amplitude_slope=float(np.mean(x*y)/(rms*rms)) if rms else None)


