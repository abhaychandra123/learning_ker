"""Prepare geometry-separated COMSOL caches and explicit reconstruction pairs.

The existing primary cache is never modified. Paths in manifest.json are
relative to that manifest so the prepared dataset can be moved to an HPC.
"""
import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np

from kernel_crops import load_plane, sha256

COMPONENTS = ('Bx', 'By', 'Bz')
PATTERN = re.compile(r'^(L1_C1|IC_2um|IC_4um)_Z(\d+)p(\d+)um_.*\.csv$')


def discover(root):
    groups = {}
    for path in sorted(Path(root).rglob('*.csv')):
        match = PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f'Unrecognized geometry/height filename: {path}')
        geometry = match[1]
        z = float(f'{match[2]}.{match[3]}')
        planes = groups.setdefault(geometry, {})
        if z in planes:
            raise ValueError(f'Duplicate {geometry} height {z}: {path} and {planes[z]}')
        planes[z] = path
    if not groups or any(len(planes) < 2 for planes in groups.values()):
        raise ValueError('Need at least two planes per geometry')
    return groups


def height_pairs(heights, policy):
    heights = sorted(heights)
    if policy == 'anchor':
        return [(heights[0], z) for z in heights[1:]]
    if policy == 'all':
        return list(itertools.combinations(heights, 2))
    raise ValueError('Unknown pairing policy')


def tag(z):
    return f'z{z:04.1f}'


def prepare(raw, output, policy='anchor'):
    raw, output = Path(raw).resolve(), Path(output).resolve()
    groups = discover(raw)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(version=1, components=list(COMPONENTS), field_unit='uT',
                    length_unit='um', pairing=policy, geometries={}, pairs=[])
    for geometry, paths in groups.items():
        folder = output / geometry
        folder.mkdir()
        planes, reference_grid = {}, None
        for z, path in sorted(paths.items()):
            print(f'Converting {geometry} z={z:g} um: {path.name}', flush=True)
            fields, grid = load_plane(path)
            if reference_grid is not None and any(
                not np.isclose(grid[k], reference_grid[k], rtol=0, atol=1e-6)
                for k in grid
            ):
                raise ValueError(f'Grid mismatch within {geometry}: {path}')
            reference_grid = grid
            target = folder / f'{tag(z)}.npy'
            np.save(target, fields.astype(np.float32), allow_pickle=False)
            planes[str(z)] = dict(path=str(target.relative_to(output)),
                                 sha256=sha256(target), source=str(path.relative_to(raw)),
                                 source_sha256=sha256(path))
        manifest['geometries'][geometry] = dict(grid=reference_grid, planes=planes)
        for sharp_z, blur_z in height_pairs(paths, policy):
            manifest['pairs'].append(dict(
                id=f'{geometry}_{tag(blur_z)}_to_{tag(sharp_z)}', geometry=geometry,
                sharp_z_um=sharp_z, blur_z_um=blur_z, dz_um=blur_z-sharp_z,
                sharp=planes[str(sharp_z)]['path'], blurred=planes[str(blur_z)]['path']))
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print(f"Prepared {len(manifest['pairs'])} pairs / "
          f"{3*len(manifest['pairs'])} component pairs in {output}", flush=True)
    return manifest


def read_manifest(path, verify=True):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if (data.get('version') != 1 or data.get('field_unit') != 'uT'
            or data.get('length_unit') != 'um' or data.get('components') != list(COMPONENTS)):
        raise ValueError('Unsupported manifest schema/units/components')
    registered = {}
    for geometry, entry in data['geometries'].items():
        grid = entry['grid']
        for z, plane in entry['planes'].items():
            p = (path.parent/plane['path']).resolve()
            if not p.is_relative_to(path.parent):
                raise ValueError('Cache path escapes dataset root')
            if plane['path'] in registered:
                raise ValueError('Duplicate cache path')
            registered[plane['path']] = (geometry, float(z))
            arr = np.load(p, mmap_mode='r', allow_pickle=False)
            if arr.shape != (3, grid['ny'], grid['nx']) or not np.isfinite(arr).all():
                raise ValueError(f'Invalid cached field: {p}')
            if verify and sha256(p) != plane['sha256']:
                raise ValueError(f'Checksum mismatch: {p}')
    seen = set()
    if not data['pairs']:
        raise ValueError('Empty pair list')
    for pair in data['pairs']:
        if pair['id'] in seen or not re.fullmatch(r'[A-Za-z0-9_.-]+', pair['id']):
            raise ValueError('Invalid/duplicate pair ID')
        seen.add(pair['id'])
        for role, key in [('sharp', 'sharp_z_um'), ('blurred', 'blur_z_um')]:
            if registered.get(pair[role]) != (pair['geometry'], pair[key]):
                raise ValueError('Pair crosses geometries or mislabels its height')
        if pair['blur_z_um'] <= pair['sharp_z_um'] or not np.isclose(
                pair['dz_um'], pair['blur_z_um']-pair['sharp_z_um']):
            raise ValueError('Invalid pair heights')
    return data


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pairing', choices=('anchor', 'all'), default='anchor',
                   help='anchor: lowest plane per geometry; all: every lower/higher pair')
    p.add_argument('--inventory-only', action='store_true')
    args = p.parse_args(argv)
    if args.inventory_only:
        for geometry, planes in discover(args.raw_root).items():
            print(geometry, sorted(planes), f'{len(height_pairs(planes, args.pairing))} pairs')
    else:
        prepare(args.raw_root, args.output, args.pairing)


if __name__ == '__main__':
    main()
