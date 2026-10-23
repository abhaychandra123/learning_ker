"""Physical transforms, noise calibration, provenance and portable dataset checks."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from synthetic_data import (SyntheticDataset, add_noise, digest, export, prepare,
                            propagate, rotated_grid, rotate_component, rotation_component,
                            source_inventory, stable_seed, transfer)


def source_fixture(root, heldout_scale=20):
    root.mkdir()
    y, x = np.mgrid[:20, :24]
    grid = dict(nx=24, ny=20, x0=-3., y0=-2., dx=.3, dy=.4)
    base = np.stack([np.sin(x*.4)*np.cos(y*.2), np.cos(x*.2+y*.1), np.sin(y*.3)])
    manifest = dict(version=1, components=['Bx', 'By', 'Bz'], field_unit='uT',
                    length_unit='um', geometries={}, pairs=[])
    for g, factor in [('A', 2), ('B', heldout_scale)]:
        (root/g).mkdir()
        fields = (base*factor).astype(np.float32)
        blurred = np.stack([propagate(a, 1, .3, .4, 4) for a in fields]).astype(np.float32)
        planes = {}
        for z, a in [(.5, fields), (1.5, blurred)]:
            p = root/g/f'{z}.npy'; np.save(p, a)
            planes[str(z)] = dict(path=str(p.relative_to(root)), sha256=digest(p))
        manifest['geometries'][g] = dict(grid=grid, planes=planes)
        manifest['pairs'].append(dict(id=g+'_pair', geometry=g, sharp_z_um=.5, blur_z_um=1.5,
                                      dz_um=1., sharp=planes['0.5']['path'], blurred=planes['1.5']['path']))
    path = root/'manifest.json'; path.write_text(json.dumps(manifest))
    return path


class SyntheticTests(unittest.TestCase):
    def make(self, root, output='generated', **kwargs):
        options = dict(source_manifest=root/'source/manifest.json', height_count=2,
                       min_dz_um=.5, pad=4, interior_margin=2,
                       noise_percent=[0, 1, 1.5], rotations=[0, 90], test_geometry=['B'])
        options.update(kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare(root/output, **options)

    def test_periodic_fourier_mode_and_zero_distance(self):
        y, x = np.mgrid[:30, :40]; dx, dy = .2, .3
        field = np.cos(2*np.pi*(3*x/40 + 2*y/30))
        expected = field*np.exp(-2*np.pi*np.hypot(3/(40*dx), 2/(30*dy))*.7)
        np.testing.assert_allclose(propagate(field, .7, dx, dy, 0), expected, atol=1e-14)
        np.testing.assert_allclose(propagate(field, 0, dx, dy, 6), field, atol=1e-14)
        with self.assertRaises(ValueError):
            propagate(field, -.5, dx, dy, 0)

    def test_zero_padded_operator_adjoint_and_rotation_consistency(self):
        rng = np.random.default_rng(12)
        a, b = rng.normal(size=(2, 19, 26))
        aa = propagate(a, .8, .2, .35, 7); ab = propagate(b, .8, .2, .35, 7)
        self.assertAlmostEqual(float(np.vdot(aa, b)), float(np.vdot(a, ab)), places=12)
        expected = rotate_component(aa, 90)
        actual = propagate(rotate_component(a, 90), .8, .35, .2, 7)
        np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_physical_vector_rotation_with_y_increasing_rows(self):
        fields = np.zeros((3, 3, 3)); fields[0, 1, 2] = 1.
        rotated = np.stack([rotate_component(fields[rotation_component(c, 90)[0]], 90,
                                             rotation_component(c, 90)[1]) for c in range(3)])
        # +x vector at the right of centre becomes +y at the top in physical coordinates.
        self.assertEqual(rotated[1, 2, 1], 1.)
        self.assertEqual(np.count_nonzero(rotated), 1)
        by = np.zeros_like(fields); by[1, 1, 2] = 1
        original, sign = rotation_component(0, 90)
        self.assertEqual(rotate_component(by[original], 90, sign)[2, 1], -1)
        grid = rotated_grid(dict(nx=24, ny=20, x0=0, y0=0, dx=.2, dy=.4), 90)
        self.assertEqual((grid['nx'], grid['ny'], grid['dx'], grid['dy']), (20, 24, .4, .2))

    def test_noise_standard_deviation_and_seed_reproducibility(self):
        clean = np.linspace(-20, 30, 700*700, dtype=np.float32).reshape(700, 700)
        for percentage in (1., 1.5):
            noisy, sigma = add_noise(clean, percentage, 53)
            residual = noisy.astype(np.float64)-clean
            self.assertEqual(sigma, percentage*.5)
            self.assertLess(abs(residual.mean()), sigma*.005)
            self.assertLess(abs(residual.std()/sigma-1), .005)
            np.testing.assert_array_equal(noisy, add_noise(clean, percentage, 53)[0])
        np.testing.assert_array_equal(clean, add_noise(clean, 0, 53)[0])
        flat = np.zeros((3, 4), dtype=np.float32)
        np.testing.assert_array_equal(flat, add_noise(flat, 1.5, 53)[0])

    def test_portable_inventory_splits_train_scale_and_noise_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source_fixture(root/'source')
            m = self.make(root)
            self.assertEqual(m['inventory']['vector_pairs'], 24)
            self.assertEqual(m['inventory']['component_pairs'], 72)
            self.assertEqual(m['inventory']['vector_pairs_by_split'], {'train': 12, 'test': 12})
            self.assertAlmostEqual(m['normalization']['scale_uT'], 2.)
            shutil.move(root/'generated', root/'moved')
            shutil.rmtree(root/'source')
            ds = SyntheticDataset(root/'moved')
            i = next(i for i,s in enumerate(ds.samples) if s['split']=='train' and s['noise_percent']==1)
            a=ds.component(i,'Bx'); b=ds.component(i,'Bx',epoch=1)
            np.testing.assert_array_equal(a['sharp'], b['sharp'])
            np.testing.assert_array_equal(a['clean'], b['clean'])
            self.assertFalse(np.array_equal(a['blurred'], b['blurred']))
            np.testing.assert_array_equal(a['blurred'],ds.component(i,'Bx')['blurred'])
            np.testing.assert_allclose(ds.component(i,'Bx',normalized=True)['blurred']*ds.scale,a['blurred'])
            j=next(i for i,s in enumerate(ds.samples) if s['split']=='test')
            with self.assertRaisesRegex(ValueError,'Only training'):
                ds.component(j,'Bz',epoch=1)

    def test_source_checksum_output_checksums_and_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source_fixture(root/'source'); self.make(root)
            path=root/'generated/references/A.npy'; a=np.load(path); a[0,0,0]+=1; np.save(path,a)
            with self.assertRaisesRegex(ValueError,'checksum'):
                SyntheticDataset(root/'generated')
            m=json.loads((root/'generated/manifest.json').read_text())
            m['geometries']['A']['reference_path']='../outside.npy'
            (root/'generated/manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'escapes'):
                SyntheticDataset(root/'generated',verify=False)
            path=root/'source/A/0.5.npy'; a=np.load(path); a[0,0,0]+=1; np.save(path,a)
            with self.assertRaisesRegex(ValueError,'checksum'):
                self.make(root,'bad')
            self.assertEqual(json.loads((root/'bad/status.json').read_text())['status'],'failed')

    def test_boundary_audit_and_comparison_source_and_bounded_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=source_fixture(root/'source'); m=json.loads(path.read_text())
            run=root/'run'; (run/'comparison').mkdir(parents=True)
            (run/'setup.json').write_text(json.dumps({'manifest':m}))
            for pair in m['pairs']:
                np.savez_compressed(run/'comparison'/f"{pair['id']}.npz",
                                    sharp=np.load(root/'source'/pair['sharp']),
                                    blurred=np.load(root/'source'/pair['blurred']))
            self.make(root,source_manifest=None,source_run=run)
            audit=json.loads((root/'generated/boundary_audit.json').read_text())
            for row in audit['rows']:
                if row['pad_pixels']==4:
                    self.assertLess(row['full']['rmse_uT'],1e-6)
            ds=SyntheticDataset(root/'generated')
            with contextlib.redirect_stdout(io.StringIO()):
                records=export(root/'generated',root/'exported',limit=2)
            self.assertEqual(len(records),2)
            with np.load(root/'exported'/records[0]['path']) as item:
                np.testing.assert_array_equal(item['blurred'],ds.component(0,'Bx')['blurred'])
            with self.assertRaises(FileExistsError):
                export(root/'generated',root/'exported',limit=1)

    def test_invalid_configuration_and_split_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source_fixture(root/'source')
            for options in [dict(noise_percent=[-1]),dict(rotations=[45]),dict(pad=-1),
                            dict(validation_geometry=['B']),dict(min_dz_um=2)]:
                with self.assertRaises(ValueError):self.make(root,**options)
                self.assertFalse((root/'generated').exists())
            self.make(root)
            m=json.loads((root/'generated/manifest.json').read_text());m['samples'][0]['split']='test'
            (root/'generated/manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'cross'):
                SyntheticDataset(root/'generated')


if __name__ == '__main__':
    unittest.main()
