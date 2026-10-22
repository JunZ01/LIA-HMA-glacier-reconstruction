"""Batch version of export_fullchain_inputs_npz_20261021.py for the pilot list.

Same arrays, same sampling, same checks as the three-system exporter; the only
differences are: systems come from pilot_systems.csv, the output directory is
an argument, the comparison against the earlier modern_control_<uid>.npz runs
only where such a file exists, and one failing system is recorded in the
manifest instead of stopping the batch. Read-only: no tau fit, PP, LIA,
climate or dynamic run. Needs oggm_env2 (xarray, pyproj, rasterio).

Usage (repo root, oggm_env2):
    python export_fullchain_inputs_batch_20261021.py PILOT_CSV OUT_DIR
"""

from pathlib import Path
import gzip
import hashlib
import json
import pickle
import shutil
import sys
import traceback

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from rasterio.transform import Affine
from scipy.ndimage import map_coordinates

import diagnose_extension_operators_20261021 as diag
from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.oggm_flowline_io import load_flowline_inputs
from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.gridded_bed import load_gridded_bed

PROJECT = Path('E:/3spyderProject/1_LIA_himalaya_oggm')
CONFIG = PROJECT / 'validation_data/results/paleo_surface_validation_v0/core191_production_candidate_v1/core191_tau_calibration_config_v1.json'
TAU = CONFIG.parent / 'modern_tau_calibration_v1/modern_tau_calibration_v0.csv'
PRIOR = diag.BASE / 'extension_operator_controls_20261021_v1/npz_export'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample(values, xy, order=1):
    return map_coordinates(values, np.vstack([xy[:, 1], xy[:, 0]]), order=order,
                           mode='constant', cval=np.nan, prefilter=False)


def export_one(case, config, tau_table, out_dir, *, file_stem=None, require_tau=True, check_prior=True,
               config_source_path=None):
    """Export one modern gdir. Defaults reproduce the 22-system pilot exporter exactly.

    ``file_stem`` names the NPZ (member exports use '<uid>_<rgi_id>');
    ``require_tau=False`` records NaN legacy tau instead of failing when the
    archived calibration has no row for this gdir (non-main members);
    ``check_prior=False`` skips the comparison with the earlier single-system
    modern-control exports.
    """
    uid = int(case['lia_unique_id'])
    d = diag.load_case(case)
    gdir = Path(case['gdir'])
    inputs, raw = load_flowline_inputs(gdir)
    native = load_gridded_bed(gdir, 'farinotti2019')
    grid = json.loads((gdir / 'glacier_grid.json').read_text(encoding='utf-8'))
    sx, sy = grid['dxdy']
    x0, y0 = grid['x0y0']
    assert native.map_dx_m == abs(sx)
    with xr.open_dataset(gdir / 'gridded_data.nc') as ds:
        native_topo = np.asarray(ds.topo, dtype=np.float64)
        native_h = np.asarray(ds[diag.THICK_VAR], dtype=np.float64)
    with xr.open_dataset(case['lia_netcdf']) as ds:
        x = np.asarray(ds.x, dtype=np.float64)
        y = np.asarray(ds.y, dtype=np.float64)
        crs = ds.spatial_ref.attrs['crs_wkt']
        transform = np.fromstring(ds.spatial_ref.attrs['GeoTransform'], sep=' ')
    with gzip.open(gdir / diag.INVERSION, 'rb') as stream:
        inversion = pickle.load(stream)
    offsets = np.concatenate([[0], np.cumsum([len(i.coords_xy) for i in inputs])]).astype(np.int64)
    xy_native = np.concatenate([i.coords_xy for i in inputs])
    world_native = np.column_stack([x0 + xy_native[:, 0] * sx, y0 + xy_native[:, 1] * sy])
    tx, ty = Transformer.from_crs(grid['proj'], crs, always_xy=True).transform(world_native[:, 0], world_native[:, 1])
    world_target = np.column_stack([tx, ty])
    xy_target = np.column_stack([(tx - x[0]) / (x[1] - x[0]), (ty - y[0]) / (y[1] - y[0])])
    target_bed = d['topo'] - d['hw']
    fl_surface = np.concatenate([i.reference_surface_m for i in inputs])
    inv_h = np.concatenate([np.asarray(i['thick'], dtype=np.float64) for i in inversion])
    inv_volume = np.concatenate([np.asarray(i['volume'], dtype=np.float64) for i in inversion])
    array = {
        'schema_version': np.asarray('modern-fullchain-inputs-v1'),
        'lia_unique_id': np.asarray(uid), 'region': np.asarray(case['region']), 'rgi_id': np.asarray(case['rgi_id']),
        'reference_H_m': d['hw'], 'topo_m': d['topo'], 'modern_mask': d['modern'], 'bed_m': target_bed,
        'dx_m': np.asarray(d['dx']), 'grid_x_m': x, 'grid_y_m': y, 'crs_wkt': np.asarray(crs),
        'grid_transform_gdal': transform,
        'constraint_rows': d['rr'].astype(np.int32), 'constraint_cols': d['cc'].astype(np.int32),
        'native_reference_H_m': native_h, 'native_topo_m': native_topo,
        'native_modern_mask': native.glacier_mask, 'native_bed_m': native.bed_m,
        'native_dx_m': np.asarray(native.map_dx_m), 'native_grid_json': np.asarray(json.dumps(grid)),
        'native_transform_gdal': np.asarray(Affine(sx, 0, x0 - .5*sx, 0, sy, y0 - .5*sy).to_gdal()),
        'native_volume_km3': np.asarray(d['native_volume']),
        'inversion_1d_volume_km3': np.asarray(d['inversion_1d_volume']),
        'flowline_offsets': offsets,
        'flowline_index': np.asarray([i.flowline_index for i in inputs], dtype=np.int32),
        'flowline_flows_to': np.asarray([-1 if i.flows_to is None else i.flows_to for i in inputs], dtype=np.int32),
        'flowline_flows_to_indice': np.asarray([-1 if i.flows_to_indice is None else i.flows_to_indice for i in inputs], dtype=np.int32),
        'flowline_dx_m': np.asarray([i.dx_meter for i in inputs]),
        'flowline_distance_m': np.concatenate([np.arange(len(i.coords_xy)) * i.dx_meter for i in inputs]),
        'flowline_xy_native': xy_native, 'flowline_xy_lia': xy_target,
        'flowline_xy_projected_native_m': world_native, 'flowline_xy_projected_lia_m': world_target,
        'flowline_surface_oggm_m': fl_surface,
        'flowline_inversion_H_m': inv_h, 'flowline_inversion_volume_m3': inv_volume,
        'flowline_inversion_bed_m': fl_surface - inv_h,
        'flowline_bed_native_bilinear_m': sample(native.bed_m, xy_native),
        'flowline_bed_lia_bilinear_m': sample(target_bed, xy_target),
        'flowline_bed_native_nearest_m': sample(native.bed_m, xy_native, order=0),
        'flowline_bed_lia_nearest_m': sample(target_bed, xy_target, order=0),
        'flowline_surface_dem_native_m': sample(native_topo, xy_native),
        'flowline_surface_dem_lia_m': sample(d['topo'], xy_target),
        'flowline_reference_H_native_m': sample(np.where(native.glacier_mask, np.nan_to_num(native_h), 0.0), xy_native),
        'flowline_reference_H_lia_m': sample(d['hw'], xy_target),
        'shape_factor': np.asarray(config['modern_tau_calibration']['shape_factor']),
        'ice_density_kg_m3': np.asarray(900.0), 'gravity_m_s2': np.asarray(9.81),
        'maximum_thickness_m': np.asarray(2000.0),
        'legacy_config_json': np.asarray(json.dumps(config, ensure_ascii=False)),
    }
    tau_rows = tau_table[(tau_table.rgi_id == case['rgi_id']) & (tau_table.bed_scenario == 'farinotti2019')]
    if tau_rows.empty and require_tau:
        raise ValueError('no archived legacy tau row for this rgi_id/farinotti2019')
    if tau_rows.empty:
        array['legacy_tau_kpa'] = np.asarray(np.nan)
        array['legacy_tau_row_json'] = np.asarray('{}')
    else:
        tau_row = tau_rows.iloc[0]
        array['legacy_tau_kpa'] = np.asarray(float(tau_row.tau_kpa))
        array['legacy_tau_row_json'] = np.asarray(json.dumps({k: None if pd.isna(v) else v for k, v in tau_row.to_dict().items()}, allow_nan=False))
    for key in ['flowline_bed_native_bilinear_m', 'flowline_bed_lia_bilinear_m',
                'flowline_surface_dem_native_m', 'flowline_surface_dem_lia_m']:
        if not np.isfinite(array[key]).all():
            raise ValueError(f'non-finite exported profile {key}; no filling or snapping performed')
    assert offsets[-1] == len(xy_native) == len(inv_h)
    for i, parent in enumerate(array['flowline_flows_to']):
        join = array['flowline_flows_to_indice'][i]
        assert parent == -1 or (0 <= parent < len(inputs) and 0 <= join < offsets[parent+1]-offsets[parent])
    assert np.isclose(inv_volume.sum()/1e9, d['inversion_1d_volume'], rtol=1e-12)
    prior = PRIOR / f'modern_control_{uid}.npz'
    prior_checked = check_prior and prior.exists()
    if prior_checked:
        with np.load(prior, allow_pickle=False) as previous:
            assert np.array_equal(previous['modern_mask'], d['modern'])
            assert np.array_equal(previous['constraint_rows'], d['rr'])
            assert np.array_equal(previous['constraint_cols'], d['cc'])
            assert np.array_equal(previous['reference_H_m'], d['hw'].astype(np.float32))
    source_paths = [gdir/'gridded_data.nc', gdir/'glacier_grid.json', gdir/'inversion_flowlines.pkl',
                    gdir/diag.INVERSION, Path(case['lia_netcdf']), Path(config_source_path) if config_source_path else CONFIG, TAU]
    source_hashes = {str(p): sha(p) for p in source_paths}
    array['source_manifest_json'] = np.asarray(json.dumps(source_hashes))
    path = out_dir / f'fullchain_inputs_{file_stem or uid}.npz'
    np.savez_compressed(path, **array)
    with np.load(path, allow_pickle=False) as loaded:
        assert all(loaded[k].dtype.kind != 'O' for k in loaded.files)
    if any(sha(p) != h for p, h in source_hashes.items()):
        raise RuntimeError('source changed during export')
    return {'file': path.name, 'uid': uid, 'rgi_id': case['rgi_id'], 'bytes': path.stat().st_size,
            'sha256': sha(path), 'flowlines': len(inputs), 'flowline_nodes': int(offsets[-1]),
            'native_dx_m': native.map_dx_m, 'target_dx_m': d['dx'],
            'modern_area_km2': float(d['modern'].sum() * d['dx']**2 / 1e6),
            'reference_volume_km3': float(d['hw'][d['modern']].sum() * d['dx']**2 / 1e9),
            'prior_control_checked': prior_checked, 'source_hashes': source_hashes}


def main():
    pilot = pd.read_csv(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    tau_table = pd.read_csv(TAU)
    records = []
    for case in pilot.to_dict('records'):
        try:
            record = {'status': 'exported', **export_one(case, config, tau_table, out_dir)}
        except Exception as error:  # recorded, never silently dropped
            record = {'status': 'export-failed', 'uid': int(case['lia_unique_id']), 'rgi_id': case['rgi_id'],
                      'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k not in ('source_hashes', 'traceback')}), flush=True)
    manifest = {'status': 'completed', 'schema': 'modern-fullchain-inputs-v1',
                'scope': 'read-only data export; no tau calibration, PP, LIA or dynamic solve',
                'exported': sum(r['status'] == 'exported' for r in records),
                'failed': sum(r['status'] != 'exported' for r in records), 'records': records}
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    shutil.copy2(__file__, out_dir / Path(__file__).name)
    shutil.copy2(sys.argv[1], out_dir / 'pilot_systems.csv')
    print('EXPORTED', manifest['exported'], 'FAILED', manifest['failed'], out_dir, flush=True)


if __name__ == '__main__':
    main()
