"""Auditable adapter for the frozen upstream Haut Glacier d'Arolla example."""
from pathlib import Path
import hashlib
import json
import numpy as np


def prepare_arolla(out):
    from scipy.io import loadmat
    from .runner import demo_config,write_json
    source=Path(__file__).resolve().parents[1]/'reference/geo_arolla.mat'
    geo=loadmat(source,squeeze_me=True,struct_as_record=False)['geo']
    x=np.asarray(geo.xi,dtype=float); bed=np.asarray(geo.hB,dtype=float); h=np.asarray(geo.H,dtype=float)
    out.mkdir(parents=True,exist_ok=False)
    # Use the stored H verbatim; do not add the extra Hmin in the MATLAB driver.
    np.savetxt(out/'geometry.csv',np.column_stack((x,bed,h)),delimiter=',',header='x_m,bed_m,thickness_m',comments='',fmt='%.16g')
    note=('PoLIM distributed Arolla profile; stored H retained verbatim, including upstream 0.1 m offset. '
          'Wsurf has 31 values for 51 x nodes; explicit unit-width flowline, not observed total glacier volume. '
          'Temperature, geothermal flux and SMB are demonstration assumptions, not Arolla observations.')
    provenance={'source':'https://github.com/WangYuzhe/PoLIM-Polythermal-Land-Ice-Model',
                'commit':'cc5d08541ab67987f7652adcdc7a734530580e82','source_file':'PoLIM_classic/geo_inputs/geo_arolla.mat',
                'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'nodes':len(x),
                'upstream_width_count':int(np.size(geo.Wsurf)),
                'max_stored_H_minus_surface_minus_bed_m':float(np.max(abs(h-(geo.hS-bed)))),
                'unit_width':True,'note':note,'scientific_acceptance':False}
    write_json(out/'provenance.json',provenance)
    config=demo_config(); config.update(case_id='arolla-profile',dt_years=.01,steps=5)
    config['geometry']={'kind':'csv','path':'geometry.csv','unit_width':True,'note':note}
    config['parameters'].update(layers=9,flowband=False)
    write_json(out/'case.json',config)
    return str((out/'case.json').resolve())
