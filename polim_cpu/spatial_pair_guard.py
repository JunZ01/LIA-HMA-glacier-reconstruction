"""Enforce the scientific source/network pair before consuming spatial profiles."""
import json
from pathlib import Path
from .v3.runner import PROJECT, digest


def require_paired_source(package_path,network_path):
    package=json.loads(Path(package_path).read_text(encoding='utf-8'))
    if package.get('schema')!='polim-paired-spatial-source-v1':
        raise ValueError('Spatial source has no registered V3 source/network pair')
    sid=int(package['system']); source=PROJECT/package['source']['path']; network=Path(network_path).resolve()
    expected_source=PROJECT/'validation_data/results/paleo_surface_validation_v0/core191_anchor_reroute_f19_v3'
    expected_network=PROJECT/f'validation_data/results/paleo_surface_validation_v0/lia_gdir_anchor_reroute_core191_v3/{sid}'
    if source.parent.resolve()!=expected_source.resolve() or not source.name.startswith(f'lia_{sid}_'):
        raise ValueError('Source is outside the authoritative V3/F19 family')
    if expected_network.resolve() not in network.parents or network.name!='lia_centerline_network.pkl.gz':
        raise ValueError('Network/system crosswalk mismatch')
    if (PROJECT/package['network']['path']).resolve()!=network:
        raise ValueError('Package identifies a different network')
    for path,key in [(source,'source'),(network,'network'),(PROJECT/package['array_archive']['path'],'array_archive')]:
        if digest(path)!=package[key]['sha256'].lower(): raise ValueError('Paired source checksum mismatch: '+key)
    return package
