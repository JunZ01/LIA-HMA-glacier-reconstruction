"""Static source binding: preserve volume ledger without selecting momentum H.

Zero-area geometric nodes remain in the profile archive and node crosswalk.
They have no stored mass and are bypassed only in the transport graph.
"""
import json
from pathlib import Path
import numpy as np
from ..v3.runner import PROJECT,digest,write_json
from ..spatial_pair_guard import require_paired_source


def bind(profile_path,out):
    path=Path(profile_path); package=json.loads(path.read_text()); source_path=PROJECT/package['source']['path']
    if digest(source_path)!=package['source']['sha256'].lower(): raise ValueError('Source package changed')
    source=json.loads(source_path.read_text()); paired=require_paired_source(source_path,PROJECT/source['network']['path'])
    if not package['status'].startswith('paired-source-diagnostics-'): raise ValueError('Profile source pairing unverified')
    count=sum(b['node_count'] for b in package['branches']); successor=np.full(count,-2,dtype=np.int64)
    fields={k:np.zeros(count) for k in ('allocated_area_m2','allocated_volume_m3','easting_m','northing_m','centreline_thickness_m','centreline_bed_m','equivalent_thickness_m')}
    branch=np.full(count,-1,dtype=int)
    for b in package['branches']:
        p=PROJECT/b['file']['path']
        if digest(p)!=b['file']['sha256'].lower(): raise ValueError('Profile file changed')
        start=b['global_node_start']; n=b['node_count']; take=slice(start,start+n)
        with np.load(p,allow_pickle=False) as data:
            for key in fields: fields[key][take]=data[key]
        branch[take]=b['branch']; successor[start:start+n-1]=np.arange(start+1,start+n)
        successor[start+n-1]=b['junction']['parent_global_node'] if b['junction'] else -1
    if np.any(successor== -2) or np.any(branch<0): raise ValueError('Incomplete profile topology')
    area=fields['allocated_area_m2']; volume=fields['allocated_volume_m3']; positive=area>0
    if np.any(volume[~positive]!=0) or np.any(area<0): raise ValueError('Zero-area node contains mass')
    def downstream(i,include=False):
        visited=set(); j=i if include else int(successor[i])
        while j>=0:
            if j in visited: raise ValueError('Cycle in downstream crosswalk')
            visited.add(j)
            if positive[j]: return j
            j=int(successor[j])
        return -1
    # Verify the entire source graph is rooted, including non-storage nodes.
    for i in range(count):
        j=i; seen=set()
        while j>=0:
            if j in seen: raise ValueError('Cycle in source graph')
            seen.add(j); j=int(successor[j])
    keep=np.flatnonzero(positive); inverse=np.full(count,-1,dtype=np.int64); inverse[keep]=np.arange(len(keep))
    targets=np.array([downstream(int(i)) for i in keep]); receiver=np.array([inverse[i] if i>=0 else -1 for i in targets])
    edges=np.column_stack((np.arange(len(keep)),receiver)); crossing=(receiver>=0)&(branch[keep]!=branch[keep[np.maximum(receiver,0)]])
    geometric_to_storage=np.array([inverse[j] if (j:=downstream(i,True))>=0 else -1 for i in range(count)])
    with np.load(PROJECT/package['partition']['path'],allow_pickle=False) as allocation:
        if digest(PROJECT/package['partition']['path'])!=package['partition']['sha256'].lower(): raise ValueError('Pixel allocation changed')
        owner=allocation['node_id']; mask=allocation['source_ice_mask']; source_counts=np.bincount(owner[mask],minlength=count)
    if not np.array_equal(source_counts>0,positive): raise ValueError('Storage nodes differ from pixel ledger')
    if not np.isclose(volume.sum(),paired['volume_m3'],rtol=1e-12) or not np.isclose(area.sum(),paired['area_m2'],rtol=1e-12): raise ValueError('Source ledger closure failed')
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    arrays={'edges':edges,'original_node_id':keep,'geometry_successor':successor,'geometry_to_storage':geometric_to_storage,
            'initial_volume_m3':volume[keep],'source_area_m2':area[keep],'branch_index':branch[keep],
            'easting_m':fields['easting_m'][keep],'northing_m':fields['northing_m'][keep],
            'centreline_thickness_m':fields['centreline_thickness_m'][keep],
            'equivalent_thickness_m':fields['equivalent_thickness_m'][keep],
            'junction_edge_mask':crossing,'pixel_storage_owner':np.where(mask,inverse[np.maximum(owner,0)],-1)}
    with (out/'network.npz').open('xb') as f: np.savez_compressed(f,**arrays)
    result={'schema':'polim-v4-static-network-binding-v1','source_profile':{'path':str(path.resolve()),'sha256':digest(path)},
            'paired_source_sha256':digest(source_path),'geometry_nodes':count,'storage_nodes':len(keep),
            'zero_mass_geometry_nodes_retained':int((~positive).sum()),'edges':len(edges),'junction_edges':int(crossing.sum()),
            'initial_source_volume_m3':float(volume.sum()),'initial_source_area_m2':float(area.sum()),
            'network_sha256':digest(out/'network.npz'),'momentum_thickness_selected':False,
            'flux_computed':False,'climate_decoded':False,'ice_advanced':False,'scientific_acceptance':False,
            'boundary':'Explicit outlet edge; caller must supply flux. No new downstream domain fabricated.',
            'limitations':'Static diagnostic geometry still contains reverse slopes and inferred connectors; no real glacier dynamics authorized by this binding.'}
    write_json(out/'binding.json',result); return result
