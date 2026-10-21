"""Independent raster-cell traversal vs polygon-intersection width verification."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
from html.parser import HTMLParser
import gzip,hashlib,json,pickle
import numpy as np
import xarray as xr

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];COUNTS=Counter()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def close(a,b,atol=1e-7):assert np.allclose(a,b,rtol=3e-9,atol=atol,equal_nan=True),(a,b)
def numeric(x):return np.asarray([np.nan if v is None else v for v in x],float)
def empirical_quantiles(x,p):
    ordered=np.sort(x);locations=np.array(p)*(len(x)-1);i=np.floor(locations).astype(int);j=np.ceil(locations).astype(int)
    return ordered[i]+(ordered[j]-ordered[i])*(locations-i)


def cell_traversal_width(mask,x,y,p,normal):
    """Intersect line with regular grid boundaries and classify cell midpoints; no polygons."""
    dx=x[1]-x[0];dy=y[1]-y[0];xe=np.r_[x-dx/2,x[-1]+dx/2];ye=np.r_[y-dy/2,y[-1]+dy/2]
    crossings=[-10000.,10000.]
    for edges,origin,direction in [(xe,p[0],normal[0]),(ye,p[1],normal[1])]:
        if abs(direction)>1e-12:
            t=(edges-origin)/direction;crossings.extend(t[(t>-10000)&(t<10000)].tolist())
    edges=np.unique(np.array(crossings));mid=(edges[:-1]+edges[1:])/2
    points=p+mid[:,None]*normal
    cols=np.floor((points[:,0]-(x[0]-dx/2))/dx).astype(int);rows=np.floor((points[:,1]-(y[0]-dy/2))/dy).astype(int)
    good=(cols>=0)&(cols<len(x))&(rows>=0)&(rows<len(y));inside=np.zeros(len(mid),bool);inside[good]=mask[rows[good],cols[good]]
    # Locate the contiguous ice segment containing the centre. Boundary contact is handled by either neighbour.
    indices=np.where(inside&(edges[:-1]<=1e-8)&(edges[1:]>=-1e-8))[0]
    if not len(indices):return np.nan,False
    left=right=int(indices[0])
    while left>0 and inside[left-1]:left-=1
    while right+1<len(inside) and inside[right+1]:right+=1
    return float(edges[right+1]-edges[left]),bool(edges[left]<=-9999.999 or edges[right+1]>=9999.999)


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='a' and 'href' in a:self.links.append(a['href'])
        if tag=='img' and 'src' in a:self.links.append(a['src'])


def main():
    methods=read(OUT/'methods.json');validation=read(OUT/'validation.json')
    for p,h in read(OUT/'source_manifest.json').items():assert sha(ROOT/p)==h,p
    for p,h in methods['authority'].items():assert sha(ROOT/p)==h,p
    for p,h in read(OUT/'output_manifest.json').items():
        if p in [Path(__file__).name,'visual_review.json']:continue
        assert sha(OUT/p)==h,p
    # Synthetic traversal tests: vertical/horizontal, a hole, disconnected ice and centre outside.
    mask=np.ones((5,5),bool);x=np.arange(5,dtype=float);y=np.arange(4,-1,-1,dtype=float)
    close(cell_traversal_width(mask,x,y,np.array([2,2]),np.array([1.,0.]))[0],5)
    close(cell_traversal_width(mask,x,y,np.array([2,2]),np.array([0.,1.]))[0],5)
    mask[2,3]=False;close(cell_traversal_width(mask,x,y,np.array([2,2]),np.array([1.,0.]))[0],3)
    mask[2,2]=False;assert np.isnan(cell_traversal_width(mask,x,y,np.array([2,2]),np.array([1.,0.]))[0])
    close(empirical_quantiles(np.array([1,2,3,4]),[0,.5,1]),[1,2.5,4]);COUNTS['fixtures']=5
    stats={r['system']:r for r in read(OUT/'geometry_6.json')};profiles={r['system']:r for r in read(OUT/'profiles_6.json')}
    inv={r['lia_unique_id']:r for r in read(ROOT/'analysis/lia_51_inventory_v1/inventory.json')}
    oldhyp={r['system']:r for r in read(ROOT/'analysis/lia_51_systematic_analysis_v1/hypsometry.json')}
    assert set(stats)==set(profiles)=={1012,1044,2622,2827,2928,2939}
    for sid,s in stats.items():
        p=profiles[sid]
        with xr.open_dataset(ROOT/s['source_grid']) as ds:
            mi=list(ds.member_id.values.astype(str)).index('farinotti2019');z=ds.surface_elevation_m.isel(member=mi).values.astype(float);bed=ds.bed_elevation_m.isel(member=mi).values.astype(float);h=ds.thickness_m.isel(member=mi).values.astype(float)
            outline=ds.lia_mask.values>0;valid=(ds.valid_mask.isel(member=mi).values>0)&outline&np.isfinite(z)&np.isfinite(bed)&np.isfinite(h);ice=valid&(h>1e-6);x=ds.x.values;y=ds.y.values
            excluded=ds.excluded_component_mask.values>0
        dx,dy=x[1]-x[0],y[1]-y[0];cell=abs(dx*dy);zz=z[ice];hh=h[ice];rel=(zz-zz.min())/(zz.max()-zz.min())
        assert s['positive_cells']==ice.sum();close(s['area_km2'],len(zz)*cell/1e6);close(s['volume_km3'],hh.sum()*cell/1e9)
        close([s['outline_area_km2'],s['valid_area_km2'],s['excluded_area_km2']],[outline.sum()*cell/1e6,valid.sum()*cell/1e6,excluded.sum()*cell/1e6])
        close(s['positive_coverage'],ice.sum()/outline.sum());close(s['mean_thickness_m'],hh.mean());close(s['normalized_mean_height'],rel.mean())
        close([s['zmin_m'],s['zmax_m'],s['zmedian_m'],s['zmean_m']],[zz.min(),zz.max(),np.median(zz),zz.mean()])
        close(s['elevation_quantiles_m'],empirical_quantiles(zz,s['quantile_probabilities']));close(s['normalized_elevation_quantiles'],empirical_quantiles(rel,s['quantile_probabilities']))
        for key,value in s['relative_area_fractions'].items():close(value,np.count_nonzero(rel<float(key))/len(rel))
        for key,value in s['below_terminus_plus_fractions'].items():close(value,np.count_nonzero(zz<s['terminus_surface_m']+float(key))/len(zz))
        for width in [100,200,300]:
            bins=s['hypsometry'][str(width)];indices=np.floor((zz-2500)/width).astype(int);indices=np.minimum(indices,len(bins['counts'])-1)
            counts=np.bincount(indices,minlength=len(bins['counts']));assert np.array_equal(counts,bins['counts'])
            close(bins['area_km2'],counts*cell/1e6);close(bins['area_fraction'],counts/len(zz));COUNTS['histograms']+=1
        close(s['hypsometry']['200']['area_km2'],oldhyp[sid]['area_km2'])
        with gzip.open(ROOT/s['source_network'],'rb') as f:net=pickle.load(f)
        b=next(b for b in net['branches'] if b['branch_index']==net['main_branch_index']);xy=np.array(b['coords_projected_m']);chain=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))];length=chain[-1]
        close(length/1000,s['mainline_length_km']);close(length/1000,inv[sid]['lia_length_km']);d=np.array(p['distance_from_terminus_m']);stations=length-d
        points=np.c_[np.interp(stations,chain,xy[:,0]),np.interp(stations,chain,xy[:,1])];close(points,p['xy_projected_m'])
        def nearest(point):
            c=int(np.rint((point[0]-x[0])/dx));r=int(np.rint((point[1]-y[0])/dy))
            rr,cc=np.meshgrid(np.arange(max(0,r-3),min(len(y),r+4)),np.arange(max(0,c-3),min(len(x),c+4)),indexing='ij')
            rr=rr.ravel();cc=cc.ravel();ok=valid[rr,cc];rr=rr[ok];cc=cc[ok]
            distances=np.hypot(x[cc]-point[0],y[rr]-point[1]);k=int(np.argmin(distances));return rr[k],cc[k],distances[k]
        for i,point in enumerate(points):
            r,c,distance=nearest(point);assert distance<=s['sample_limit_m']
            close(p['distance_m'][i],distance)
            for key,array in [('surface_m',z),('bed_m',bed),('thickness_m',h)]:close(p[key][i],array[r,c])
            assert p['valid'][i];COUNTS['profile_points']+=1
        assert s['profile_count']==s['profile_valid_count']==len(points);close(s['profile_max_sample_distance_m'],max(p['distance_m']))
        close(p['surface_m'][0],inv[sid]['lia_terminus_surface_m']);close(p['surface_m'][-1],inv[sid]['lia_head_surface_m'])
        for zone in s['surface_slope_zones']:
            values=[];distances=[]
            for delta in [zone['start_m'],zone['end_m']]:
                q=np.array([np.interp(length-delta,chain,xy[:,0]),np.interp(length-delta,chain,xy[:,1])]);r,c,distance=nearest(q);values.append(z[r,c]);distances.append(distance)
            close(zone['surface_slope_deg'],np.degrees(np.arctan((values[1]-values[0])/(zone['end_m']-zone['start_m']))));close(zone['endpoint_sample_distance_m'],distances);COUNTS['slope_windows']+=1
        for half in [100,250,500]:
            w=p['widths'][str(half)]
            for i,(point,station) in enumerate(zip(points,stations)):
                pos=np.clip([station-half,station+half],0,length);local=np.c_[np.interp(pos,chain,xy[:,0]),np.interp(pos,chain,xy[:,1])];v=local[1]-local[0];normal=np.array([-v[1],v[0]])/np.linalg.norm(v)
                close(normal,w['normal_unit_vectors'][i]);width,censored=cell_traversal_width(ice,x,y,point,normal)
                close(width,np.nan if w['width_m'][i] is None else w['width_m'][i],atol=2e-6);assert censored==w['censored'][i]
                if np.isnan(width):assert w['reasons'][i]=='centre_outside_positive_ice'
                else:assert w['reasons'][i]==('normal_limit' if censored else 'valid')
                COUNTS['independent_widths']+=1
        for zone in s['width_zones']:
            widths=numeric(p['widths'][str(zone['tangent_half_window_m'])]['width_m']);censored=np.array(p['widths'][str(zone['tangent_half_window_m'])]['censored']);region=(d>=zone['start_m'])&(d<=zone['end_m']);ok=region&np.isfinite(widths)&~censored
            assert zone['n']==ok.sum() and zone['eligible_n']==region.sum();close(zone['coverage'],ok.sum()/region.sum());close(zone['median_width_m'],np.median(widths[ok]));COUNTS['width_windows']+=1
        COUNTS['systems']+=1
    prior={p['pair_id']:p for p in read(ROOT/'analysis/lia_51_same_pixel_pairs_v1/pairs.json')}
    for p in read(OUT/'pair_comparisons.json'):
        a,b=stats[p['a']],stats[p['b']];close(p['absolute_median_elevation_gap_m'],b['zmedian_m']-a['zmedian_m'])
        close(p['normalized_quantile_distance'],sum(abs(x-y) for x,y in zip(a['normalized_elevation_quantiles'],b['normalized_elevation_quantiles']))/1001)
        for width,value in p['absolute_histogram_overlap'].items():close(value,sum(min(x,y) for x,y in zip(a['hypsometry'][width]['area_fraction'],b['hypsometry'][width]['area_fraction'])))
        q=prior[p['pair_id']]
        for k in ['length','volume']:close(p['prior_'+k+'_late_contrast_pp'],q['metrics'][k+('_km' if k=='length' else '_km3')]['late_contrast_pp'])
        close(p['lower_half_area_fraction_a'],a['relative_area_fractions']['0.5']);close(p['lower_half_area_fraction_b'],b['relative_area_fractions']['0.5']);COUNTS['pairs']+=1
    parser=Links();parser.feed((OUT/'index.html').read_text(encoding='utf-8'))
    for link in parser.links:
        if '://' not in link and not link.startswith('#'):assert (OUT/link).exists(),link
    COUNTS['html_links']=len(parser.links);assert len(list(OUT.glob('*.png')))==5
    validation['independent_verification']=dict(status='pass',checked_utc=datetime.now(timezone.utc).isoformat(),counts=dict(COUNTS),method='Raster grid-boundary traversal independently verifies every polygon-intersection width; cumulative-coordinate interpolation, local raster nearest-cell checks, independently binned areas, explicit empirical quantiles and pair/historical links. Numerical/source verification, not observational or causal validation.')
    validation['status']='numerical-pass-visual-review-pending'
    if (OUT/'visual_review.json').exists():
        review=read(OUT/'visual_review.json');assert review['status']=='pass'
        for p,h in review['figure_hashes'].items():assert sha(OUT/p)==h,p
        validation['visual_review']=review;validation['status']='pass'
    save('validation.json',validation);save('output_manifest.json',{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='output_manifest.json' and '__pycache__' not in p.parts})
    print(json.dumps(validation['independent_verification'],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
