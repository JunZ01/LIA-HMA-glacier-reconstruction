import geopandas as gpd
import pandas as pd
lia = gpd.read_file("/mnt/e/2Data/022_工作2/出图数据/Himalaya_LIA_with_RGIID_Complete/Himalaya_LIA_with_RGIID_Complete.shp")
rgi = gpd.read_file("/mnt/e/2Data/022_工作2/出图数据/himalaya_glaciers_complete/himalaya_glaciers_complete.shp")
# 筛选 RGI 面积 >= 10 km²
rgi_large = rgi[rgi['Area'] >= 10].copy()
large_rgi_ids = set(rgi_large['RGIId'])

# 找包含这些大冰川的 LIA
def contains_large_rgi(rgi_str):
    if pd.isna(rgi_str):
        return False
    ids = [x.strip() for x in str(rgi_str).split(';')]
    return bool(set(ids) & large_rgi_ids)

lia_matched = lia[lia['AllRGIs'].apply(contains_large_rgi)].copy()

print("LIA筛选完成: {} 条".format(len(lia_matched)))
# 筛选MainRGI对应的RGI冰川
main_rgi_ids = lia_matched['MainRGI'].tolist()
rgi_to_calibrate = rgi[rgi['RGIId'].isin(main_rgi_ids)].copy()

# 转WGS84
lia_matched = lia_matched.to_crs("EPSG:4326")
rgi_to_calibrate = rgi_to_calibrate.to_crs("EPSG:4326")

# 建立映射字典: LIA的UniqueID → MainRGI
lia_to_main_rgi = dict(zip(
    lia_matched['UniqueID'].astype(int),
    lia_matched['MainRGI']
))

print("RGI待校准: {} 条".format(len(rgi_to_calibrate)))
print("映射关系: {} 条".format(len(lia_to_main_rgi)))
from oggm import cfg, workflow, tasks, utils

WORKING_DIR = "/mnt/f/oggm_related/Rec_climate/Rec_rgi_calibration"
DEM_PATH = "/mnt/d/01 文件整理/04 数据/HMA_newregion/DEM_clip/Himalaya_3regions_DEM.tif"
MB_PATHS = [
    "/mnt/d/01 文件整理/04 数据/03 物质平衡数据/hugonnet time_series_13/dh_13_rgi60_pergla_rates.csv",
    "/mnt/d/01 文件整理/04 数据/03 物质平衡数据/hugonnet time_series_14/dh_14_rgi60_pergla_rates.csv",
    "/mnt/d/01 文件整理/04 数据/03 物质平衡数据/hugonnet time_series_15/dh_15_rgi60_pergla_rates.csv",
]
MB_PERIOD = "2000-01-01_2020-01-01"
# 读取MB数据
dfs = []
for path in MB_PATHS:
    print("读取: {}".format(path))
    df = pd.read_csv(path)
    dfs.append(df)
mb_all = pd.concat(dfs, ignore_index=True)
mb_data = mb_all[mb_all['period'] == MB_PERIOD].copy()
mb_data['dmdtda_kg'] = mb_data['dmdtda'] * 1000
mb_data = mb_data.set_index('rgiid')
print("MB数据: {} 条".format(len(mb_data)))
# 清理RGI数据
rgi = rgi_to_calibrate.copy()
rgi = rgi[rgi.geometry.notna() & ~rgi.is_empty].copy()
rgi['geometry'] = rgi.geometry.buffer(0)
rgi = rgi[rgi.is_valid].copy()
rgi = rgi[rgi['RGIId'].isin(mb_data.index)].copy()
print("有MB数据的RGI冰川: {}".format(len(rgi)))
from oggm import cfg, workflow, tasks, GlacierDirectory
cfg.initialize()

cfg.initialize(logging_level='WARNING')
cfg.PATHS['working_dir'] = WORKING_DIR
cfg.PATHS['dem_file'] = DEM_PATH
cfg.PARAMS['border'] = 360
cfg.PARAMS['use_intersects'] = False
cfg.PARAMS['use_rgi_area'] = False
cfg.PARAMS['continue_on_error'] = True
cfg.PARAMS['use_multiprocessing'] = True
cfg.PARAMS['mp_processes'] = 20

gdirs_rgi = workflow.init_glacier_directories(rgi, reset=False)
print("初始化完成: {}".format(len(gdirs_rgi)))
for gdir in gdirs_rgi:
    tasks.define_glacier_region(gdir, source='USER')
print("define_glacier_region完成: {}".format(len(gdirs_rgi)))
cfg.PARAMS['baseline_climate'] = 'CRU'
for gdir in gdirs_rgi:
    tasks.process_cru_data(gdir)
print("气候数据处理完成")
from oggm.workflow import execute_entity_task
cfg.PARAMS['use_multiprocessing'] = True

execute_entity_task(tasks.glacier_masks, gdirs_rgi)
execute_entity_task(tasks.compute_centerlines, gdirs_rgi)
execute_entity_task(tasks.initialize_flowlines, gdirs_rgi)
execute_entity_task(tasks.compute_downstream_line, gdirs_rgi)
execute_entity_task(tasks.compute_downstream_bedshape, gdirs_rgi)
execute_entity_task(tasks.catchment_area, gdirs_rgi)
execute_entity_task(tasks.catchment_intersections, gdirs_rgi)
execute_entity_task(tasks.catchment_width_geom, gdirs_rgi)
execute_entity_task(tasks.catchment_width_correction, gdirs_rgi)
print("预处理完成")
cfg.PARAMS['use_winter_prcp_fac'] = False
cfg.PARAMS['prcp_fac'] = 8
from oggm.core.massbalance import mb_calibration_from_scalar_mb
import os
import json
success = 0
fail = 0
for gdir in gdirs_rgi:
    try:
        ref_mb = float(mb_data.loc[gdir.rgi_id, 'dmdtda_kg'])
        mb_calibration_from_scalar_mb(
            gdir,
            ref_mb=ref_mb,
            ref_mb_years=(2000, 2020),
            overwrite_gdir=True,
            calibrate_param1='melt_f',
            calibrate_param2='temp_bias',
        )
        success += 1
    except Exception as e:
        fail += 1
        if fail <= 3:
            print("失败: {} - {}".format(gdir.rgi_id, str(e)[:100]))

print("\nCRU校准完成: 成功 {}, 失败 {}".format(success, fail))

# 验证几条
for gdir in gdirs_rgi[:5]:
    with open(os.path.join(gdir.dir, 'mb_calib.json'), 'r') as f:
        p = json.load(f)
    print("{}: melt_f={:.3f} temp_bias={:.3f} baseline={}".format(
        gdir.rgi_id, p['melt_f'], p['temp_bias'], p['baseline_climate_source']))
    lia_sorted = lia_matched.sort_values('UniqueID').copy()
    lia_cooked = utils.cook_rgidf(
        lia_sorted,
        o1_region='15',
        bgndate='16500101',
        ids=lia_sorted['UniqueID'].astype(int).tolist(),
        assign_column_values={'LIAArea': 'Area'}
    )

    lia_cooked['UniqueID'] = lia_sorted['UniqueID'].values
    lia_cooked['MainRGI'] = lia_sorted['MainRGI'].values

    print("cook完成: {} 条".format(len(lia_cooked)))
    print(lia_cooked[['RGIId', 'UniqueID', 'Area']].head())
    LIA_WORKING_DIR = "/mnt/f/oggm_related/Rec_climate/Rec_oggm_LIA_spinup"
    BARE_DEM_PATH = "/mnt/f/oggm_related/oggm_LIA_himalaya_great/output/Himalaya_bare_DEM.tif"

    cfg.PATHS['working_dir'] = LIA_WORKING_DIR
    cfg.PATHS['dem_file'] = BARE_DEM_PATH
    cfg.PARAMS['border'] = 360
    cfg.PARAMS['use_intersects'] = False
    cfg.PARAMS['use_rgi_area'] = False
    cfg.PARAMS['continue_on_error'] = True
    cfg.PARAMS['use_multiprocessing'] = True
    cfg.PARAMS['mp_processes'] = 20

    gdirs_lia = workflow.init_glacier_directories(lia_cooked, reset=False)
    print("LIA初始化完成: {} 条".format(len(gdirs_lia)))
    # 1. 重建lia_mb_params（用新的CRU校准参数）
    import json, os, numpy as np

    rgi_params_cru = {}
    for gdir in gdirs_rgi:
        with open(os.path.join(gdir.dir, 'mb_calib.json'), 'r') as f:
            rgi_params_cru[gdir.rgi_id] = json.load(f)

    lia_mb_params_cru = {}
    missing = []
    for idx, row in lia_matched.iterrows():
        unique_id = int(row['UniqueID'])
        main_rgi = row['MainRGI']
        if main_rgi in rgi_params_cru:
            p = rgi_params_cru[main_rgi]
            lia_mb_params_cru[unique_id] = {
                'melt_f': p['melt_f'],
                'prcp_fac': p['prcp_fac'],
                'temp_bias': p['temp_bias'],
            }
        else:
            missing.append((unique_id, main_rgi))

    # 缺失的用区域平均
    avg_melt_f = np.mean([p['melt_f'] for p in rgi_params_cru.values()])
    avg_temp_bias = np.mean([p['temp_bias'] for p in rgi_params_cru.values()])
    for uid, rgi in missing:
        lia_mb_params_cru[uid] = {'melt_f': avg_melt_f, 'prcp_fac': 8.0, 'temp_bias': avg_temp_bias}

    print("LIA参数映射: {} 条, 缺失补充: {} 条".format(len(lia_mb_params_cru), len(missing)))
    print("melt_f范围: {:.2f} ~ {:.2f}".format(
        min(p['melt_f'] for p in lia_mb_params_cru.values()),
        max(p['melt_f'] for p in lia_mb_params_cru.values())))

    # 2. 写入LIA冰川的mb_calib.json
    for gdir in gdirs_lia:
        unique_id = int(gdir.rgi_id.split('.')[-1])
        params = lia_mb_params_cru[unique_id]

        mb_calib = {
            'rgi_id': gdir.rgi_id,
            'bias': 0,
            'melt_f': params['melt_f'],
            'prcp_fac': params['prcp_fac'],
            'temp_bias': params['temp_bias'],
            'reference_mb': 0,
            'reference_mb_err': None,
            'reference_period': '2000-01-01_2020-01-01',
            'mb_global_params': {
                'temp_default_gradient': -0.0065,
                'temp_all_solid': 0.0,
                'temp_all_liq': 2.0,
                'temp_melt': -1.0,
            },
            'baseline_climate_source': 'CRU TS4.04',
        }

        fpath = os.path.join(gdir.dir, 'mb_calib.json')
        with open(fpath, 'w') as f:
            json.dump(mb_calib, f, indent=2)

    print("LIA冰川mb_calib写入完成: {} 条".format(len(gdirs_lia)))

    # 验证IK
    with open(os.path.join(gdirs_lia[0].dir, 'mb_calib.json'), 'r') as f:
        p = json.load(f)
    print("样本: melt_f={:.3f} baseline={}".format(p['melt_f'], p['baseline_climate_source']))