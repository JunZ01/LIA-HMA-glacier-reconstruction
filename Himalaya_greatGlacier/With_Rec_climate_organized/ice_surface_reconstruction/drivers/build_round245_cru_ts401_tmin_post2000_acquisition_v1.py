from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "acquire_round240_cru_ts401_tmean_same_version_v1.py"
TARGET = HERE / "acquire_round245_cru_ts401_tmin_post2000_v1.py"

text = SOURCE.read_text(encoding="utf-8")
replacements = [
    (
        'round240_cru_ts401_tmean_same_version_acquisition_v1_contract.md',
        'round245_cru_ts401_tmin_post2000_acquisition_v1_contract.md',
    ),
    (
        'round_240_cru_ts401_tmean_same_version_acquisition_v1',
        'round_245_cru_ts401_tmin_post2000_acquisition_v1',
    ),
    ('cruts.1709081022.v4.01/tmp/', 'cruts.1709081022.v4.01/tmn/'),
    (
        'DECADES = ("1951.1960", "1961.1970", "1971.1980", "1981.1990", "1991.2000")',
        'DECADES = ("2001.2010", "2011.2016")',
    ),
    ('cru_ts4.01.{decade}.tmp.dat.nc.gz', 'cru_ts4.01.{decade}.tmn.dat.nc.gz'),
    ('CRU TS4.01 monthly mean temperature', 'CRU TS4.01 monthly minimum temperature'),
    ('# Round 240 CRU TS4.01 Tmean acquisition', '# Round 245 CRU TS4.01 post-2000 Tmin acquisition'),
]
for old, new in replacements:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"replacement-count:{count}:{old}")
    text = text.replace(old, new)
if TARGET.exists():
    raise FileExistsError(TARGET)
TARGET.write_text(text, encoding="utf-8")
print(TARGET)
