from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "run_round239_npl065_tmin_vs_tmean_failure_attribution_v1.py"
TARGET = HERE / "run_round241_npl065_same_version_cru_berkeley_attribution_v1.py"

text = SOURCE.read_text(encoding="utf-8")
start = text.index("def read_cru_tmean_djf(")
stop = text.index("\ndef main() -> int:", start)
reader = '''def read_cru_tmean_djf(
    manifest_path: Path,
) -> tuple[np.ndarray, dict[str, Any], list[Path]]:
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if manifest.get("forbidden_decades_not_requested") != ["1921.1930", "1931.1940"]:
        raise RuntimeError("round240-boundary-contract-changed")
    monthly: dict[tuple[int, int], float] = {}
    files: list[Path] = []
    checks: list[dict[str, Any]] = []
    grid_lat = None
    grid_lon = None
    for record in manifest["files"]:
        first_year, last_year = [int(value) for value in record["allowed_year_range"].split("-")]
        path = Path(record["netcdf_path"])
        if fingerprint(path)["sha256"] != record["netcdf_sha256"]:
            raise RuntimeError(f"round240-netcdf-hash-changed:{path.name}")
        files.append(path)
        with netcdf_file(path, "r", mmap=True) as dataset:
            latitudes = np.asarray(dataset.variables["lat"].data, dtype=float).copy()
            longitudes = np.asarray(dataset.variables["lon"].data, dtype=float).copy()
            iy = int(np.argmin(np.abs(latitudes - SITE_LAT)))
            ix = int(np.argmin(np.abs(longitudes - SITE_LON)))
            values = np.asarray(dataset.variables["tmp"].data[:, iy, ix], dtype=float).copy()
            fill = float(dataset.variables["tmp"]._attributes.get("_FillValue", -99999.0))
            values[np.isclose(values, fill) | (values <= -9000) | (values >= 1e20)] = np.nan
            grid_lat = float(latitudes[iy])
            grid_lon = float(longitudes[ix])
        periods = pd.period_range(f"{first_year}-01", periods=12 * (last_year - first_year + 1), freq="M")
        if len(periods) != len(values):
            raise RuntimeError(f"cru-tmean-month-count-changed:{path.name}")
        for period, value in zip(periods, values):
            key = (int(period.year), int(period.month))
            if key in monthly:
                raise RuntimeError(f"duplicate-cru-tmean-month:{key}")
            monthly[key] = float(value)
        checks.append({"file": path.name, "range": [first_year, last_year]})
    result = np.full(len(YEARS), np.nan)
    for index, year in enumerate(YEARS):
        keys = ((int(year) - 1, 12), (int(year), 1), (int(year), 2))
        winter = np.asarray([monthly.get(key, np.nan) for key in keys], dtype=float)
        if np.isfinite(winter).all():
            result[index] = float(np.mean(winter))
    if not np.isfinite(result).all():
        raise RuntimeError("incomplete-cru-tmean-target")
    return result, {
        "dataset": "CRU TS4.01 monthly mean temperature",
        "variable": "tmp",
        "source_decades": checks,
        "grid_latitude": grid_lat,
        "grid_longitude": grid_lon,
        "season": "previous_December_current_January_February",
    }, files
'''
text = text[:start] + reader + text[stop:]
replacements = [
    (
        'round239_npl065_tmin_vs_tmean_failure_attribution_v1_contract.md',
        'round241_npl065_same_version_cru_berkeley_attribution_v1_contract.md',
    ),
    (
        'round_239_npl065_tmin_vs_tmean_failure_attribution_v1',
        'round_241_npl065_same_version_cru_berkeley_attribution_v1',
    ),
    (
        'phyda = Path(r"D:\\Users\\JUNZ\\MATLAB\\Projects\\untitled\\PHYDA-v1-master")',
        'round240 = loop / "round_240_cru_ts401_tmean_same_version_acquisition_v1"',
    ),
    (
        '"cru_tmean": phyda / "input-recon/cruts.2004151855.v4.04/tmp/cru_ts4.04.1901.2019.tmp.dat.nc",',
        '"cru_tmean_manifest": round240 / "acquisition_manifest.json",',
    ),
    (
        'targets["CRU_TMEAN"], cru_tmean_check = read_cru_tmean_djf(paths["cru_tmean"])',
        'targets["CRU_TMEAN"], cru_tmean_check, cru_tmean_files = read_cru_tmean_djf(paths["cru_tmean_manifest"])\n    paths.update({f"cru_tmean_{index:02d}": path for index, path in enumerate(cru_tmean_files, start=1)})\n    before.update({name: fingerprint(path) for name, path in paths.items() if name not in before})',
    ),
    (
        'decision = "mean_temperature_product_representation_difference_requires_same_version_test"',
        'decision = "berkeley_local_representation_difference_confirmed_against_same_version_cru"',
    ),
    ('# Round 239 NPL065 Tmin versus Tmean attribution', '# Round 241 NPL065 same-version CRU versus Berkeley attribution'),
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
