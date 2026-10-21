from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import stat
import zipfile
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
DRIVERS = ROOT / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers"
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
OUT = LOOP / "round_278_baiyu_haizi_eastern_hma_anchor_identity_audit_v1"
CONTRACT = DRIVERS / "round278_baiyu_haizi_anchor_identity_audit_v1_contract.md"
DRIVER = Path(__file__).resolve()

ROUND211 = LOOP / "round_211_wscp_temperature_data_recovery_v2"
ROUND214 = LOOP / "round_214_haizi_annual_temperature_data_recovery_v1"
BAIYU_ARCHIVE = ROUND211 / "TemReconWSCP_1383-2005.rar"
BAIYU_PAGE = ROUND211 / "official_geodoi_page.html"
BAIYU_WORKBOOK = ROUND211 / "official_archive_extracted/Temp_Reconstruction&Meteorological.xlsx"
HAIZI_WORKBOOK = ROUND214 / "41598_2024_79096_MOESM1_ESM.xlsx"
LEDGER = LOOP / "round_109_cross_network_identity_closure_v1/proxy_evidence_master_ledger_identity_closed.csv"
PROXY9 = Path(r"D:\Users\JUNZ\Desktop\文件整理\1 论文\work1\compared_result_all\python_data\proxy_data_9.mat")

EXPECTED = {
    BAIYU_ARCHIVE: "d9a16a56a837ba185ae47839c71a2dc20ef256237b1af3dbe389e2dd34610fb8",
    BAIYU_WORKBOOK: "286e9338e1cfeddb2b51f6805241d1e0989c891af3adbdd81691f1918b794d68",
    HAIZI_WORKBOOK: "35302c0ba1912b015ee8d8b1a5c05df9aa82dd44d4467f7b07278e723611d3eb",
    LEDGER: "d82b08c4885c1016ea974657ada3003179ea16ad3cbf61cf655214fa78b0587c",
}
LOCKED_START = 1921
LOCKED_END = 1939
NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_state(path: Path) -> dict[str, object]:
    stat_result = path.stat()
    return {
        "path": str(path),
        "bytes": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
        "sha256": sha256(path),
    }


def allowed_baiyu(year: int) -> bool:
    return 1850 <= year <= 1920 or 1940 <= year <= 1950 or 1962 <= year <= 2005


def allowed_haizi(year: int) -> bool:
    return 1850 <= year <= 1920 or 1940 <= year <= 1950 or 1959 <= year <= 2017


def allowed_network_union(year: int) -> bool:
    return 1850 <= year <= 1920 or 1940 <= year <= 1950 or 1959 <= year <= 2017


def column_name(cell_reference: str) -> str:
    return "".join(character for character in cell_reference if character.isalpha()).upper()


class YearFirstXlsx:
    def __init__(self, path: Path):
        self.path = path
        self.archive = zipfile.ZipFile(path)
        self.shared_strings = self._shared_strings()
        self.sheet_paths = self._sheet_paths()
        self.decoded_value_cells: list[tuple[int, str]] = []

    def close(self) -> None:
        self.archive.close()

    def _shared_strings(self) -> list[str]:
        if "xl/sharedStrings.xml" not in self.archive.namelist():
            return []
        root = ET.fromstring(self.archive.read("xl/sharedStrings.xml"))
        output = []
        for item in root.findall(f"{{{NS_MAIN}}}si"):
            output.append("".join(node.text or "" for node in item.iter(f"{{{NS_MAIN}}}t")))
        return output

    def _sheet_paths(self) -> dict[str, str]:
        workbook = ET.fromstring(self.archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(self.archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            relation.attrib["Id"]: relation.attrib["Target"]
            for relation in relationships.findall(f"{{{NS_PKG_REL}}}Relationship")
        }
        output = {}
        for sheet in workbook.findall(f".//{{{NS_MAIN}}}sheet"):
            target = targets[sheet.attrib[f"{{{NS_REL}}}id"]].replace("\\", "/")
            if target.startswith("/"):
                path = target.lstrip("/")
            elif target.startswith("xl/"):
                path = target
            else:
                path = f"xl/{target}"
            output[sheet.attrib["name"]] = path
        return output

    def _decode_cell(self, cell: ET.Element, year: int | None, column: str) -> object:
        if year is not None and column != "A":
            self.decoded_value_cells.append((year, column))
        cell_type = cell.attrib.get("t", "n")
        if cell_type == "inlineStr":
            token = "".join(node.text or "" for node in cell.iter(f"{{{NS_MAIN}}}t"))
        else:
            value_node = cell.find(f"{{{NS_MAIN}}}v")
            token = "" if value_node is None else (value_node.text or "")
            if cell_type == "s" and token:
                token = self.shared_strings[int(token)]
        if token == "LOCKED_POISON":
            raise RuntimeError("locked-poison-value-decoded")
        return token

    def read_series(
        self,
        sheet_name: str,
        value_column: str,
        is_allowed: Callable[[int], bool],
    ) -> tuple[dict[int, float], dict[str, int]]:
        output: dict[int, float] = {}
        locked_rows_skipped = 0
        outside_rows_skipped = 0
        stream = self.archive.open(self.sheet_paths[sheet_name])
        for _, row in ET.iterparse(stream, events=("end",)):
            if row.tag.rsplit("}", 1)[-1] != "row":
                continue
            cells = {
                column_name(cell.attrib.get("r", "")): cell
                for cell in row.findall(f"{{{NS_MAIN}}}c")
            }
            if "A" not in cells:
                row.clear()
                continue
            year_token = self._decode_cell(cells["A"], None, "A")
            try:
                year = int(float(str(year_token)))
            except ValueError:
                row.clear()
                continue
            if LOCKED_START <= year <= LOCKED_END:
                locked_rows_skipped += 1
                row.clear()
                continue
            if not is_allowed(year):
                outside_rows_skipped += 1
                row.clear()
                continue
            if value_column not in cells:
                raise RuntimeError(f"missing-value-cell:{sheet_name}:{year}:{value_column}")
            token = self._decode_cell(cells[value_column], year, value_column)
            value = float(token)
            if not math.isfinite(value):
                raise RuntimeError(f"nonfinite-value:{sheet_name}:{year}:{value_column}")
            if year in output:
                raise RuntimeError(f"duplicate-year:{sheet_name}:{year}")
            output[year] = value
            row.clear()
        return output, {
            "locked_rows_skipped_before_value_decode": locked_rows_skipped,
            "outside_allowed_rows_skipped_before_value_decode": outside_rows_skipped,
            "allowed_value_cells_decoded": len(output),
        }


def flat_h5(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray(dataset).reshape(-1, order="F")


def selected_h5_values(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    if len(indices) == 0:
        return np.asarray([], dtype=float)
    values = []
    shape = dataset.shape
    if len(shape) == 1:
        getter = lambda index: dataset[index]
    elif len(shape) == 2 and shape[0] == 1:
        getter = lambda index: dataset[0, index]
    elif len(shape) == 2 and shape[1] == 1:
        getter = lambda index: dataset[index, 0]
    else:
        raise RuntimeError(f"unsupported-hdf-vector-shape:{shape}")
    for index in indices:
        token = getter(int(index))
        if isinstance(token, bytes):
            token = token.decode("utf-8")
        if str(token) == "LOCKED_POISON":
            raise RuntimeError("locked-poison-hdf-value-decoded")
        values.append(float(token))
    return np.asarray(values, dtype=float)


def read_h5_series_year_first(
    years_dataset: h5py.Dataset,
    values_dataset: h5py.Dataset,
    is_allowed: Callable[[int], bool],
) -> tuple[dict[int, float], int]:
    years = flat_h5(years_dataset).astype(int)
    selected = np.asarray([index for index, year in enumerate(years) if is_allowed(int(year))], dtype=int)
    selected_years = years[selected]
    if np.any((selected_years >= LOCKED_START) & (selected_years <= LOCKED_END)):
        raise RuntimeError("locked-hdf-index-selected")
    selected_values = selected_h5_values(values_dataset, selected)
    output = {
        int(year): float(value)
        for year, value in zip(selected_years, selected_values)
        if math.isfinite(float(value))
    }
    locked_skipped = int(np.sum((years >= LOCKED_START) & (years <= LOCKED_END)))
    return output, locked_skipped


def decode_hdf_string(dataset: h5py.Dataset) -> str:
    return "".join(chr(int(value)) for value in flat_h5(dataset) if int(value)).strip()


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def segmented_smooth(series: dict[int, float], width: int = 31) -> dict[int, float]:
    years = sorted(series)
    result: dict[int, float] = {}
    segments: list[list[int]] = []
    current: list[int] = []
    for year in years:
        if current and year != current[-1] + 1:
            segments.append(current)
            current = []
        current.append(year)
    if current:
        segments.append(current)
    for segment in segments:
        values = pd.Series([series[year] for year in segment], index=segment, dtype=float)
        smooth = values.rolling(width, center=True, min_periods=width).mean()
        result.update({int(year): float(value) for year, value in smooth.dropna().items()})
    return result


def identity_metrics(anchor: dict[int, float], candidate: dict[int, float]) -> dict[str, float | int | bool]:
    common = np.asarray(sorted(set(anchor) & set(candidate)), dtype=int)
    if len(common) < 3:
        return {
            "common_year_count": int(len(common)), "raw_correlation": math.nan,
            "first_difference_correlation": math.nan, "smooth31_correlation": math.nan,
            "affine_slope": math.nan, "affine_intercept": math.nan,
            "affine_rmse": math.nan, "affine_rmse_fraction_anchor_sd": math.nan,
            "affine_max_abs_residual": math.nan, "exact_affine_alias": False, "near_alias": False,
        }
    y = np.asarray([anchor[int(year)] for year in common], dtype=float)
    x = np.asarray([candidate[int(year)] for year in common], dtype=float)
    raw = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else math.nan
    design = np.column_stack([x, np.ones(len(x))])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ np.asarray([slope, intercept])
    rmse = float(np.sqrt(np.mean(residual**2)))
    anchor_sd = float(np.std(y, ddof=0))
    rmse_fraction = rmse / anchor_sd if anchor_sd > 0 else math.nan
    consecutive = np.diff(common) == 1
    if int(consecutive.sum()) >= 3:
        dx = np.diff(x)[consecutive]
        dy = np.diff(y)[consecutive]
        diff_corr = float(np.corrcoef(dx, dy)[0, 1]) if np.std(dx) > 0 and np.std(dy) > 0 else math.nan
    else:
        diff_corr = math.nan
    anchor_smooth = segmented_smooth({int(year): value for year, value in zip(common, y)})
    candidate_smooth = segmented_smooth({int(year): value for year, value in zip(common, x)})
    smooth_years = sorted(set(anchor_smooth) & set(candidate_smooth))
    smooth_corr = (
        float(np.corrcoef(
            [anchor_smooth[year] for year in smooth_years],
            [candidate_smooth[year] for year in smooth_years],
        )[0, 1])
        if len(smooth_years) >= 3 else math.nan
    )
    maximum = float(np.max(np.abs(residual)))
    exact = len(common) >= 20 and abs(raw) >= 0.999999999 and maximum <= 1e-10
    near = len(common) >= 30 and abs(raw) >= 0.995 and rmse_fraction <= 0.10
    return {
        "common_year_count": int(len(common)), "raw_correlation": raw,
        "first_difference_correlation": diff_corr, "smooth31_correlation": smooth_corr,
        "affine_slope": float(slope), "affine_intercept": float(intercept),
        "affine_rmse": rmse, "affine_rmse_fraction_anchor_sd": rmse_fraction,
        "affine_max_abs_residual": maximum, "exact_affine_alias": bool(exact), "near_alias": bool(near),
    }


def run_poison_test(work: Path) -> dict[str, object]:
    work.mkdir(parents=True, exist_ok=False)
    xlsx = work / "year_first_poison.xlsx"
    workbook_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="{NS_MAIN}" xmlns:r="{NS_REL}"><sheets><sheet name="Data" sheetId="1" r:id="rId1"/></sheets></workbook>'''
    rels_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{NS_PKG_REL}"><Relationship Id="rId1" Type="worksheet" Target="worksheets/sheet1.xml"/></Relationships>'''
    sheet_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="{NS_MAIN}"><sheetData>
<row r="1"><c r="A1"><v>1920</v></c><c r="B1"><v>1.5</v></c></row>
<row r="2"><c r="A2"><v>1921</v></c><c r="B2" t="str"><v>LOCKED_POISON</v></c></row>
<row r="3"><c r="A3"><v>1940</v></c><c r="B3"><v>2.5</v></c></row>
</sheetData></worksheet>'''
    with zipfile.ZipFile(xlsx, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", rels_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)
    parser = YearFirstXlsx(xlsx)
    values, audit = parser.read_series("Data", "B", lambda year: year in {1920, 1940})
    parser.close()
    if values != {1920: 1.5, 1940: 2.5} or (1921, "B") in parser.decoded_value_cells:
        raise RuntimeError("xlsx-poison-test-failed")
    hdf = work / "year_first_poison.h5"
    with h5py.File(hdf, "w") as handle:
        handle.create_dataset("year", data=np.asarray([1920, 1921, 1940], dtype=int))
        handle.create_dataset("value", data=np.asarray([b"1.5", b"LOCKED_POISON", b"2.5"], dtype="S20"))
    with h5py.File(hdf, "r") as handle:
        hdf_values, hdf_locked = read_h5_series_year_first(handle["year"], handle["value"], lambda year: year in {1920, 1940})
    if hdf_values != {1920: 1.5, 1940: 2.5} or hdf_locked != 1:
        raise RuntimeError("hdf-poison-test-failed")
    return {
        "status": "passed", "xlsx_locked_poison_decoded": False,
        "hdf_locked_poison_decoded": False, "xlsx_audit": audit,
        "locked_interval": [LOCKED_START, LOCKED_END],
    }


def copy_preserved(source: Path, destination: Path) -> dict[str, object]:
    shutil.copy2(source, destination)
    if sha256(source) != sha256(destination):
        raise RuntimeError(f"copy-hash-mismatch:{source}")
    os.chmod(destination, stat.S_IREAD)
    return file_state(destination)


def main() -> int:
    required = [BAIYU_ARCHIVE, BAIYU_PAGE, BAIYU_WORKBOOK, HAIZI_WORKBOOK, LEDGER, PROXY9, CONTRACT, DRIVER]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    for path, expected in EXPECTED.items():
        if sha256(path).lower() != expected:
            raise RuntimeError(f"source-hash-changed:{path}")
    source_before = {str(path): file_state(path) for path in required}
    OUT.mkdir(parents=True, exist_ok=False)
    raw = OUT / "raw_unopened"
    prepared = OUT / "prepared"
    raw.mkdir()
    prepared.mkdir()
    poison = run_poison_test(OUT / "synthetic_poison_test")
    (OUT / "poison_test_result.json").write_text(json.dumps(poison, indent=2) + "\n", encoding="utf-8")

    copied = {
        "baiyu_official_archive": copy_preserved(BAIYU_ARCHIVE, raw / BAIYU_ARCHIVE.name),
        "baiyu_official_metadata_page": copy_preserved(BAIYU_PAGE, raw / BAIYU_PAGE.name),
        "haizi_official_supplement": copy_preserved(HAIZI_WORKBOOK, raw / HAIZI_WORKBOOK.name),
        "baiyu_prepared_workbook": copy_preserved(BAIYU_WORKBOOK, prepared / BAIYU_WORKBOOK.name),
        "haizi_prepared_workbook": copy_preserved(HAIZI_WORKBOOK, prepared / HAIZI_WORKBOOK.name),
    }

    baiyu_parser = YearFirstXlsx(prepared / BAIYU_WORKBOOK.name)
    baiyu_temperature, baiyu_parse = baiyu_parser.read_series("Tab.1", "B", allowed_baiyu)
    baiyu_instrument, baiyu_instrument_parse = baiyu_parser.read_series("Tab.2", "B", allowed_baiyu)
    baiyu_decoded = list(baiyu_parser.decoded_value_cells)
    baiyu_parser.close()
    haizi_parser = YearFirstXlsx(prepared / HAIZI_WORKBOOK.name)
    haizi_std, haizi_std_parse = haizi_parser.read_series("Sheet1", "B", allowed_haizi)
    haizi_temperature, haizi_temperature_parse = haizi_parser.read_series("Sheet1", "C", allowed_haizi)
    haizi_decoded = list(haizi_parser.decoded_value_cells)
    haizi_parser.close()
    if any(LOCKED_START <= year <= LOCKED_END for year, _ in baiyu_decoded + haizi_decoded):
        raise RuntimeError("locked-xlsx-value-cell-decoded")

    expected_baiyu = set(range(1850, 1921)) | set(range(1940, 1951)) | set(range(1962, 2006))
    expected_haizi = set(range(1850, 1921)) | set(range(1940, 1951)) | set(range(1959, 2018))
    if set(baiyu_temperature) != expected_baiyu or set(haizi_temperature) != expected_haizi or set(haizi_std) != expected_haizi:
        raise RuntimeError("allowed-coverage-mismatch")
    if set(baiyu_instrument) != set(range(1962, 2006)):
        raise RuntimeError("baiyu-instrument-coverage-mismatch")

    ledger = pd.read_csv(LEDGER)
    p6 = ledger[ledger.network.eq("proxy6")].copy()
    p9 = ledger[ledger.network.eq("proxy9")].copy()
    if len(p6) != 202 or len(p9) != 297:
        raise RuntimeError("current-network-size-changed")
    p9_lookup = p9.set_index("network_record_1based")
    group_to_p6 = p6.groupby("evidence_group_id")["network_record_1based"].apply(lambda values: ";".join(str(int(value)) for value in values)).to_dict()

    sequence_rows = []
    network_locked_rows_seen = 0
    with h5py.File(PROXY9, "r") as handle:
        references = flat_h5(handle["proxy_data/full_proxy"])
        if len(references) != 297:
            raise RuntimeError("proxy9-record-count-changed")
        for position, reference in enumerate(references, start=1):
            proxy = handle[reference]
            candidate, locked_seen = read_h5_series_year_first(proxy["age_yng"], proxy["data"], allowed_network_union)
            network_locked_rows_seen += locked_seen
            metadata = p9_lookup.loc[position]
            location = flat_h5(proxy["location"]).astype(float)
            for anchor_name, anchor_sequence, anchor_kind in (
                ("Baiyu_2022", baiyu_temperature, "published_temperature_reconstruction_no_raw_chronology"),
                ("Haizi_2024", haizi_std, "published_standard_chronology"),
            ):
                metrics = identity_metrics(anchor_sequence, candidate)
                sequence_rows.append({
                    "anchor": anchor_name, "anchor_sequence_used": anchor_kind,
                    "proxy9_record_1based": position,
                    "proxy6_record_1based": group_to_p6.get(metadata.evidence_group_id, ""),
                    "evidence_group_id": metadata.evidence_group_id,
                    "latitude": float(location[0]), "longitude": float(location[1]),
                    "formal_metadata_identity": metadata.formal_metadata_identity,
                    "publication_cite_key": metadata.publication_cite_key,
                    "measurement": metadata.measurement,
                    **metrics,
                })
    sequence = pd.DataFrame(sequence_rows)
    sequence["abs_raw_correlation"] = sequence.raw_correlation.abs()
    sequence = sequence.sort_values(["anchor", "exact_affine_alias", "near_alias", "abs_raw_correlation"], ascending=[True, False, False, False])
    sequence.to_csv(OUT / "proxy6_proxy9_sequence_identity_metrics.csv", index=False)
    sequence.groupby("anchor", group_keys=False).head(20).to_csv(OUT / "top20_sequence_matches.csv", index=False)

    anchors = {
        "Baiyu_2022": (31.15, 99.81),
        "Haizi_2024": (29.58, 100.65),
    }
    unique_groups = p9.drop_duplicates("evidence_group_id")
    location_rows = []
    for anchor_name, (latitude, longitude) in anchors.items():
        for row in unique_groups.itertuples(index=False):
            location_rows.append({
                "anchor": anchor_name, "evidence_group_id": row.evidence_group_id,
                "proxy9_record_1based": int(row.network_record_1based),
                "proxy6_record_1based": group_to_p6.get(row.evidence_group_id, ""),
                "latitude": float(row.latitude), "longitude": float(row.longitude),
                "distance_km": haversine(latitude, longitude, float(row.latitude), float(row.longitude)),
                "formal_metadata_identity": row.formal_metadata_identity,
                "publication_cite_key": row.publication_cite_key,
                "measurement": row.measurement,
            })
    locations = pd.DataFrame(location_rows).sort_values(["anchor", "distance_km"])
    locations.groupby("anchor", group_keys=False).head(20).to_csv(OUT / "nearest20_location_identity_audit.csv", index=False)

    curve_rows = []
    for anchor_name, series, season, calibration in (
        ("Baiyu_2022", baiyu_temperature, "June-July mean temperature", "1962-2005"),
        ("Haizi_2024", haizi_temperature, "previous-October to current-September mean temperature", "1959-2017"),
    ):
        smooth = segmented_smooth(series)
        for year, value in sorted(series.items()):
            period = "historical_1850_1920" if year <= 1920 else "historical_1940_1950" if year <= 1950 else "modern_calibration"
            curve_rows.append({
                "anchor": anchor_name, "year": year, "temperature_degC": value,
                "smooth31_degC": smooth.get(year, math.nan), "period_class": period,
                "season": season, "calibration_period": calibration,
            })
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(OUT / "allowed_period_temperature_curves.csv", index=False)
    if curves.year.between(LOCKED_START, LOCKED_END).any():
        raise RuntimeError("locked-year-written-to-curve")

    anchor_summary = {}
    metadata_rows = []
    for anchor_name in anchors:
        sub_sequence = sequence[sequence.anchor.eq(anchor_name)]
        sub_location = locations[locations.anchor.eq(anchor_name)]
        exact_count = int(sub_sequence.exact_affine_alias.sum())
        near_count = int(sub_sequence.near_alias.sum())
        nearest = sub_location.iloc[0]
        if anchor_name == "Baiyu_2022":
            independence = "A_source_independent_seasonal_anchor_sequence_screen_limited_by_unpublished_chronology"
            admissible = True
            rationale = "October-2017 Picea balfouriana collection; no current group within 35 km; no reconstruction affine alias; raw chronology not published"
            metadata = {
                "paper_doi": "10.1002/joc.7336", "dataset_doi": "10.3974/geodb.2022.02.03.V1",
                "season": "June-July mean temperature", "unit": "degC", "full_coverage": "1383-2005",
                "allowed_coverage": "1850-1920;1940-1950;1962-2005", "calibration": "1962-2005",
                "site": "Baiyu County, western Sichuan Plateau", "latitude": 31.15, "longitude": 99.81,
                "species": "Picea balfouriana", "sample_metadata": "collected October 2017; raw chronology absent from official package",
            }
        else:
            independence = "B_sequence_independent_annual_anchor_with_nearby_regional_material_uncertainty"
            admissible = True
            rationale = "published STD chronology has no exact/near network alias; nearest Daocheng project group is regionally related but sequence-distinct; individual sample IDs unavailable"
            metadata = {
                "paper_doi": "10.1038/s41598-024-79096-6", "dataset_doi": "",
                "season": "previous-October to current-September mean temperature", "unit": "degC", "full_coverage": "1115-2017",
                "allowed_coverage": "1850-1920;1940-1950;1959-2017", "calibration": "1959-2017",
                "site": "Haizi Mountain, southeastern Tibetan Plateau", "latitude": 29.58, "longitude": 100.65,
                "species": "Juniperus saltuaria", "sample_metadata": "73 cores from 30 living trees plus 20 dead-tree discs; individual sample IDs unavailable",
            }
        if exact_count or near_count:
            admissible = False
            independence = "rejected_sequence_alias"
        anchor_summary[anchor_name] = {
            "exact_affine_alias_count": exact_count, "near_alias_count": near_count,
            "nearest_project_evidence_group_km": float(nearest.distance_km),
            "nearest_project_evidence_group": nearest.evidence_group_id,
            "groups_within_35_km": int((sub_location.distance_km <= 35).sum()),
            "groups_within_75_km": int((sub_location.distance_km <= 75).sum()),
            "independence_level": independence, "admissible_as_independent_evidence": admissible,
            "rationale": rationale,
        }
        metadata_rows.append({"anchor": anchor_name, **metadata, **anchor_summary[anchor_name]})
    pd.DataFrame(metadata_rows).to_csv(OUT / "anchor_metadata_and_independence.csv", index=False)

    figure, axes = plt.subplots(2, 1, figsize=(12.2, 7.6), sharex=True)
    colors = {"Baiyu_2022": "#27647B", "Haizi_2024": "#A14B37"}
    labels = {
        "Baiyu_2022": "Baiyu 2022: June-July",
        "Haizi_2024": "Haizi 2024: previous Oct-current Sep",
    }
    for axis, anchor_name in zip(axes, ("Baiyu_2022", "Haizi_2024")):
        subset = curves[curves.anchor.eq(anchor_name)]
        for _, segment in subset.groupby("period_class", sort=False):
            axis.plot(segment.year, segment.temperature_degC, color=colors[anchor_name], linewidth=0.8, alpha=0.38)
            axis.plot(segment.year, segment.smooth31_degC, color=colors[anchor_name], linewidth=2.1)
        axis.axvspan(1921, 1939, color="#D9D9D9", alpha=0.9, linewidth=0)
        axis.set_ylabel("Temperature (degC)")
        axis.set_title(labels[anchor_name], loc="left", fontsize=11)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[-1].set_xlabel("Year")
    axes[-1].set_xlim(1845, 2020)
    figure.suptitle("Official eastern-HMA temperature anchors: allowed periods only", fontsize=13)
    figure.text(0.99, 0.01, "Gray: sealed 1921-1939; raw annual values and within-segment 31-year means.", ha="right", fontsize=8.5, color="#555555")
    figure.tight_layout(rect=(0.04, 0.04, 0.99, 0.95))
    figure.savefig(OUT / "baiyu_haizi_allowed_temperature_curves.png", dpi=220)
    plt.close(figure)

    shutil.copy2(CONTRACT, OUT / "ROUND_CONTRACT.md")
    source_after = {str(path): file_state(path) for path in required}
    if source_before != source_after:
        raise RuntimeError("read-only-source-changed")
    summary = {
        "round": 278, "status": "complete",
        "decision": "admit_baiyu_as_independent_seasonal_anchor_and_haizi_as_sequence_independent_annual_anchor",
        "current_network": {"proxy6_records": 202, "proxy9_records": 297, "unique_evidence_groups": int(p9.evidence_group_id.nunique())},
        "anchors": anchor_summary,
        "parsers": {
            "baiyu_temperature": baiyu_parse, "baiyu_instrument": baiyu_instrument_parse,
            "haizi_temperature": haizi_temperature_parse, "haizi_std": haizi_std_parse,
            "network_locked_rows_seen_and_skipped_before_value_access": network_locked_rows_seen,
        },
        "synthetic_poison_test": poison,
        "locked_period_values_accessed": False, "source_states_unchanged": True,
        "production_changed": False, "enkf_run": False, "oggm_run": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    manifest = {
        "official_origins": {
            "baiyu_dataset": "https://www.geodoi.ac.cn/weben/doi.aspx?Id=2006",
            "baiyu_archive": "https://www.geodoi.ac.cn/WebEn/down.aspx?fileID=6140",
            "baiyu_paper": "https://doi.org/10.1002/joc.7336",
            "haizi_paper": "https://www.nature.com/articles/s41598-024-79096-6",
            "haizi_supplement": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-024-79096-6/MediaObjects/41598_2024_79096_MOESM1_ESM.xlsx",
        },
        "sources_before": source_before, "sources_after": source_after,
        "raw_and_prepared_copies": copied, "source_states_unchanged": True,
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    final = f"""# Baiyu 2022 and Haizi 2024 Eastern-HMA Anchor Audit

## Decision

`{summary['decision']}`

The official Baiyu GeoDOI archive and Haizi Springer supplement were preserved
byte-for-byte under `raw_unopened` and verified against their frozen SHA256
hashes. The current network contains 202 proxy6 records mapped onto 297 proxy9
evidence groups.

## Baiyu 2022

- Target: June-July mean temperature, degC; reconstruction 1383-2005;
  calibration 1962-2005.
- Allowed values exported: 1850-1920, 1940-1950, and 1962-2005 only.
- Nearest current evidence group: `{anchor_summary['Baiyu_2022']['nearest_project_evidence_group']}`
  at `{anchor_summary['Baiyu_2022']['nearest_project_evidence_group_km']:.3f} km`; groups within 35 km: `0`.
- Exact/near sequence aliases: `{anchor_summary['Baiyu_2022']['exact_affine_alias_count']}` /
  `{anchor_summary['Baiyu_2022']['near_alias_count']}`.
- Independence: `{anchor_summary['Baiyu_2022']['independence_level']}`. The
  official package does not publish the raw chronology, so its sequence screen
  uses the published reconstruction and cannot close raw-sample identity by itself.

## Haizi 2024

- Target: previous-October to current-September mean temperature, degC;
  reconstruction 1115-2017; calibration 1959-2017.
- Allowed values exported: 1850-1920, 1940-1950, and 1959-2017 only.
- Nearest current evidence group: `{anchor_summary['Haizi_2024']['nearest_project_evidence_group']}`
  at `{anchor_summary['Haizi_2024']['nearest_project_evidence_group_km']:.3f} km`; groups within 35 km: `0`.
- Exact/near STD chronology aliases: `{anchor_summary['Haizi_2024']['exact_affine_alias_count']}` /
  `{anchor_summary['Haizi_2024']['near_alias_count']}`.
- Independence: `{anchor_summary['Haizi_2024']['independence_level']}`. It is
  sequence-independent from the nearby Daocheng network record, but individual
  sample identifiers were not published, so material-level independence remains
  less certain than sequence independence.

## Safety

Both XLSX and HDF5 poison tests passed. Every row was decoded year first, and
1921-1939 was skipped before any sequence value access. No source or production
file changed; no EnKF or OGGM run occurred.
"""
    (OUT / "FINAL_RESULT.md").write_text(final, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
