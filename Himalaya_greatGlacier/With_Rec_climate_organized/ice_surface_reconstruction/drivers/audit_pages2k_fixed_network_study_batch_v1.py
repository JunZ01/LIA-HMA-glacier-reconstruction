from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat


EXPERIMENT_ID = "climate_reconstruction_pages2k_fixed_network_study_batch_v1"
PAGES_TS = Path(
    r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\proxy-data\PAGES2k_v2"
    r"\proxy-data\pages2k_proxy\Pages2kTSv1_13_1.mat"
)
STUDY_FIELDS = {
    "data_author": "dataPub1_author",
    "data_cite_key": "dataPub1_citeKey",
    "publication_author": "pub1_author",
    "publication_cite_key": "pub1_citeKey",
    "publication_year": "pub1_pubYear",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_state(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest(path),
    }


def clean(value: object) -> str:
    if value is None:
        return "<missing>"
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, bytes):
        text = value.decode("latin-1").strip()
    else:
        array = np.asarray(value)
        if array.size == 0:
            return "<missing>"
        if array.size == 1:
            text = str(array.reshape(-1)[0]).strip()
        else:
            text = " ".join(str(item).strip() for item in array.reshape(-1)).strip()
    return text if text and text.lower() not in {"nan", "none"} else "<missing>"


def collapse_sites(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("site_id", as_index=False).agg(
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        event_change_z=("event_change_z", "mean"),
        record_count=("proxy_index", "size"),
    )


def retained_result(frame: pd.DataFrame, omitted: pd.Series, label: str, kind: str) -> dict[str, object]:
    full_sites = collapse_sites(frame)
    omitted_records = frame[omitted]
    retained_records = frame[~omitted]
    retained_sites = collapse_sites(retained_records)
    return {
        "deletion_kind": kind,
        "omitted_group": label,
        "omitted_record_count": int(omitted.sum()),
        "omitted_unique_site_count": int(omitted_records["site_id"].nunique()),
        "omitted_record_mean_event_change_z": float(omitted_records["event_change_z"].mean()),
        "retained_record_count": int(len(retained_records)),
        "retained_site_count": int(len(retained_sites)),
        "full_site_mean_event_change_z": float(full_sites["event_change_z"].mean()),
        "retained_site_mean_event_change_z": float(retained_sites["event_change_z"].mean()),
        "change_from_full_site_mean_z": float(retained_sites["event_change_z"].mean() - full_sites["event_change_z"].mean()),
        "retained_sign_negative": bool(float(retained_sites["event_change_z"].mean()) < 0),
        "eligible_interpretation": bool(int(omitted.sum()) >= 2 and len(retained_sites) >= 10),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    metadata_root = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_pages2k_fixed_network_metadata_family_v1"
    output = root / f"validation_data/results/paleo_surface_validation_v0/{EXPERIMENT_ID}"
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")

    sources = {
        "fixed_rich_metadata": metadata_root / "fixed_network_rich_pages2k_metadata.csv",
        "identity_chain": metadata_root / "pages2k_identity_chain.csv",
        "metadata_summary": metadata_root / "summary.json",
        "pages_ts": PAGES_TS,
        "facts": root / "LIA_PROJECT_FACTS.md",
    }
    if missing := [str(path) for path in sources.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    before = {name: source_state(path) for name, path in sources.items()}

    records = pd.read_csv(sources["fixed_rich_metadata"])
    identity = pd.read_csv(sources["identity_chain"])
    identity_summary = json.loads(sources["metadata_summary"].read_text(encoding="utf-8"))
    if len(records) != 50 or len(identity) != 47 or not identity_summary["all_identity_chains_unique"]:
        raise RuntimeError("upstream-identity-contract-failed")

    ts_values = np.asarray(loadmat(PAGES_TS, squeeze_me=True, struct_as_record=False, variable_names=["TS"])["TS"]).reshape(-1)
    index: dict[tuple[str, str], list[object]] = defaultdict(list)
    for item in ts_values:
        index[(clean(getattr(item, "dataSetName", "")), clean(getattr(item, "paleoData_TSid", "")))].append(item)

    study_rows = []
    for row in identity.itertuples(index=False):
        matches = index.get((row.official_dataset_name, row.official_tsid), [])
        if len(matches) != 1:
            raise RuntimeError(f"ts-study-identity-not-unique:{row.proxy_index}:{len(matches)}")
        item = matches[0]
        values = {"proxy_index": int(row.proxy_index)}
        for output_field, source_field in STUDY_FIELDS.items():
            values[output_field] = clean(getattr(item, source_field, ""))
        study_rows.append(values)
    study = pd.DataFrame(study_rows)
    records = records.merge(study, on="proxy_index", how="left", validate="one_to_one")
    for field in STUDY_FIELDS:
        records[field] = records[field].fillna("<non_pages2k_named_tree>")

    pages_mask = records["source_family"] == "PAGES2k2017"
    single_rows = []
    for field in ("data_author", "data_cite_key", "publication_cite_key"):
        for group in sorted(records.loc[pages_mask, field].unique()):
            omitted = pages_mask & (records[field] == group)
            single_rows.append(retained_result(records, omitted, f"{field}={group}", "single_study_group"))
    single = pd.DataFrame(single_rows)

    cook = pages_mask & records["data_author"].eq("Cook, E.R.")
    shao = pages_mask & records["data_author"].eq("Shao, X.M.")
    pages_consortium = pages_mask & records["data_author"].eq("PAGES 2k Consortium")
    north_west = pages_mask & (records["latitude"] >= 35.0) & (records["longitude"] < 80.0)
    north_east = pages_mask & (records["latitude"] >= 35.0) & (records["longitude"] >= 90.0) & (records["longitude"] < 101.0)
    north_all = pages_mask & (records["latitude"] >= 35.0)
    joint_definitions = {
        "Cook": cook,
        "Shao": shao,
        "PAGES_consor": pages_consortium,
        "Cook_or_Shao": cook | shao,
        "Cook_or_Shao_or_PAGES_consor": cook | shao | pages_consortium,
        "north_west_ge35_lt80E": north_west,
        "north_east_ge35_90to101E": north_east,
        "north_west_or_east": north_west | north_east,
        "all_PAGES_ge35N": north_all,
    }
    joint = pd.DataFrame([
        retained_result(records, omitted, label, "predefined_joint_or_spatial_block")
        for label, omitted in joint_definitions.items()
    ])

    north_records = records[records["latitude"] >= 35.0].copy()
    north_pages = north_records["source_family"] == "PAGES2k2017"
    north_rows = []
    for group in sorted(north_records.loc[north_pages, "data_author"].unique()):
        omitted = north_pages & north_records["data_author"].eq(group)
        north_rows.append(retained_result(north_records, omitted, f"data_author={group}", "north_single_study_group"))
    north_single = pd.DataFrame(north_rows)

    group_summary = records[pages_mask].groupby("data_author", as_index=False).agg(
        record_count=("proxy_index", "size"),
        unique_site_count=("site_id", "nunique"),
        mean_event_change_z=("event_change_z", "mean"),
        latitude_min=("latitude", "min"),
        latitude_max=("latitude", "max"),
        longitude_min=("longitude", "min"),
        longitude_max=("longitude", "max"),
    ).sort_values("record_count", ascending=False)

    output.mkdir(parents=True)
    records.to_csv(output / "fixed_network_study_metadata.csv", index=False)
    group_summary.to_csv(output / "study_group_summary.csv", index=False)
    single.to_csv(output / "single_study_group_deletion.csv", index=False)
    north_single.to_csv(output / "north_study_group_deletion.csv", index=False)
    joint.to_csv(output / "joint_study_and_spatial_block_deletion.csv", index=False)

    eligible_single = single[single["eligible_interpretation"]]
    eligible_north = north_single[north_single["eligible_interpretation"]]
    all_study_negative = bool(eligible_single["retained_sign_negative"].all())
    north_study_negative = bool(eligible_north["retained_sign_negative"].all())
    cook_shao = joint[joint["omitted_group"] == "Cook_or_Shao"].iloc[0]
    both_spatial = joint[joint["omitted_group"] == "north_west_or_east"].iloc[0]
    all_north = joint[joint["omitted_group"] == "all_PAGES_ge35N"].iloc[0]

    summary = {
        "fixed_record_count": int(len(records)),
        "pages2k_record_count": int(pages_mask.sum()),
        "data_author_group_count": int(records.loc[pages_mask, "data_author"].nunique()),
        "largest_data_author_groups": group_summary.head(5).to_dict(orient="records"),
        "eligible_single_group_deletion_count": int(len(eligible_single)),
        "all_eligible_single_study_deletions_retain_negative_sign": all_study_negative,
        "eligible_north_single_group_deletion_count": int(len(eligible_north)),
        "all_eligible_north_single_study_deletions_retain_negative_sign": north_study_negative,
        "cook_or_shao_retained_site_mean_z": float(cook_shao["retained_site_mean_event_change_z"]),
        "north_west_or_east_retained_site_mean_z": float(both_spatial["retained_site_mean_event_change_z"]),
        "all_pages_ge35n_retained_site_mean_z": float(all_north["retained_site_mean_event_change_z"]),
        "classification": (
            "two_independent_northern_study_batches_share_the_negative_event"
            if all_study_negative and north_study_negative
            else "single_study_batch_sensitivity_detected"
        ),
        "interpretation": (
            "Study-batch deletion tests redundancy and concentration. It does not decide whether the shared event is climatic, "
            "chronology-related, or an operator mismatch."
        ),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8), constrained_layout=True)
    ordered_authors = group_summary["data_author"].tolist()
    positions = {author: index for index, author in enumerate(ordered_authors)}
    rng = np.random.default_rng(20260815)
    pages_records = records[pages_mask].copy()
    y = np.asarray([positions[value] for value in pages_records["data_author"]], dtype=float)
    y += rng.uniform(-0.12, 0.12, size=len(y))
    axes[0].scatter(pages_records["event_change_z"], y, c=pages_records["latitude"], cmap="viridis", s=42, alpha=0.85)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_yticks(np.arange(len(ordered_authors)), ordered_authors, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("1901-1920 event change (z)")
    axes[0].set_title("Records by data author")

    view = eligible_single.sort_values("retained_site_mean_event_change_z").copy()
    axes[1].scatter(view["retained_site_mean_event_change_z"], np.arange(len(view)), color="#C9483B", s=38)
    axes[1].axvline(0, color="black", linewidth=0.8)
    if len(view):
        axes[1].axvline(float(view["full_site_mean_event_change_z"].iloc[0]), color="#247BA0", linestyle="--", linewidth=1.1)
    axes[1].set_yticks(np.arange(len(view)), view["omitted_group"], fontsize=7)
    axes[1].set_xlabel("Retained all-site mean (z)")
    axes[1].set_title("Single study-group deletions")

    view_joint = joint.sort_values("retained_site_mean_event_change_z").copy()
    axes[2].scatter(view_joint["retained_site_mean_event_change_z"], np.arange(len(view_joint)), color="#E07A1F", s=42)
    axes[2].axvline(0, color="black", linewidth=0.8)
    axes[2].axvline(float(view_joint["full_site_mean_event_change_z"].iloc[0]), color="#247BA0", linestyle="--", linewidth=1.1)
    axes[2].set_yticks(np.arange(len(view_joint)), view_joint["omitted_group"], fontsize=7)
    axes[2].set_xlabel("Retained all-site mean (z)")
    axes[2].set_title("Joint and spatial-block deletions")
    fig.savefig(output / "pages2k_study_batch_robustness.png", dpi=220)
    plt.close(fig)

    after = {name: source_state(path) for name, path in sources.items()}
    unchanged = before == after
    summary["all_declared_sources_unchanged"] = unchanged
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
    if not unchanged:
        raise RuntimeError("declared-source-changed-during-audit")
    (output / "FINAL_RESULT.md").write_text(
        "# PAGES2k Fixed-network Study-batch Audit v1\n\n"
        f"All eligible single-study deletions retain a negative all-site event: {all_study_negative}. "
        f"All eligible northern single-study deletions retain a negative northern event: {north_study_negative}.\n\n"
        f"Removing Cook and Shao together leaves {float(cook_shao['retained_site_mean_event_change_z']):.3f} z. "
        f"Removing the predefined western and eastern northern blocks leaves {float(both_spatial['retained_site_mean_event_change_z']):.3f} z. "
        f"Removing all PAGES records at >=35N leaves {float(all_north['retained_site_mean_event_change_z']):.3f} z.\n\n"
        "The negative event is shared by more than one northern study batch. This robustness result does not establish whether the event is climatic or proxy/PSM related.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
