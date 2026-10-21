"""Run one named planned task from the Core-191 tradeoff pilot manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .lia_191_controller import _atomic_csv, _run_one, controller_root, load_config
from .lia_191_tradeoff_pilot import sync


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    root = controller_root(config)
    manifest_path = root / "task_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    selected = manifest[
        manifest["task_id"].astype(str).eq(args.task_id)
        & manifest["task_state"].eq("planned")
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one planned task {args.task_id!r}, found {len(selected)}"
        )

    result = _run_one(config, selected.iloc[0].to_dict())
    rows = []
    for row in manifest.to_dict(orient="records"):
        rows.append(result if row["task_id"] == args.task_id else row)
    _atomic_csv(manifest_path, pd.DataFrame(rows))
    scores = sync(config)
    print(
        f"task={result['task_id']} state={result['task_state']} "
        f"scored={len(scores)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
