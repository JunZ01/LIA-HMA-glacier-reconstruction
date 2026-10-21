from __future__ import annotations

import json
from pathlib import Path

import matplotlib.axes

import run_temperature_multimodel_stage_b_pseudoproxy_v1 as v1
from run_temperature_multimodel_stage_b_pseudoproxy_v2 import SharedObservationNoise


DRIVER = Path(__file__).resolve()
V1_DRIVER = Path(v1.__file__).resolve()
V2_DRIVER = DRIVER.with_name("run_temperature_multimodel_stage_b_pseudoproxy_v2.py")
v1.OUT = v1.LOOP / "temperature_multimodel_evolving_prior_stage_b_pseudoproxy_v3"


ORIGINAL_BOXPLOT = matplotlib.axes.Axes.boxplot


def compatible_boxplot(self, *args, **kwargs):
    if "tick_labels" in kwargs:
        kwargs["labels"] = kwargs.pop("tick_labels")
    return ORIGINAL_BOXPLOT(self, *args, **kwargs)


def main() -> None:
    v1.np.random.default_rng = lambda seed: SharedObservationNoise(seed)
    matplotlib.axes.Axes.boxplot = compatible_boxplot
    v1.__file__ = str(DRIVER)
    v1.main()

    summary_path = v1.OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["supersedes_technical_failure"] = (
        "temperature_multimodel_evolving_prior_stage_b_pseudoproxy_v2"
    )
    summary["v3_change"] = (
        "translate matplotlib boxplot tick_labels to labels; scientific computation unchanged"
    )
    summary["shared_noise_contract"] = (
        "one proxy6/proxy9 noise realization shared across all three nontruth priors"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest_path = v1.OUT / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["implementation_v1"] = {
        "path": str(V1_DRIVER),
        "sha256": v1.sha256(V1_DRIVER),
    }
    manifest["shared_noise_wrapper_v2"] = {
        "path": str(V2_DRIVER),
        "sha256": v1.sha256(V2_DRIVER),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
