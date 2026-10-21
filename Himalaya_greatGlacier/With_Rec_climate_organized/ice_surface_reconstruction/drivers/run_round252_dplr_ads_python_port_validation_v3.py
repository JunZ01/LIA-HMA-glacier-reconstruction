from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


TOLERANCE = 1e-8


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ads_core(y_input: np.ndarray, stiffness_input: np.ndarray) -> np.ndarray:
    """Direct 1-based translation of dplR 1.7.9 src/adsf.f95."""
    values = np.asarray(y_input, dtype=float)
    stiffness_values = np.asarray(stiffness_input, dtype=int)
    n = len(values)
    if n < 4:
        raise ValueError("too-few-points")
    if stiffness_values.shape != (n,):
        raise ValueError("stiffness-shape")
    nm2 = n - 2
    y = np.zeros(n + 1, dtype=float)
    y[1:] = values
    stiffness = np.zeros(n + 1, dtype=int)
    stiffness[1:] = stiffness_values
    a = np.zeros((nm2 + 1, 5), dtype=float)
    p = np.zeros(nm2 + 1, dtype=float)
    c1 = np.array([0.0, 1.0, -4.0, 6.0, -2.0], dtype=float)
    c2 = np.array([0.0, 0.0, 1.0 / 3.0, 4.0 / 3.0], dtype=float)
    for i in range(1, nm2 + 1):
        v = float(stiffness[i])
        arg = 2.0 * math.pi / v
        p[i] = 6.0 * (math.cos(arg) - 1.0) ** 2 / (math.cos(arg) + 2.0)
    for i in range(1, nm2 + 1):
        for j in range(1, 4):
            a[i, j] = c1[j] + p[i] * c2[j]
            a[i, 4] = y[i] + c1[4] * y[i + 1] + y[i + 2]
    a[1, 1] = c2[1]
    a[1, 2] = c2[1]
    a[2, 1] = c2[1]
    nc = 2
    rn = 1.0 / (nm2 * 16.0)
    ncp1 = nc + 1
    for i in range(1, nc + 1):
        for j in range(i, nc + 1):
            k = ncp1 - j
            a[i, k] = 0.0
    for i in range(1, nm2 + 1):
        imncp1 = i - ncp1
        i1 = max(1, 1 - imncp1)
        for j in range(i1, ncp1 + 1):
            l = imncp1 + j
            i2 = ncp1 - j
            total = a[i, j]
            jm1 = j - 1
            if jm1 > 0:
                for k in range(1, jm1 + 1):
                    m = i2 + k
                    total -= a[i, k] * a[l, m]
            if j == ncp1:
                if a[i, j] + total * rn <= a[i, j] or total <= 0:
                    raise RuntimeError(f"ads-matrix-not-positive-definite:i={i}:j={j}:total={total:.17g}:stored={a[i,j]:.17g}:rn={rn:.17g}")
                a[i, j] = 1.0 / math.sqrt(total)
            else:
                a[i, j] = total * a[l, ncp1]
    nc1 = nc + 1
    iw = 0
    lag = 0
    for i in range(1, nm2 + 1):
        total = a[i, 4]
        if nc > 0:
            if iw != 0:
                lag += 1
                if lag > nc:
                    lag = nc
                k = nc1 - lag
                kl = i - lag
                for j in range(k, nc + 1):
                    total -= a[kl, 4] * a[i, j]
                    kl += 1
            elif total != 0.0:
                iw = 1
        a[i, 4] = total * a[i, nc1]
    a[nm2, 4] *= a[nm2, nc1]
    n1 = nm2 + 1
    for i in range(2, nm2 + 1):
        k = n1 - i
        total = a[k, 4]
        if nc > 0:
            kl = k + 1
            k1 = min(nm2, k + nc)
            lag = 1
            for j in range(kl, k1 + 1):
                total -= a[j, 4] * a[j, nc1 - lag]
                lag += 1
        a[k, 4] = total * a[k, nc1]
    residual = np.zeros(n + 1, dtype=float)
    for i in range(3, nm2 + 1):
        residual[i] = a[i - 2, 4] + c1[4] * a[i - 1, 4] + a[i, 4]
    residual[1] = a[1, 4]
    residual[2] = c1[4] * a[1, 4] + a[2, 4]
    residual[n - 1] = a[nm2 - 1, 4] + c1[4] * a[nm2, 4]
    residual[n] = a[nm2, 4]
    return values - residual[1:]


def ads(y: np.ndarray, nyrs0: int = 50, pos_slope: bool = True) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    stiffness = np.arange(nyrs0, nyrs0 + len(values), dtype=int)
    fitted = ads_core(values, stiffness)
    if not pos_slope:
        differences = np.r_[0.0, np.diff(fitted)]
        eligible = np.flatnonzero(differences <= 0)
        cutoff = int(eligible[-1]) if len(eligible) else 0
        constrained = fitted.copy()
        constrained[cutoff:] = constrained[cutoff]
        fitted = ads_core(constrained, stiffness)
    return fitted


def main() -> int:
    root = project_root()
    drivers = Path(__file__).resolve().parent
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_252_dplr_ads_python_port_validation_v3"
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": drivers / "round252_dplr_ads_python_port_validation_v1_contract.md",
        "driver": Path(__file__).resolve(),
        "ads_fortran": loop / "round_251_dplr179_method_source_acquisition_v1/vendor_source/src/adsf.f95",
        "ads_r": loop / "round_251_dplr179_method_source_acquisition_v1/vendor_source/R/ads.R",
        "round246_driver": drivers / "run_round246_npl065_calibration_endpoint_attribution_v2.py",
        "rwl": loop / "round_237_npl065_source_acquisition_preflight_v1/raw_unopened/npl065.rwl",
    }
    before = {name: fingerprint(path) for name, path in paths.items()}
    vendor = loop / "round_167_chin074_qamdo2_identity_recovery_v1/vendor"
    sys.path.insert(0, str(vendor))
    import dplpy
    from smoothingspline import spline

    round246 = load_module("round252_round246", paths["round246_driver"])
    raw, parse_metadata = round246.parse_modern_rwl(paths["rwl"])
    sample = raw["GRC010"].dropna().to_numpy(float)
    x = np.arange(1, 74, dtype=float)
    vectors = {
        "constant": np.full(73, 5.0),
        "linear": 2.0 + 0.03 * x,
        "trend_cycle": 3.0 + 0.02 * x + 0.4 * np.sin(2 * np.pi * x / 11.0) + 0.15 * np.cos(2 * np.pi * x / 5.0),
        "npl065_GRC010_modern": sample,
    }
    rows = []
    for name, values in vectors.items():
        constant_stiffness = np.full(len(values), 30, dtype=int)
        port = ads_core(values, constant_stiffness)
        reference = np.asarray(spline(np.arange(len(values), dtype=float), values, 30), dtype=float)
        rows.append({
            "vector": name,
            "length": len(values),
            "max_absolute_error_vs_dplpy_constant30": float(np.max(np.abs(port - reference))),
            "rmse_vs_dplpy_constant30": float(np.sqrt(np.mean((port - reference) ** 2))),
            "age_dependent_output_finite": bool(np.isfinite(ads(values, nyrs0=30)).all()),
        })
    max_error = max(row["max_absolute_error_vs_dplpy_constant30"] for row in rows)
    constant_error = float(np.max(np.abs(ads_core(vectors["constant"], np.full(73, 30)) - vectors["constant"])))
    linear_error = float(np.max(np.abs(ads_core(vectors["linear"], np.full(73, 30)) - vectors["linear"])))
    passes = bool(max_error <= TOLERANCE and constant_error <= TOLERANCE and linear_error <= TOLERANCE)
    if not passes:
        decision = "reject_ads_python_port"
    else:
        decision = "admit_ads_python_port_to_modern_only_method_experiment"
    after = {name: fingerprint(path) for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-fingerprint-changed")
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "source": {"dplr_version": "1.7.9", "algorithm": "direct_translation_of_src_adsf_f95"},
        "tests": rows,
        "max_constant_stiffness_closure_error": max_error,
        "constant_invariant_max_error": constant_error,
        "linear_invariant_max_error": linear_error,
        "tolerance": TOLERANCE,
        "passes": passes,
        "decision": decision,
        "parse_metadata": parse_metadata,
        "climate_values_accessed": False,
        "locked_1921_1939_numeric_values_decoded": False,
        "paleo_proxy_values_decoded": False,
        "jionglacuo_opened_or_hashed": False,
        "binary_executed": False,
        "enkf_run": False,
        "production_reconstruction_run": False,
        "oggm_run": False,
        "source_fingerprints_unchanged": unchanged,
    }
    (output / "validation_summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after}, indent=2), encoding="ascii")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 252 dplR ADS Python-port validation\n\n"
        f"- Maximum constant-30 spline closure error: `{max_error:.3e}`.\n"
        f"- Constant invariant error: `{constant_error:.3e}`.\n"
        f"- Linear invariant error: `{linear_error:.3e}`.\n"
        f"- Passes frozen tolerance: `{str(passes).lower()}`.\n\n"
        f"Decision: `{decision}`.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    if not passes:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
