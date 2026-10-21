"""Read-only v3 essential curve standard for the fixed retained-32 cohort.

This script consumes only the immutable v1/v2 analysis products.  It does not
run OGGM, change the accepted ledger, or mutate controller state.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
V1 = BASE / "lia_191_retained_32_curve_analysis_v1"
V2 = BASE / "lia_191_retained_32_curve_analysis_v2_executable_standard"
OUTPUT = BASE / "lia_191_retained_32_curve_analysis_v3_essential_standard"

SMOOTH_YEARS = 21
ACTIVE_DERIVATIVE_SD = 0.25
MAX_NEUTRAL_BRIDGE_YEARS = 5
MIN_PHASE_YEARS = 20
MIN_PHASES_EACH_DIRECTION = 2
MIN_SUSTAINED_VARIATION_SHARE = 0.60


def runs(states: np.ndarray) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, len(states) + 1):
        if index == len(states) or states[index] != states[start]:
            result.append((int(states[start]), start, index - 1))
            start = index
    return result


def persistent_directional_phases(
    derivative: np.ndarray,
) -> list[tuple[int, int, int]]:
    threshold = ACTIVE_DERIVATIVE_SD * float(np.std(derivative))
    states = np.where(
        derivative > threshold,
        1,
        np.where(derivative < -threshold, -1, 0),
    ).astype(int)

    # Join only short neutral gaps between phases of the same direction.
    for sign in (-1, 1):
        indices = np.flatnonzero(states == sign)
        for left, right in zip(indices[:-1], indices[1:]):
            gap = right - left - 1
            if 0 < gap <= MAX_NEUTRAL_BRIDGE_YEARS and np.all(
                states[left + 1 : right] == 0
            ):
                states[left + 1 : right] = sign

    return [
        phase
        for phase in runs(states)
        if phase[0] != 0 and phase[2] - phase[1] + 1 >= MIN_PHASE_YEARS
    ]


def phase_metrics(group: pd.DataFrame) -> dict[str, float | int | bool]:
    group = group.sort_values("year")
    derivative = group["dln_length_21yr_per_year"].to_numpy(float)
    length = group["length_21yr_km"].to_numpy(float)
    phases = persistent_directional_phases(derivative)

    positive = sum(sign > 0 for sign, _, _ in phases)
    negative = sum(sign < 0 for sign, _, _ in phases)
    total_variation = float(np.sum(np.abs(np.diff(length))))
    sustained_variation = 0.0
    for _, start, end in phases:
        sustained_variation += float(np.sum(np.abs(np.diff(length[start : end + 1]))))
    share = sustained_variation / max(total_variation, 1.0e-12)

    passed = bool(
        positive >= MIN_PHASES_EACH_DIRECTION
        and negative >= MIN_PHASES_EACH_DIRECTION
        and share >= MIN_SUSTAINED_VARIATION_SHARE
    )
    return {
        "persistent_growth_phase_count": positive,
        "persistent_retreat_phase_count": negative,
        "persistent_phase_count_total": len(phases),
        "sustained_variation_share": share,
        "bidirectional_multistage_gate_pass": passed,
    }


def bool_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def markdown_table(table: pd.DataFrame) -> str:
    lines = [
        "| ID | 端点上下界 | 滞后响应 | 增长/退缩持续阶段 | 持续变化占比 | 伪影 | 证据冲突 | 暂定结论 |\n",
        "|---:|---|---|---:|---:|---|---|---|\n",
    ]
    for _, row in table.sort_values("system_id").iterrows():
        artifacts: list[str] = []
        if row["flag_terminal_endpoint_drop"]:
            artifacts.append("末期端点坠落")
        if row["flag_high_frequency"]:
            artifacts.append("高频振荡")
        lines.append(
            f"| {int(row['system_id'])} | "
            f"{'通过' if row['original_joint_endpoint_bounds_pass'] else '失败'} | "
            f"{'通过' if row['lag_gate_pass'] else '失败/边界'} "
            f"({int(row['length_best_lag_years'])}年, r={row['length_best_lag_correlation']:.2f}, "
            f"同号={row['length_lag_sign_consistency']:.2f}) | "
            f"{int(row['persistent_growth_phase_count'])}/{int(row['persistent_retreat_phase_count'])} "
            f"({'通过' if row['bidirectional_multistage_gate_pass'] else '失败'}) | "
            f"{row['sustained_variation_share']:.2f} | "
            f"{', '.join(artifacts) if artifacts else '无'} | "
            f"{'是' if row['evidence_conflict'] else '否'} | {row['proposed_decision_zh']} |\n"
        )
    return "".join(lines)


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite versioned output: {OUTPUT}")

    annual = pd.read_csv(V1 / "annual_smoothed_multivariate_series.csv")
    prior = pd.read_csv(V2 / "rescored_32_executable_standard.csv")
    if len(prior) != 32 or prior["system_id"].nunique() != 32:
        raise RuntimeError("v2 input is not exactly retained-32")

    phase_rows = []
    for system_id, group in annual.groupby("system_id"):
        phase_rows.append({"system_id": int(system_id), **phase_metrics(group)})
    phases = pd.DataFrame(phase_rows)

    keep = [
        "system_id",
        "rgi_id",
        "model_family_registry_label",
        "route",
        "source",
        "source_sha256",
        "source_status",
        "analysis_status",
        "direct_surface_mb_available",
        "mb_role",
        "original_joint_endpoint_bounds_pass",
        "length_best_lag_years",
        "length_best_lag_correlation",
        "length_lag_sign_consistency",
        "lag_at_upper_boundary",
        "lag_gate_pass",
        "flag_terminal_endpoint_drop",
        "flag_high_frequency",
        "evidence_conflict",
    ]
    table = prior[keep].merge(phases, on="system_id", validate="one_to_one")
    for column in [
        "original_joint_endpoint_bounds_pass",
        "lag_at_upper_boundary",
        "lag_gate_pass",
        "flag_terminal_endpoint_drop",
        "flag_high_frequency",
        "evidence_conflict",
        "bidirectional_multistage_gate_pass",
    ]:
        table[column] = table[column].map(bool_value)

    table["artifact_gate_pass"] = ~(
        table["flag_terminal_endpoint_drop"] | table["flag_high_frequency"]
    )
    table["all_scientific_curve_gates_pass"] = (
        table["original_joint_endpoint_bounds_pass"]
        & table["lag_gate_pass"]
        & table["bidirectional_multistage_gate_pass"]
        & table["artifact_gate_pass"]
    )

    decisions_en: list[str] = []
    decisions_zh: list[str] = []
    reasons: list[str] = []
    for _, row in table.iterrows():
        failed: list[str] = []
        if not row["original_joint_endpoint_bounds_pass"]:
            failed.append("endpoint-bounds")
        if not row["lag_gate_pass"]:
            failed.append("lag-response")
        if not row["bidirectional_multistage_gate_pass"]:
            failed.append("insufficient-bidirectional-persistent-phases")
        if row["flag_terminal_endpoint_drop"]:
            failed.append("terminal-endpoint-drop")
        if row["flag_high_frequency"]:
            failed.append("numerical-high-frequency")
        reasons.append(";".join(failed) if failed else "none")

        if row["evidence_conflict"]:
            decisions_en.append("evidence-conflict-hold")
            decisions_zh.append("证据冲突，暂缓")
        elif row["all_scientific_curve_gates_pass"]:
            decisions_en.append("provisional-forward-curve-pass")
            decisions_zh.append("暂定通过")
        else:
            decisions_en.append("curve-screen-fail")
            decisions_zh.append("淘汰")
    table["failed_scientific_reasons"] = reasons
    table["proposed_decision"] = decisions_en
    table["proposed_decision_zh"] = decisions_zh

    passes = table.loc[
        table["proposed_decision"] == "provisional-forward-curve-pass", "system_id"
    ].astype(int).tolist()
    failures = table.loc[
        table["proposed_decision"] == "curve-screen-fail", "system_id"
    ].astype(int).tolist()
    holds = table.loc[
        table["proposed_decision"] == "evidence-conflict-hold", "system_id"
    ].astype(int).tolist()

    OUTPUT.mkdir(parents=True, exist_ok=False)
    table.to_csv(OUTPUT / "rescored_32_essential_standard_v3.csv", index=False)
    (OUTPUT / "rescored_32_essential_standard_v3.md").write_text(
        markdown_table(table), encoding="utf-8"
    )

    standard = {
        "schema": "lia-retained32-essential-curve-standard-v3",
        "status": "proposed-read-only-not-canonical",
        "population": 32,
        "source_analysis": str(V1.relative_to(ROOT)).replace("\\", "/"),
        "source_endpoint_score": str(V2.relative_to(ROOT)).replace("\\", "/"),
        "core_scientific_gates": {
            "endpoint_bounds": {
                "lia_peak_total_error_max": 0.10,
                "rgi_2000_total_error_max": 0.10,
                "lia_peak_separation_error_max": 0.20,
                "rgi_2000_separation_error_max": 0.20,
            },
            "lagged_internal_length_response": {
                "smooth_years": SMOOTH_YEARS,
                "driver": "derived net-volume-change-equivalent B(t); not direct surface MB",
                "response": "dlnL/dt",
                "lag_years_inclusive": [0, 100],
                "minimum_correlation": 0.30,
                "minimum_sign_consistency": 0.55,
                "best_lag_at_100": "hold for boundary review",
            },
            "bidirectional_multistage_response": {
                "active_derivative_deadband_sd": ACTIVE_DERIVATIVE_SD,
                "maximum_neutral_bridge_years": MAX_NEUTRAL_BRIDGE_YEARS,
                "minimum_phase_years": MIN_PHASE_YEARS,
                "minimum_persistent_growth_phases": MIN_PHASES_EACH_DIRECTION,
                "minimum_persistent_retreat_phases": MIN_PHASES_EACH_DIRECTION,
                "minimum_sustained_variation_share": MIN_SUSTAINED_VARIATION_SHARE,
                "note": "counts sustained directional phases, not raw peaks",
            },
            "artifact_exclusion": {
                "terminal_endpoint_drop": "fail",
                "numerical_high_frequency_oscillation": "fail",
            },
        },
        "not_hard_gates": [
            "sign of net 1500-2000 length change",
            "raw peak count",
            "peak year alone",
            "area amplitude",
            "volume amplitude",
            "B-to-volume agreement (definitional)",
            "whole-curve corridor",
            "minimum at year 2000",
            "modern MB continuation",
        ],
        "formal_acceptance_requirement": (
            "all curve gates plus inherited numerical/geometry/source gates, "
            "independent reproduction, and controller evidence transaction"
        ),
    }
    (OUTPUT / "executable_standard_v3.json").write_text(
        json.dumps(standard, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "schema": "lia-retained32-essential-standard-result-v3",
        "status": "read-only-proposed-no-controller-effect",
        "population": 32,
        "provisional_pass_count": len(passes),
        "provisional_pass_system_ids": passes,
        "curve_screen_fail_count": len(failures),
        "curve_screen_fail_system_ids": failures,
        "evidence_conflict_hold_count": len(holds),
        "evidence_conflict_hold_system_ids": holds,
        "canonical_accepted_count_effect": 0,
        "simulation_launched": False,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = f"""# 32 条曲线：抓住本质的可执行标准 v3

状态：**只读建议；非 canonical；不改变 accepted=58，不授权任何模拟。**

## 核心判断

`1500–2000` 平滑长度净增或净减不再是硬门槛。净变化只描述首尾，不能区分“中间经历多阶段响应”与“长期单向、末端一次转折”。峰数和峰年也不单独决定通过。

真正的形态核心是：**长度必须对两个方向都出现重复、持续的低频响应，而不能主要靠一个长期单向阶段或末端突变命中端点。**

统一处理为 21 年居中反射平滑；在 `dlnL/dt` 上以 `0.25 × 标准差` 设活动死区，最多桥接 5 个中性年。一个有效阶段至少持续 20 年。通过要求：

- 至少 2 个持续增长阶段；
- 至少 2 个持续退缩阶段；
- 这些持续阶段合计解释至少 60% 的平滑长度总变化。

这是“持续方向阶段”计数，不是原始峰数。一个宽峰、一两个局部峰或峰年本身均不导致淘汰。

## 四个科学核心门槛

1. **原有端点上下界**：1470–2000 前向峰值和 2000 年长度，各自同时满足总长度误差不超过 10%、端点分离归一化误差不超过 20%。
2. **滞后内部响应**：由真实存在的体积和面积推导 `B(t)=rho*dV/mean(A)`；21 年平滑后，搜索 `B(t)` 到较晚 `dlnL/dt` 的 0–100 年滞后，要求 `r >= 0.30`、方向同号率 `>= 0.55`；最优值恰在 100 年边界则暂缓。这里证明的是内部几何响应一致性，不等同于独立气候验证。
3. **双向多阶段覆盖**：采用上述至少 `2 个增长 + 2 个退缩` 的持续阶段，并要求持续阶段贡献至少 60% 总变化。
4. **伪影排除**：末期端点坠落或数值高频振荡直接淘汰。

面积幅度、体积幅度、峰数、峰年、1500–2000 净变化、整段 corridor、2000 年是否为最小值和现代 MB 延伸均只作诊断，不是本标准硬门槛。

## 32 条暂定结果

- 暂定通过：**{len(passes)} 条**。
- 曲线淘汰：**{len(failures)} 条** — {', '.join(map(str, failures))}。
- 证据冲突暂缓：**{len(holds)} 条** — {', '.join(map(str, holds))}。

### 两条淘汰曲线

- **2827**：3 个持续增长阶段、仅 1 个持续退缩阶段；持续变化占比虽够，但不满足双向重复响应，因此属于增长主导、末段一次退缩的单周期曲线。淘汰理由不是净增长本身。
- **4565**：同样为 3 个持续增长阶段、仅 1 个持续退缩阶段；并且独立触发末期端点坠落。属于长期增长后把主要退缩压到终点附近的曲线。

### 暂缓成员

- **1326**：科学曲线门槛通过，但 accepted registry 与正式历史/复现实证的 LIA 分离误差存在冲突；在证据统一前不得据此作新接受决定。

## Forward run 使用方式

四个科学门槛全部通过，只能标记为“暂定 forward 曲线通过”。正式接受还必须继承上游数值、几何、来源哈希和观测域门槛，并完成独立 reproduction 与 controller 证据事务。该标准不改变当前 accepted=58。
"""
    (OUTPUT / "essential_standard_v3_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
