"""Plot all formally frozen Core-191 length curves and build an inline gallery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_registry(registry_root: Path) -> list[dict]:
    registry = pd.read_csv(registry_root / "frozen_candidates.csv")
    registry["lia_unique_id"] = pd.to_numeric(registry["lia_unique_id"]).astype(int)
    if len(registry) != 21 or registry["lia_unique_id"].nunique() != 21:
        raise RuntimeError(
            f"Expected 21 unique frozen systems, found {len(registry)}/"
            f"{registry['lia_unique_id'].nunique()}"
        )
    curves: list[dict] = []
    for row in registry.sort_values("lia_unique_id").itertuples(index=False):
        uid = int(row.lia_unique_id)
        system_root = registry_root / f"system_{uid}"
        record = json.loads(
            (system_root / "candidate_record.json").read_text(encoding="utf-8")
        )
        series = pd.read_csv(system_root / "timeseries.csv")
        forward = series.loc[series["phase"].astype(str).eq("forward")].copy()
        if forward.empty:
            raise RuntimeError(f"System {uid} has no frozen forward curve")
        length_column = (
            "observation_length_m"
            if "observation_length_m" in forward.columns
            else "length_m"
        )
        forward["year"] = pd.to_numeric(forward["year"], errors="coerce")
        forward["length_m"] = pd.to_numeric(
            forward[length_column], errors="coerce"
        )
        forward = forward.dropna(subset=["year", "length_m"]).sort_values("year")
        scores = record["endpoint_scores"]
        params = record["parameters"]
        lia = float(scores["lia_target_m"])
        rgi = float(scores["rgi_target_m"])
        separation = abs(lia - rgi)
        curves.append(
            {
                "lia_unique_id": uid,
                "rgi_id": str(record["rgi_id"]),
                "spinup_delta_t_k": params.get(
                    "spinup_delta_t_k", params.get("delta_t_k")
                ),
                "forward_delta_t_k": params.get(
                    "forward_delta_t_k", params.get("delta_t_k")
                ),
                "lia_target_km": lia / 1000.0,
                "rgi_target_km": rgi / 1000.0,
                "tolerance_km": 0.2 * separation / 1000.0,
                "forward_peak_year": int(scores["forward_peak_year"]),
                "forward_peak_km": float(scores["forward_peak_m"]) / 1000.0,
                "forward_2000_km": float(scores["forward_2000_m"]) / 1000.0,
                "e_lia_sep": float(scores["e_lia_sep"]),
                "e_rgi_sep": float(scores["e_rgi_sep"]),
                "points": [
                    {
                        "year": int(round(float(item.year))),
                        "length_km": round(float(item.length_m) / 1000.0, 5),
                    }
                    for item in forward.itertuples(index=False)
                ],
            }
        )
    return curves


def _plot_static(curves: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(7, 3, figsize=(18, 27), constrained_layout=True)
    for axis, curve in zip(axes.flat, curves, strict=True):
        years = np.array([point["year"] for point in curve["points"]], dtype=float)
        lengths = np.array(
            [point["length_km"] for point in curve["points"]], dtype=float
        )
        lia = curve["lia_target_km"]
        rgi = curve["rgi_target_km"]
        tol = curve["tolerance_km"]
        axis.axhspan(lia - tol, lia + tol, color="#d97706", alpha=0.09)
        axis.axhspan(rgi - tol, rgi + tol, color="#15803d", alpha=0.09)
        axis.axhline(lia, color="#d97706", linewidth=1.1, linestyle="--")
        axis.axhline(rgi, color="#15803d", linewidth=1.1, linestyle="--")
        axis.plot(years, lengths, color="#2563eb", linewidth=1.8)
        axis.scatter(
            [curve["forward_peak_year"], 2000],
            [curve["forward_peak_km"], curve["forward_2000_km"]],
            color="#2563eb",
            s=18,
            zorder=5,
        )
        values = np.r_[lengths, lia - tol, lia + tol, rgi - tol, rgi + tol]
        pad = max(float(np.ptp(values)) * 0.08, 0.15)
        axis.set_ylim(max(0.0, float(values.min()) - pad), float(values.max()) + pad)
        axis.set_xlim(float(years.min()), float(years.max()))
        axis.set_xticks([1500, 1700, 1900, 2000])
        axis.grid(alpha=0.18, linewidth=0.6)
        axis.set_title(
            f"System {curve['lia_unique_id']}  |  "
            f"ΔT {float(curve['spinup_delta_t_k']):+.3f}→"
            f"{float(curve['forward_delta_t_k']):+.3f} K\n"
            f"errors {curve['e_lia_sep']:.3f} / {curve['e_rgi_sep']:.3f}",
            fontsize=10,
        )
        axis.set_xlabel("Year")
        axis.set_ylabel("Length (km)")
    fig.suptitle(
        "Core-191: 21 formally frozen 1470–2000 glacier-length curves\n"
        "blue = simulation; orange = LIA target; green = RGI target; shading = ±20% separation gate",
        fontsize=16,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_metrics(curves: list[dict], output: Path) -> None:
    rows = [
        {key: value for key, value in curve.items() if key != "points"}
        for curve in curves
    ]
    pd.DataFrame(rows).to_csv(output, index=False)


def _write_fragment(curves: list[dict], output: Path) -> None:
    data = json.dumps(curves, ensure_ascii=False, separators=(",", ":"))
    fragment = r'''<div id="frozen-curves-21">
  <h2>21 条正式通过冰川：1470–2000 长度曲线</h2>
  <div class="viz-row frozen-legend" aria-label="图例与显示控制">
    <button type="button" data-series="curve" aria-pressed="true"><span class="swatch curve"></span>模拟长度</button>
    <button type="button" data-series="lia" aria-pressed="true"><span class="swatch lia"></span>LIA 目标</button>
    <button type="button" data-series="rgi" aria-pressed="true"><span class="swatch rgi"></span>RGI 目标</button>
  </div>
  <div class="curve-grid" aria-live="polite"></div>
  <div class="tooltip" role="tooltip" hidden></div>
  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
  <script>
  (() => {
    const root = document.getElementById('frozen-curves-21');
    const data = __DATA__;
    const state = {curve: true, lia: true, rgi: true};
    const grid = root.querySelector('.curve-grid');
    const tooltip = root.querySelector('.tooltip');

    function drawPanel(container, glacier) {
      container.replaceChildren();
      const width = Math.max(300, container.getBoundingClientRect().width || 300);
      const height = 224;
      const margin = {top: 12, right: 14, bottom: 42, left: 58};
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;
      const values = glacier.points.map(d => d.length_km)
        .concat([
          glacier.lia_target_km - glacier.tolerance_km,
          glacier.lia_target_km + glacier.tolerance_km,
          glacier.rgi_target_km - glacier.tolerance_km,
          glacier.rgi_target_km + glacier.tolerance_km
        ]);
      const xExtent = d3.extent(glacier.points, d => d.year);
      const yExtent = d3.extent(values);
      const yPad = Math.max((yExtent[1] - yExtent[0]) * 0.08, 0.12);
      const x = d3.scaleLinear().domain(xExtent).range([0, innerW]);
      const y = d3.scaleLinear().domain([Math.max(0, yExtent[0] - yPad), yExtent[1] + yPad]).nice().range([innerH, 0]);
      const svg = d3.select(container).append('svg')
        .attr('viewBox', `0 0 ${width} ${height}`)
        .attr('role', 'img')
        .attr('aria-label', `系统 ${glacier.lia_unique_id} 的 1470 到 2000 年长度曲线`);
      svg.append('title').text(`系统 ${glacier.lia_unique_id} 冰川长度曲线`);
      svg.append('desc').text(`LIA separation 误差 ${glacier.e_lia_sep.toFixed(3)}，RGI separation 误差 ${glacier.e_rgi_sep.toFixed(3)}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      g.append('rect').attr('data-chart-frame', '').attr('width', innerW).attr('height', innerH).attr('fill', 'none').attr('stroke', 'var(--border)');

      function targetBand(kind, value) {
        g.append('rect').attr('class', `series-${kind}`).attr('x', 0).attr('width', innerW)
          .attr('y', y(value + glacier.tolerance_km))
          .attr('height', Math.abs(y(value - glacier.tolerance_km) - y(value + glacier.tolerance_km)))
          .attr('fill', `var(--viz-series-${kind === 'lia' ? 2 : 3})`).attr('opacity', 0.08);
        g.append('line').attr('class', `series-${kind}`).attr('x1', 0).attr('x2', innerW)
          .attr('y1', y(value)).attr('y2', y(value))
          .attr('stroke', `var(--viz-series-${kind === 'lia' ? 2 : 3})`).attr('stroke-width', 1.5).attr('stroke-dasharray', '5 4');
      }
      targetBand('lia', glacier.lia_target_km);
      targetBand('rgi', glacier.rgi_target_km);

      const line = d3.line().x(d => x(d.year)).y(d => y(d.length_km));
      g.append('path').datum(glacier.points).attr('class', 'series-curve')
        .attr('fill', 'none').attr('stroke', 'var(--viz-series-1)').attr('stroke-width', 2)
        .attr('d', line);
      g.append('circle').attr('class', 'series-curve').attr('cx', x(glacier.forward_peak_year)).attr('cy', y(glacier.forward_peak_km)).attr('r', 3).attr('fill', 'var(--viz-series-1)');
      g.append('circle').attr('class', 'series-curve').attr('cx', x(2000)).attr('cy', y(glacier.forward_2000_km)).attr('r', 3).attr('fill', 'var(--viz-series-1)');

      const xTicks = width < 360 ? 3 : 4;
      g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(xTicks).tickFormat(d3.format('d')))
        .call(group => group.selectAll('text').attr('fill', 'var(--foreground)').attr('font-size', 12))
        .call(group => group.selectAll('path,line').attr('stroke', 'var(--border)'));
      g.append('g').call(d3.axisLeft(y).ticks(4))
        .call(group => group.selectAll('text').attr('fill', 'var(--foreground)').attr('font-size', 12))
        .call(group => group.selectAll('path,line').attr('stroke', 'var(--border)'));
      g.append('text').attr('class', 'axis-title').attr('data-axis', 'x').attr('x', innerW / 2).attr('y', innerH + 36).attr('text-anchor', 'middle').attr('fill', 'var(--foreground)').attr('font-size', 12).text('年份');
      g.append('text').attr('class', 'axis-title').attr('data-axis', 'y').attr('transform', 'rotate(-90)').attr('x', -innerH / 2).attr('y', -44).attr('text-anchor', 'middle').attr('fill', 'var(--foreground)').attr('font-size', 12).text('长度 (km)');

      const guide = g.append('line').attr('data-chart-hover-guide', '').attr('y1', 0).attr('y2', innerH).attr('stroke', 'var(--muted-foreground)').attr('stroke-width', 1).attr('opacity', 0);
      const marker = g.append('circle').attr('data-chart-hover-marker', '').attr('r', 4).attr('fill', 'var(--viz-series-1)').attr('opacity', 0);
      g.append('rect').attr('data-chart-hit', '').attr('data-chart-hover-overlay', 'cross-series')
        .attr('width', innerW).attr('height', innerH).attr('fill', 'transparent')
        .on('pointermove', event => {
          const [px] = d3.pointer(event);
          const year = x.invert(px);
          const index = d3.bisector(d => d.year).center(glacier.points, year);
          const point = glacier.points[index];
          guide.attr('x1', px).attr('x2', px).attr('opacity', 1);
          marker.attr('cx', x(point.year)).attr('cy', y(point.length_km)).attr('opacity', state.curve ? 1 : 0);
          const rect = root.getBoundingClientRect();
          tooltip.hidden = false;
          tooltip.style.left = `${event.clientX - rect.left + 12}px`;
          tooltip.style.top = `${event.clientY - rect.top + 12}px`;
          tooltip.innerHTML = `<strong>系统 ${glacier.lia_unique_id}</strong><br>${point.year}: ${point.length_km.toFixed(2)} km<br>LIA: ${glacier.lia_target_km.toFixed(2)} km<br>RGI: ${glacier.rgi_target_km.toFixed(2)} km`;
        })
        .on('pointerleave', () => { guide.attr('opacity', 0); marker.attr('opacity', 0); tooltip.hidden = true; });

      Object.entries(state).forEach(([key, visible]) => {
        g.selectAll(`.series-${key}`).style('display', visible ? null : 'none');
      });
    }

    data.forEach(glacier => {
      const panel = document.createElement('section');
      panel.className = 'curve-panel';
      panel.innerHTML = `<h3>系统 ${glacier.lia_unique_id}</h3><div class="panel-meta">ΔT ${Number(glacier.spinup_delta_t_k).toFixed(3)} → ${Number(glacier.forward_delta_t_k).toFixed(3)} K · 误差 ${glacier.e_lia_sep.toFixed(3)} / ${glacier.e_rgi_sep.toFixed(3)}</div><div class="plot-host"></div>`;
      grid.appendChild(panel);
      const host = panel.querySelector('.plot-host');
      const redraw = () => drawPanel(host, glacier);
      redraw();
      new ResizeObserver(redraw).observe(host);
    });

    root.querySelectorAll('button[data-series]').forEach(button => {
      button.addEventListener('click', () => {
        const key = button.dataset.series;
        state[key] = !state[key];
        button.setAttribute('aria-pressed', String(state[key]));
        root.querySelectorAll(`.series-${key}`).forEach(node => {
          node.style.display = state[key] ? '' : 'none';
        });
      });
    });
  })();
  </script>
  <style>
    #frozen-curves-21 { position: relative; width: 100%; color: var(--foreground); }
    #frozen-curves-21 h2 { margin: 0 0 10px; font-weight: 500; }
    #frozen-curves-21 .frozen-legend { margin-bottom: 14px; gap: 14px; }
    #frozen-curves-21 .frozen-legend button { appearance: none; border: 0; background: transparent; color: var(--foreground); padding: 3px 0; cursor: pointer; }
    #frozen-curves-21 .frozen-legend button[aria-pressed="false"] { opacity: 0.42; }
    #frozen-curves-21 .swatch { display: inline-block; width: 20px; height: 3px; margin-right: 6px; vertical-align: middle; }
    #frozen-curves-21 .swatch.curve { background: var(--viz-series-1); }
    #frozen-curves-21 .swatch.lia { background: var(--viz-series-2); }
    #frozen-curves-21 .swatch.rgi { background: var(--viz-series-3); }
    #frozen-curves-21 .curve-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px 16px; }
    #frozen-curves-21 .curve-panel { min-width: 0; }
    #frozen-curves-21 .curve-panel h3 { margin: 0; font-weight: 500; }
    #frozen-curves-21 .panel-meta { color: var(--muted-foreground); margin: 2px 0 5px; font-size: 12px; }
    #frozen-curves-21 .plot-host { width: 100%; min-width: 0; }
    #frozen-curves-21 .tooltip { position: absolute; pointer-events: none; z-index: 10; padding: 7px 9px; background: var(--popover); color: var(--popover-foreground); border: 1px solid var(--border); font-size: 12px; }
    @media (max-width: 760px) { #frozen-curves-21 .curve-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 480px) { #frozen-curves-21 .curve-grid { grid-template-columns: 1fr; } }
  </style>
</div>'''.replace("__DATA__", data)
    output.write_text(fragment, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=root
        / "validation_data/results/paleo_surface_validation_v0/"
        "lia_191_frozen_curve_registry_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / "validation_data/results/paleo_surface_validation_v0/"
        "lia_191_frozen_curve_registry_v1/curve_gallery_v1",
    )
    parser.add_argument("--visual-html", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.visual_html.parent.mkdir(parents=True, exist_ok=True)
    curves = _load_registry(args.registry_root)
    static_path = args.output_dir / "frozen_21_length_curves_grid.png"
    _plot_static(curves, static_path)
    _write_metrics(curves, args.output_dir / "frozen_21_curve_metrics.csv")
    _write_fragment(curves, args.visual_html)
    summary = {
        "schema_version": 1,
        "launched_oggm": False,
        "frozen_system_count": len(curves),
        "systems": [curve["lia_unique_id"] for curve in curves],
        "static_plot": str(static_path),
        "metrics_csv": str(args.output_dir / "frozen_21_curve_metrics.csv"),
        "inline_visual": str(args.visual_html),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
