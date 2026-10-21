"""Plot the 35 currently accepted Core-191 glacier-length histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NEW_ACCEPTED_ROOTS = {
    3850: "w190hist3850rampendp05reprov1",
    2546: "w205hist2546rampendp0375reprov1",
    2806: "w220hist2806power_g00625_endm045reprov1",
    2574: "w232hist2574rampendm075reprov1",
    5265: "w234hist5265rampendp000reprov1",
    2226: "w237hist2226powerg050endm025reprov1",
    2259: "w252hist2259rampendp000reprov1",
    313: "w281hist313rampendm05625reprov1",
    5288: "w285hist5288rampendp0375reprov1",
    904: "w303hist904rampendm05625reprov1",
    5142: "w303hist5142rampendp0875reprov1",
    512: "w318hist512rampendp053125reprov2",
    4378: "w352hist4378rampendp0359375reprov1",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _points(frame: pd.DataFrame, length_col: str) -> list[dict]:
    out = frame.copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["length_m"] = pd.to_numeric(out[length_col], errors="coerce")
    out = out.dropna(subset=["year", "length_m"])
    out = out.loc[out["year"].between(1500, 2000)].sort_values("year")
    if out.empty:
        raise RuntimeError("No 1500-2000 length samples")
    return [
        {"year": int(round(float(r.year))), "length_km": round(float(r.length_m) / 1000, 5)}
        for r in out.itertuples(index=False)
    ]


def _targets(lia_m: float, rgi_m: float) -> tuple[float, float]:
    separation = abs(lia_m - rgi_m)
    return min(0.2 * separation, 0.1 * lia_m), min(0.2 * separation, 0.1 * rgi_m)


def load_curves(root: Path) -> list[dict]:
    registry_root = root / "validation_data/results/paleo_surface_validation_v0/lia_191_frozen_curve_registry_v1"
    registry = pd.read_csv(registry_root / "frozen_candidates.csv")
    if len(registry) != 22 or registry["lia_unique_id"].nunique() != 22:
        raise RuntimeError("Frozen registry is not the canonical 22-system set")
    curves: list[dict] = []
    for row in registry.sort_values("lia_unique_id").itertuples(index=False):
        uid = int(row.lia_unique_id)
        system_root = registry_root / f"system_{uid}"
        record = json.loads((system_root / "candidate_record.json").read_text(encoding="utf-8"))
        series = pd.read_csv(system_root / "timeseries.csv")
        series = series.loc[series["phase"].astype(str).eq("forward")]
        length_col = "observation_length_m" if "observation_length_m" in series else "length_m"
        scores = record["endpoint_scores"]
        lia_m = float(scores["lia_target_m"])
        rgi_m = float(scores["rgi_target_m"])
        lia_tol_m, rgi_tol_m = _targets(lia_m, rgi_m)
        curves.append({
            "system_id": uid,
            "rgi_id": str(record["rgi_id"]),
            "route": "frozen-legacy",
            "lia_km": lia_m / 1000,
            "rgi_km": rgi_m / 1000,
            "lia_tol_km": lia_tol_m / 1000,
            "rgi_tol_km": rgi_tol_m / 1000,
            "e_lia": float(scores["e_lia_sep"]),
            "e_rgi": float(scores["e_rgi_sep"]),
            "points": _points(series, length_col),
        })

    refs = pd.read_csv(root / "validation_data/results/paleo_surface_validation_v0/lia_191_controller_v1/core191_length_references.csv")
    refs["lia_unique_id"] = pd.to_numeric(refs["lia_unique_id"]).astype(int)
    refs = refs.set_index("lia_unique_id")
    for uid, rel_root in NEW_ACCEPTED_ROOTS.items():
        run_root = root / rel_root
        summary = json.loads((run_root / "historical_summary.json").read_text(encoding="utf-8"))
        if summary["status"] != "historical-pass" or int(summary["system_id"]) != uid:
            raise RuntimeError(f"System {uid} is not a historical pass at {run_root}")
        series = pd.read_csv(run_root / "historical_1500_2000.csv")
        ref = refs.loc[uid]
        lia_m = float(ref["lia_registered_length_m"])
        rgi_m = float(ref["rgi_registered_length_m"])
        lia_tol_m, rgi_tol_m = _targets(lia_m, rgi_m)
        curves.append({
            "system_id": uid,
            "rgi_id": str(ref["rgi_id"]),
            "route": "synthetic-system",
            "lia_km": lia_m / 1000,
            "rgi_km": rgi_m / 1000,
            "lia_tol_km": lia_tol_m / 1000,
            "rgi_tol_km": rgi_tol_m / 1000,
            "e_lia": float(summary["e_lia_separation"]),
            "e_rgi": float(summary["e_rgi_separation"]),
            "points": _points(series, "length_m"),
        })

    curves.sort(key=lambda item: item["system_id"])
    ids = [item["system_id"] for item in curves]
    if len(curves) != 35 or len(set(ids)) != 35:
        raise RuntimeError(f"Expected 35 unique accepted curves, found {len(curves)}/{len(set(ids))}")
    return curves


def plot_static(curves: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(7, 5, figsize=(23, 27), constrained_layout=True)
    for ax, curve in zip(axes.flat, curves, strict=True):
        years = np.array([p["year"] for p in curve["points"]])
        lengths = np.array([p["length_km"] for p in curve["points"]])
        lia, rgi = curve["lia_km"], curve["rgi_km"]
        lt, rt = curve["lia_tol_km"], curve["rgi_tol_km"]
        ax.axhspan(lia - lt, lia + lt, color="#d97706", alpha=0.08)
        ax.axhspan(rgi - rt, rgi + rt, color="#15803d", alpha=0.08)
        ax.axhline(lia, color="#d97706", linewidth=1.0, linestyle="--")
        ax.axhline(rgi, color="#15803d", linewidth=1.0, linestyle="--")
        ax.plot(years, lengths, color="#2563eb", linewidth=1.55)
        vals = np.r_[lengths, lia - lt, lia + lt, rgi - rt, rgi + rt]
        pad = max(float(np.ptp(vals)) * 0.08, 0.10)
        ax.set_ylim(max(0, float(vals.min()) - pad), float(vals.max()) + pad)
        ax.set_xlim(1500, 2000)
        ax.set_xticks([1500, 1700, 1900, 2000])
        ax.grid(alpha=0.18, linewidth=0.5)
        route = "S" if curve["route"] == "synthetic-system" else "F"
        ax.set_title(f"{curve['system_id']}  [{route}]   e={curve['e_lia']:.2f}/{curve['e_rgi']:.2f}", fontsize=9)
        ax.set_xlabel("Year", fontsize=8)
        ax.set_ylabel("Length (km)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(
        "Core-191: 35 accepted glacier-length histories (1500–2000)\n"
        "blue: simulation   orange: LIA target   green: RGI target   bands: formal acceptance intervals",
        fontsize=15,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_fragment(curves: list[dict], path: Path) -> None:
    data = json.dumps(curves, ensure_ascii=False, separators=(",", ":"))
    html = r'''<div id="lia-accepted-35">
  <h2>35 条已接受冰川长度曲线（1500–2000）</h2>
  <div class="viz-row legend" aria-label="图例">
    <span><i class="line sim"></i>模拟长度</span>
    <span><i class="line lia"></i>LIA 目标</span>
    <span><i class="line rgi"></i>RGI 目标</span>
    <span>F＝旧冻结链，S＝新 synthetic-system 链</span>
  </div>
  <div class="plots"></div>
  <div class="tooltip" role="tooltip" hidden></div>
  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
  <script>
  (() => {
    const root = document.getElementById('lia-accepted-35');
    const data = __DATA__;
    const grid = root.querySelector('.plots');
    const tooltip = root.querySelector('.tooltip');
    function draw(host, d) {
      host.replaceChildren();
      const width = Math.max(280, host.getBoundingClientRect().width || 280);
      const height = 210;
      const m = {top: 8, right: 10, bottom: 37, left: 55};
      const iw = width - m.left - m.right, ih = height - m.top - m.bottom;
      const ys = d.points.map(p => p.length_km).concat([d.lia_km-d.lia_tol_km,d.lia_km+d.lia_tol_km,d.rgi_km-d.rgi_tol_km,d.rgi_km+d.rgi_tol_km]);
      const ext = d3.extent(ys), pad = Math.max((ext[1]-ext[0])*0.08, .1);
      const x = d3.scaleLinear().domain([1500,2000]).range([0,iw]);
      const y = d3.scaleLinear().domain([Math.max(0,ext[0]-pad),ext[1]+pad]).nice().range([ih,0]);
      const svg = d3.select(host).append('svg').attr('viewBox',`0 0 ${width} ${height}`).attr('role','img').attr('aria-label',`系统 ${d.system_id} 1500 到 2000 年冰川长度曲线`);
      svg.append('title').text(`系统 ${d.system_id} 冰川长度曲线`);
      svg.append('desc').text(`LIA 误差 ${d.e_lia.toFixed(3)}，RGI 误差 ${d.e_rgi.toFixed(3)}`);
      const g=svg.append('g').attr('transform',`translate(${m.left},${m.top})`);
      g.append('rect').attr('data-chart-frame','').attr('width',iw).attr('height',ih).attr('fill','none').attr('stroke','var(--border)');
      function target(kind,value,tol,color){
        g.append('rect').attr('x',0).attr('width',iw).attr('y',y(value+tol)).attr('height',Math.abs(y(value-tol)-y(value+tol))).attr('fill',color).attr('opacity',.08);
        g.append('line').attr('x1',0).attr('x2',iw).attr('y1',y(value)).attr('y2',y(value)).attr('stroke',color).attr('stroke-width',1.3).attr('stroke-dasharray','5 4');
      }
      target('lia',d.lia_km,d.lia_tol_km,'var(--viz-series-2)');
      target('rgi',d.rgi_km,d.rgi_tol_km,'var(--viz-series-3)');
      const line=d3.line().x(p=>x(p.year)).y(p=>y(p.length_km));
      g.append('path').datum(d.points).attr('fill','none').attr('stroke','var(--viz-series-1)').attr('stroke-width',1.8).attr('d',line);
      const xa=g.append('g').attr('transform',`translate(0,${ih})`).call(d3.axisBottom(x).tickValues(width<340?[1500,1750,2000]:[1500,1700,1900,2000]).tickFormat(d3.format('d')));
      const ya=g.append('g').call(d3.axisLeft(y).ticks(4));
      [xa,ya].forEach(a=>{a.selectAll('text').attr('fill','var(--foreground)').attr('font-size',11);a.selectAll('path,line').attr('stroke','var(--border)')});
      g.append('text').attr('class','axis-title').attr('data-axis','x').attr('x',iw/2).attr('y',ih+32).attr('text-anchor','middle').attr('fill','var(--foreground)').attr('font-size',12).text('年份');
      g.append('text').attr('class','axis-title').attr('data-axis','y').attr('transform','rotate(-90)').attr('x',-ih/2).attr('y',-42).attr('text-anchor','middle').attr('fill','var(--foreground)').attr('font-size',12).text('长度 (km)');
      const guide=g.append('line').attr('data-chart-hover-guide','').attr('y1',0).attr('y2',ih).attr('stroke','var(--muted-foreground)').attr('opacity',0);
      const mark=g.append('circle').attr('data-chart-hover-marker','').attr('r',3.5).attr('fill','var(--viz-series-1)').attr('opacity',0);
      g.append('rect').attr('data-chart-hit','').attr('data-chart-hover-overlay','cross-series').attr('width',iw).attr('height',ih).attr('fill','transparent')
        .on('pointermove',event=>{const [px]=d3.pointer(event);const yr=x.invert(px);const i=d3.bisector(p=>p.year).center(d.points,yr);const p=d.points[i];guide.attr('x1',px).attr('x2',px).attr('opacity',1);mark.attr('cx',x(p.year)).attr('cy',y(p.length_km)).attr('opacity',1);const rr=root.getBoundingClientRect();tooltip.hidden=false;tooltip.style.left=`${event.clientX-rr.left+12}px`;tooltip.style.top=`${event.clientY-rr.top+12}px`;tooltip.innerHTML=`<strong>系统 ${d.system_id}</strong><br>${p.year}: ${p.length_km.toFixed(2)} km<br>LIA: ${d.lia_km.toFixed(2)} km<br>RGI: ${d.rgi_km.toFixed(2)} km`;})
        .on('pointerleave',()=>{guide.attr('opacity',0);mark.attr('opacity',0);tooltip.hidden=true;});
    }
    data.forEach(d=>{const panel=document.createElement('section');panel.className='panel';const tag=d.route==='synthetic-system'?'S':'F';panel.innerHTML=`<h3>系统 ${d.system_id} [${tag}]</h3><div class="meta">separation 误差 ${d.e_lia.toFixed(3)} / ${d.e_rgi.toFixed(3)}</div><div class="host"></div>`;grid.appendChild(panel);const host=panel.querySelector('.host');draw(host,d);new ResizeObserver(()=>draw(host,d)).observe(host);});
  })();
  </script>
  <style>
    #lia-accepted-35{position:relative;width:100%;color:var(--foreground)}
    #lia-accepted-35 h2{margin:0 0 8px;font-weight:500}
    #lia-accepted-35 .legend{gap:14px;margin-bottom:14px;color:var(--muted-foreground);font-size:12px}
    #lia-accepted-35 .line{display:inline-block;width:20px;height:3px;margin-right:5px;vertical-align:middle}
    #lia-accepted-35 .line.sim{background:var(--viz-series-1)}
    #lia-accepted-35 .line.lia{background:var(--viz-series-2)}
    #lia-accepted-35 .line.rgi{background:var(--viz-series-3)}
    #lia-accepted-35 .plots{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px 14px}
    #lia-accepted-35 .panel{min-width:0}
    #lia-accepted-35 .panel h3{margin:0;font-weight:500}
    #lia-accepted-35 .meta{margin:2px 0 4px;color:var(--muted-foreground);font-size:12px}
    #lia-accepted-35 .host{width:100%;min-width:0}
    #lia-accepted-35 .tooltip{position:absolute;pointer-events:none;z-index:10;padding:7px 9px;background:var(--popover);color:var(--popover-foreground);border:1px solid var(--border);font-size:12px}
    @media(max-width:760px){#lia-accepted-35 .plots{grid-template-columns:repeat(2,minmax(0,1fr))}}
    @media(max-width:480px){#lia-accepted-35 .plots{grid-template-columns:1fr}}
  </style>
</div>'''.replace("__DATA__", data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-html", type=Path, required=True)
    args = parser.parse_args()
    root = project_root()
    output = root / "validation_data/results/paleo_surface_validation_v0/lia_191_accepted_35_curve_gallery_v1"
    output.mkdir(parents=True, exist_ok=True)
    curves = load_curves(root)
    png = output / "accepted_35_length_curves_grid.png"
    plot_static(curves, png)
    rows = [{k: v for k, v in c.items() if k != "points"} for c in curves]
    pd.DataFrame(rows).to_csv(output / "accepted_35_curve_metrics.csv", index=False)
    write_fragment(curves, args.visual_html)
    summary = {"accepted_count": len(curves), "systems": [c["system_id"] for c in curves], "static_plot": str(png), "inline_visual": str(args.visual_html), "launched_oggm": False}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
