"""Build the in-conversation visualization for all formally accepted LIA curves."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "validation_data/results/paleo_surface_validation_v0"
METRICS = RESULTS / "lia_191_accepted_54_registry_v1/accepted_54_curve_metrics.csv"
INDEX = RESULTS / "lia_191_accepted54_evidence_index_v3/accepted54_evidence_index.csv"
FROZEN = RESULTS / "lia_191_frozen_curve_registry_v1"
OUTPUT = Path(
    r"C:\Users\JUNZ\.codex\visualizations\2026\08\16\01a008a2-6e98-73b1-8098-7f4262d2a7d2\lia-accepted-curves.html"
)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def curve_rows(system_id, route, reproduction_root):
    if route == "frozen-legacy":
        source = FROZEN / f"system_{system_id}" / "timeseries.csv"
        rows = [r for r in read_csv(source) if r.get("phase") == "forward"]
        length_key = "observation_length_m"
    else:
        source = ROOT / reproduction_root / "historical_1470_2000.csv"
        rows = read_csv(source)
        length_key = "length_m"
    points = {}
    for row in rows:
        year = int(round(float(row["year"])))
        if 1470 <= year <= 2000:
            value = row.get(length_key) or row.get("length_m")
            points[year] = round(float(value) / 1000.0, 4)
    if 1470 not in points or 2000 not in points:
        raise RuntimeError(f"system {system_id} curve lacks 1470 or 2000: {source}")
    return [[year, points[year]] for year in sorted(points)], source


def main():
    metrics = {int(r["system_id"]): r for r in read_csv(METRICS)}
    index = {int(r["system_id"]): r for r in read_csv(INDEX)}
    if len(metrics) != 54 or len(index) != 54 or set(metrics) != set(index):
        raise RuntimeError("authoritative accepted registry/index is not exactly 54 matching systems")

    data = []
    for sid in sorted(metrics):
        m = metrics[sid]
        ev = index[sid]
        if str(ev["accepted_evidence_pass"]).lower() != "true":
            raise RuntimeError(f"system {sid} evidence index is not pass")
        points, source = curve_rows(sid, m["route"], ev["reproduction_root"])
        peak = max(points, key=lambda p: (p[1], -p[0]))
        data.append(
            {
                "id": sid,
                "rgi": m["rgi_id"],
                "route": m["route"],
                "lia": round(float(m["lia_km"]), 4),
                "rgiTarget": round(float(m["rgi_km"]), 4),
                "eLia": round(float(m["e_lia"]), 4),
                "eRgi": round(float(m["e_rgi"]), 4),
                "peakYear": peak[0],
                "peak": peak[1],
                "final": points[-1][1],
                "points": points,
                "source": str(source),
            }
        )

    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    fragment = f'''<div id="lia-accepted-curves-v1" class="lia-curves-root">
  <h2>1500–2000 LIA accepted length curves — 54 formally accepted</h2>
  <div class="lia-curves-controls">
    <label for="lia-curve-select">Highlight system</label>
    <select id="lia-curve-select" aria-label="Highlight accepted glacier system"></select>
  </div>
  <div id="lia-selected-summary" class="lia-selected-summary" aria-live="polite"></div>
  <section aria-labelledby="lia-overview-title">
    <h3 id="lia-overview-title">All accepted trajectories, normalized by each LIA target</h3>
    <div id="lia-overview-chart" class="lia-chart"></div>
  </section>
  <section aria-labelledby="lia-detail-title">
    <h3 id="lia-detail-title">Selected trajectory and registered targets</h3>
    <div id="lia-detail-chart" class="lia-chart"></div>
  </section>
  <div id="lia-curves-tooltip" class="tooltip" role="tooltip" aria-hidden="true"></div>
</div>
<style>
#lia-accepted-curves-v1 {{ color: var(--foreground); width: 100%; }}
#lia-accepted-curves-v1 h2, #lia-accepted-curves-v1 h3 {{ font-weight: 500; margin: 0 0 8px; }}
#lia-accepted-curves-v1 section {{ margin-top: 20px; }}
#lia-accepted-curves-v1 .lia-curves-controls {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin: 10px 0 8px; }}
#lia-accepted-curves-v1 select {{ min-width: 250px; max-width: 100%; color: var(--foreground); background: var(--background); border: 1px solid var(--border); padding: 7px 9px; }}
#lia-accepted-curves-v1 .lia-selected-summary {{ min-height: 20px; color: var(--muted-foreground); margin-bottom: 4px; }}
#lia-accepted-curves-v1 .lia-chart {{ width: 100%; min-height: 280px; }}
#lia-accepted-curves-v1 svg {{ display: block; width: 100%; overflow: visible; }}
#lia-accepted-curves-v1 svg text {{ fill: var(--foreground); font-size: 12px; }}
#lia-accepted-curves-v1 .grid line {{ stroke: var(--border); stroke-opacity: .55; }}
#lia-accepted-curves-v1 .grid path {{ display: none; }}
#lia-accepted-curves-v1 .axis path, #lia-accepted-curves-v1 .axis line {{ stroke: var(--border); }}
#lia-accepted-curves-v1 rect[data-chart-frame] {{ fill: none; stroke: var(--border); }}
#lia-accepted-curves-v1 .ensemble {{ fill: none; stroke: var(--muted-foreground); stroke-opacity: .23; stroke-width: 1; }}
#lia-accepted-curves-v1 .selected-line {{ fill: none; stroke: var(--viz-series-1); stroke-width: 2.6; }}
#lia-accepted-curves-v1 .target-lia {{ fill: none; stroke: var(--viz-series-2); stroke-width: 1.6; stroke-dasharray: 6 4; }}
#lia-accepted-curves-v1 .target-rgi {{ fill: none; stroke: var(--viz-series-3); stroke-width: 1.6; stroke-dasharray: 2 4; }}
#lia-accepted-curves-v1 .target-label {{ font-weight: 500; }}
#lia-accepted-curves-v1 .tooltip {{ position: absolute; pointer-events: none; opacity: 0; color: var(--popover-foreground); background: var(--popover); border: 1px solid var(--border); padding: 7px 9px; z-index: 5; }}
#lia-accepted-curves-v1 .hover-guide {{ stroke: var(--foreground); stroke-opacity: .45; stroke-width: 1; }}
#lia-accepted-curves-v1 .hover-marker {{ fill: var(--viz-series-1); stroke: var(--background); stroke-width: 1.5; }}
@media (max-width: 480px) {{
  #lia-accepted-curves-v1 .lia-chart {{ min-height: 250px; }}
  #lia-accepted-curves-v1 select {{ width: 100%; min-width: 0; }}
}}
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {{
  const root = document.getElementById('lia-accepted-curves-v1');
  const curves = {payload};
  const select = root.querySelector('#lia-curve-select');
  const summary = root.querySelector('#lia-selected-summary');
  const tooltip = root.querySelector('#lia-curves-tooltip');
  curves.forEach(d => {{
    const option = document.createElement('option');
    option.value = d.id;
    option.textContent = `${{d.id}} · ${{d.rgi}}`;
    select.appendChild(option);
  }});
  let selected = curves[0];

  function interpolate(points, year) {{
    const bisect = d3.bisector(d => d[0]).center;
    const i = bisect(points, year);
    if (i <= 0) return points[0][1];
    if (i >= points.length - 1) return points[points.length - 1][1];
    const a = points[i], b = points[i + 1];
    if (a[0] === year || a[0] === b[0]) return a[1];
    const t = (year - a[0]) / (b[0] - a[0]);
    return a[1] + t * (b[1] - a[1]);
  }}

  function chartBase(container, yDomain, yLabel) {{
    container.innerHTML = '';
    const width = Math.max(320, container.getBoundingClientRect().width || 736);
    const height = width < 480 ? 250 : 300;
    const margin = {{top: 14, right: 28, bottom: 50, left: 68}};
    const svg = d3.select(container).append('svg').attr('viewBox', `0 0 ${{width}} ${{height}}`).attr('aria-label', yLabel);
    const x = d3.scaleLinear().domain([1470, 2000]).range([margin.left, width - margin.right]);
    const y = d3.scaleLinear().domain(yDomain).nice().range([height - margin.bottom, margin.top]);
    svg.append('rect').attr('data-chart-frame', '').attr('x', margin.left).attr('y', margin.top).attr('width', width-margin.left-margin.right).attr('height', height-margin.top-margin.bottom);
    svg.append('g').attr('class','grid').attr('transform',`translate(${{margin.left}},0)`).call(d3.axisLeft(y).tickSize(-(width-margin.left-margin.right)).tickFormat(''));
    svg.append('g').attr('class','axis').attr('transform',`translate(0,${{height-margin.bottom}})`).call(d3.axisBottom(x).ticks(width < 480 ? 4 : 7).tickFormat(d3.format('d')));
    svg.append('g').attr('class','axis').attr('transform',`translate(${{margin.left}},0)`).call(d3.axisLeft(y).ticks(6));
    svg.append('text').attr('class','axis-title').attr('data-axis','x').attr('x',(margin.left+width-margin.right)/2).attr('y',height-10).attr('text-anchor','middle').text('Year');
    svg.append('text').attr('class','axis-title').attr('data-axis','y').attr('transform','rotate(-90)').attr('x',-(margin.top+height-margin.bottom)/2).attr('y',16).attr('text-anchor','middle').text(yLabel);
    return {{svg,x,y,width,height,margin}};
  }}

  function drawOverview() {{
    const container = root.querySelector('#lia-overview-chart');
    const values = curves.flatMap(d => d.points.map(p => p[1] / d.lia));
    let ext = d3.extent(values.concat([1]));
    const pad = Math.max(.03, (ext[1]-ext[0])*.06);
    const c = chartBase(container, [ext[0]-pad, ext[1]+pad], 'Length / LIA target');
    const line = d3.line().x(p => c.x(p[0])).y(p => c.y(p[1]));
    c.svg.append('g').selectAll('path').data(curves).join('path').attr('class','ensemble').attr('d',d => line(d.points.map(p => [p[0],p[1]/d.lia])));
    c.svg.append('line').attr('class','target-lia').attr('x1',c.x(1470)).attr('x2',c.x(2000)).attr('y1',c.y(1)).attr('y2',c.y(1));
    c.svg.append('path').datum(selected.points.map(p => [p[0],p[1]/selected.lia])).attr('class','selected-line').attr('d',line);
  }}

  function drawDetail() {{
    const container = root.querySelector('#lia-detail-chart');
    const vals = selected.points.map(p => p[1]).concat([selected.lia, selected.rgiTarget]);
    let ext = d3.extent(vals);
    const pad = Math.max(.35, (ext[1]-ext[0])*.10);
    const c = chartBase(container, [ext[0]-pad, ext[1]+pad], 'Length (km)');
    const line = d3.line().x(p => c.x(p[0])).y(p => c.y(p[1]));
    c.svg.append('path').datum(selected.points).attr('class','selected-line').attr('d',line);
    c.svg.append('line').attr('class','target-lia').attr('x1',c.x(1470)).attr('x2',c.x(2000)).attr('y1',c.y(selected.lia)).attr('y2',c.y(selected.lia));
    c.svg.append('line').attr('class','target-rgi').attr('x1',c.x(1470)).attr('x2',c.x(2000)).attr('y1',c.y(selected.rgiTarget)).attr('y2',c.y(selected.rgiTarget));
    c.svg.append('text').attr('class','target-label').attr('x',c.x(1475)).attr('y',c.y(selected.lia)-5).text(`LIA target ${{selected.lia.toFixed(2)}} km`);
    c.svg.append('text').attr('class','target-label').attr('x',c.x(1475)).attr('y',c.y(selected.rgiTarget)+15).text(`RGI target ${{selected.rgiTarget.toFixed(2)}} km`);
    const guide = c.svg.append('line').attr('class','hover-guide').attr('data-chart-hover-guide','').style('display','none');
    const marker = c.svg.append('circle').attr('class','hover-marker').attr('data-chart-hover-marker','').attr('r',4).style('display','none');
    c.svg.append('rect').attr('data-chart-hit','').attr('data-chart-hover-overlay','cross-series').attr('x',c.margin.left).attr('y',c.margin.top).attr('width',c.width-c.margin.left-c.margin.right).attr('height',c.height-c.margin.top-c.margin.bottom).attr('fill','transparent')
      .on('pointermove', event => {{
        const [mx] = d3.pointer(event);
        const year = Math.max(1470, Math.min(2000, c.x.invert(mx)));
        const value = interpolate(selected.points, year);
        guide.style('display',null).attr('x1',c.x(year)).attr('x2',c.x(year)).attr('y1',c.margin.top).attr('y2',c.height-c.margin.bottom);
        marker.style('display',null).attr('cx',c.x(year)).attr('cy',c.y(value));
        tooltip.style.opacity=1; tooltip.setAttribute('aria-hidden','false');
        tooltip.innerHTML=`${{Math.round(year)}} · curve ${{value.toFixed(2)}} km<br>LIA target ${{selected.lia.toFixed(2)}} km<br>RGI target ${{selected.rgiTarget.toFixed(2)}} km`;
        const rb=root.getBoundingClientRect(); tooltip.style.left=`${{event.clientX-rb.left+12}}px`; tooltip.style.top=`${{event.clientY-rb.top-24}}px`;
      }})
      .on('pointerleave',()=>{{guide.style('display','none');marker.style('display','none');tooltip.style.opacity=0;tooltip.setAttribute('aria-hidden','true');}});
  }}

  function update() {{
    selected = curves.find(d => String(d.id) === select.value) || curves[0];
    summary.textContent = `System ${{selected.id}} · ${{selected.route}} · peak ${{selected.peak.toFixed(2)}} km (${{selected.peakYear}}) · 2000 ${{selected.final.toFixed(2)}} km · eLIA ${{selected.eLia.toFixed(3)}} · eRGI ${{selected.eRgi.toFixed(3)}}`;
    drawOverview(); drawDetail();
  }}
  select.value = String(curves[0].id);
  select.addEventListener('change', update);
  const observer = new ResizeObserver(() => update());
  observer.observe(root);
  update();
}})();
</script>
'''
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(fragment, encoding="utf-8")
    print(json.dumps({
        "status": "visualization-built",
        "formal_accepted_count": len(data),
        "requested_count": 57,
        "authoritative_57_registry_exists": False,
        "output": str(OUTPUT),
        "bytes": OUTPUT.stat().st_size,
        "all_curve_sources_exist": True,
    }, indent=2))


if __name__ == "__main__":
    main()
