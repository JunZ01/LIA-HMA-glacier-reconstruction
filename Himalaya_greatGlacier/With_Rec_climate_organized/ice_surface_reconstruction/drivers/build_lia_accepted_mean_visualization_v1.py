"""Build Figure 2: mean trajectory of the 54 formally accepted LIA curves."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .build_lia_accepted_curves_visualization_v1 import (
    INDEX,
    METRICS,
    curve_rows,
    read_csv,
)


OUTPUT = Path(
    r"C:\Users\JUNZ\.codex\visualizations\2026\08\16\01a008a2-6e98-73b1-8098-7f4262d2a7d2\lia-accepted-curves-mean-figure2.html"
)


def main():
    metrics = {int(r["system_id"]): r for r in read_csv(METRICS)}
    index = {int(r["system_id"]): r for r in read_csv(INDEX)}
    if len(metrics) != 54 or set(metrics) != set(index):
        raise RuntimeError("mean plot requires exactly the authoritative 54 accepted systems")

    years = np.arange(1470, 2001, dtype=float)
    curves = []
    for sid in sorted(metrics):
        m = metrics[sid]
        points, _ = curve_rows(sid, m["route"], index[sid]["reproduction_root"])
        xp = np.asarray([p[0] for p in points], dtype=float)
        fp = np.asarray([p[1] for p in points], dtype=float)
        curves.append(np.interp(years, xp, fp))
    array = np.vstack(curves)
    mean = np.mean(array, axis=0)
    p10 = np.percentile(array, 10, axis=0)
    p90 = np.percentile(array, 90, axis=0)
    mean_lia = float(np.mean([float(x["lia_km"]) for x in metrics.values()]))
    mean_rgi = float(np.mean([float(x["rgi_km"]) for x in metrics.values()]))
    peak_i = int(np.argmax(mean))
    data = [
        [int(y), round(float(a), 4), round(float(lo), 4), round(float(hi), 4)]
        for y, a, lo, hi in zip(years, mean, p10, p90)
    ]
    stats = {
        "n": 54,
        "meanLia": round(mean_lia, 4),
        "meanRgi": round(mean_rgi, 4),
        "start": round(float(mean[0]), 4),
        "peak": round(float(mean[peak_i]), 4),
        "peakYear": int(years[peak_i]),
        "final": round(float(mean[-1]), 4),
        "change": round(float(mean[-1] - mean[0]), 4),
    }
    payload = json.dumps(data, separators=(",", ":"))
    stats_payload = json.dumps(stats, separators=(",", ":"))
    fragment = f'''<div id="lia-accepted-mean-v1" class="lia-mean-root">
  <h2>Figure 2. Mean length trajectory of 54 accepted glaciers</h2>
  <div id="lia-mean-summary" class="lia-mean-summary"></div>
  <div id="lia-mean-chart" class="lia-mean-chart"></div>
  <div id="lia-mean-tooltip" class="tooltip" role="tooltip" aria-hidden="true"></div>
</div>
<style>
#lia-accepted-mean-v1 {{ color: var(--foreground); width: 100%; }}
#lia-accepted-mean-v1 h2 {{ font-weight: 500; margin: 0 0 8px; }}
#lia-accepted-mean-v1 .lia-mean-summary {{ color: var(--muted-foreground); margin-bottom: 8px; min-height: 20px; }}
#lia-accepted-mean-v1 .lia-mean-chart {{ width: 100%; min-height: 320px; }}
#lia-accepted-mean-v1 svg {{ display: block; width: 100%; overflow: visible; }}
#lia-accepted-mean-v1 svg text {{ fill: var(--foreground); font-size: 12px; }}
#lia-accepted-mean-v1 .axis path, #lia-accepted-mean-v1 .axis line {{ stroke: var(--border); }}
#lia-accepted-mean-v1 .grid line {{ stroke: var(--border); stroke-opacity: .55; }}
#lia-accepted-mean-v1 .grid path {{ display: none; }}
#lia-accepted-mean-v1 rect[data-chart-frame] {{ fill: none; stroke: var(--border); }}
#lia-accepted-mean-v1 .range-band {{ fill: var(--viz-series-1); opacity: .16; }}
#lia-accepted-mean-v1 .mean-line {{ fill: none; stroke: var(--viz-series-1); stroke-width: 2.8; }}
#lia-accepted-mean-v1 .lia-target {{ stroke: var(--viz-series-2); stroke-width: 1.6; stroke-dasharray: 6 4; }}
#lia-accepted-mean-v1 .rgi-target {{ stroke: var(--viz-series-3); stroke-width: 1.6; stroke-dasharray: 2 4; }}
#lia-accepted-mean-v1 .hover-guide {{ stroke: var(--foreground); stroke-opacity: .45; }}
#lia-accepted-mean-v1 .hover-marker {{ fill: var(--viz-series-1); stroke: var(--background); stroke-width: 1.5; }}
#lia-accepted-mean-v1 .tooltip {{ position: absolute; pointer-events: none; opacity: 0; color: var(--popover-foreground); background: var(--popover); border: 1px solid var(--border); padding: 7px 9px; z-index: 5; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {{
  const root = document.getElementById('lia-accepted-mean-v1');
  const data = {payload};
  const stats = {stats_payload};
  root.querySelector('#lia-mean-summary').textContent = `Mean peak ${{stats.peak.toFixed(2)}} km (${{stats.peakYear}}) · 1470 ${{stats.start.toFixed(2)}} km · 2000 ${{stats.final.toFixed(2)}} km · net change ${{stats.change.toFixed(2)}} km`;
  const container = root.querySelector('#lia-mean-chart');
  const tooltip = root.querySelector('#lia-mean-tooltip');
  function draw() {{
    container.innerHTML='';
    const width=Math.max(320,container.getBoundingClientRect().width||736), height=width<480?290:340;
    const margin={{top:16,right:28,bottom:50,left:68}};
    const x=d3.scaleLinear().domain([1470,2000]).range([margin.left,width-margin.right]);
    const extent=d3.extent(data.flatMap(d=>[d[2],d[3],stats.meanLia,stats.meanRgi]));
    const pad=Math.max(.4,(extent[1]-extent[0])*.06);
    const y=d3.scaleLinear().domain([extent[0]-pad,extent[1]+pad]).nice().range([height-margin.bottom,margin.top]);
    const svg=d3.select(container).append('svg').attr('viewBox',`0 0 ${{width}} ${{height}}`).attr('aria-label','Mean glacier length with 10 to 90 percent range');
    svg.append('rect').attr('data-chart-frame','').attr('x',margin.left).attr('y',margin.top).attr('width',width-margin.left-margin.right).attr('height',height-margin.top-margin.bottom);
    svg.append('g').attr('class','grid').attr('transform',`translate(${{margin.left}},0)`).call(d3.axisLeft(y).tickSize(-(width-margin.left-margin.right)).tickFormat(''));
    svg.append('g').attr('class','axis').attr('transform',`translate(0,${{height-margin.bottom}})`).call(d3.axisBottom(x).ticks(width<480?4:7).tickFormat(d3.format('d')));
    svg.append('g').attr('class','axis').attr('transform',`translate(${{margin.left}},0)`).call(d3.axisLeft(y).ticks(6));
    svg.append('text').attr('class','axis-title').attr('data-axis','x').attr('x',(margin.left+width-margin.right)/2).attr('y',height-10).attr('text-anchor','middle').text('Year');
    svg.append('text').attr('class','axis-title').attr('data-axis','y').attr('transform','rotate(-90)').attr('x',-(margin.top+height-margin.bottom)/2).attr('y',16).attr('text-anchor','middle').text('Mean length (km)');
    const area=d3.area().x(d=>x(d[0])).y0(d=>y(d[2])).y1(d=>y(d[3]));
    const line=d3.line().x(d=>x(d[0])).y(d=>y(d[1]));
    svg.append('path').datum(data).attr('class','range-band').attr('d',area);
    svg.append('path').datum(data).attr('class','mean-line').attr('d',line);
    svg.append('line').attr('class','lia-target').attr('x1',x(1470)).attr('x2',x(2000)).attr('y1',y(stats.meanLia)).attr('y2',y(stats.meanLia));
    svg.append('line').attr('class','rgi-target').attr('x1',x(1470)).attr('x2',x(2000)).attr('y1',y(stats.meanRgi)).attr('y2',y(stats.meanRgi));
    svg.append('text').attr('x',x(1476)).attr('y',y(stats.meanLia)-6).text(`Mean LIA target ${{stats.meanLia.toFixed(2)}} km`);
    svg.append('text').attr('x',x(1476)).attr('y',y(stats.meanRgi)+15).text(`Mean RGI target ${{stats.meanRgi.toFixed(2)}} km`);
    const guide=svg.append('line').attr('class','hover-guide').attr('data-chart-hover-guide','').style('display','none');
    const marker=svg.append('circle').attr('class','hover-marker').attr('data-chart-hover-marker','').attr('r',4).style('display','none');
    svg.append('rect').attr('data-chart-hit','').attr('data-chart-hover-overlay','cross-series').attr('x',margin.left).attr('y',margin.top).attr('width',width-margin.left-margin.right).attr('height',height-margin.top-margin.bottom).attr('fill','transparent')
      .on('pointermove',event=>{{
        const [mx]=d3.pointer(event); const year=Math.round(Math.max(1470,Math.min(2000,x.invert(mx)))); const d=data[year-1470];
        guide.style('display',null).attr('x1',x(year)).attr('x2',x(year)).attr('y1',margin.top).attr('y2',height-margin.bottom);
        marker.style('display',null).attr('cx',x(year)).attr('cy',y(d[1]));
        tooltip.style.opacity=1; tooltip.setAttribute('aria-hidden','false'); tooltip.innerHTML=`${{year}} · mean ${{d[1].toFixed(2)}} km<br>10–90% range ${{d[2].toFixed(2)}}–${{d[3].toFixed(2)}} km`;
        const rb=root.getBoundingClientRect(); tooltip.style.left=`${{event.clientX-rb.left+12}}px`; tooltip.style.top=`${{event.clientY-rb.top-24}}px`;
      }})
      .on('pointerleave',()=>{{guide.style('display','none');marker.style('display','none');tooltip.style.opacity=0;tooltip.setAttribute('aria-hidden','true');}});
  }}
  new ResizeObserver(draw).observe(root); draw();
}})();
</script>
'''
    OUTPUT.write_text(fragment, encoding="utf-8")
    print(json.dumps({"status":"figure2-built","n":54,"stats":stats,"output":str(OUTPUT),"bytes":OUTPUT.stat().st_size},indent=2))


if __name__ == "__main__":
    main()
