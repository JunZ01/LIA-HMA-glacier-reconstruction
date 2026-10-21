"""Link the completed day-one maps into the existing working pages."""
from .outlet_evolution_v1 import R,REPORT,read
from .v3.runner import write_json,digest
from pathlib import Path
import re

def main():
 assert read(REPORT/'summary.json')['status']=='completed'
 write_json(REPORT/'visual-review.json',dict(status='reviewed',images=['hour-24.png','timeseries.png'],observations='Correct north-up map orientation,consistent fixed scales,legible Chinese labels,old A marker;time-series trends match hourly values. No browser-interaction test claimed.',hourly_files=25))
 main=R/'index.html';working=R/'reports/system2387-working-v1/index.html';saved={}
 for path,label in [(main,'project'),(working,'working')]:
  before=path.read_text(encoding='utf-8');backup=REPORT/f'previous-{label}-index.html';backup.write_text(before,encoding='utf-8')
  if label=='project':
   link='<a class="current" href="reports/independent-outlet-day1-v1/index.html"><b>查看新增结果：24 小时动力演化</b><p>逐小时播放冰厚与速度变化；最高冰速 50.345 → 50.316 米/年，最大冰厚变化 0.298 米。</p></a>'
   updated=before.replace('<h1>2387 冰川模型</h1>','<h1>2387 冰川模型</h1>'+link,1)
  else:
   link='<div id="day1-link" style="position:fixed;right:20px;bottom:18px;z-index:9999;background:#087e9e;color:white;padding:14px 20px;border-radius:12px;font:16px Microsoft YaHei,sans-serif;box-shadow:0 3px 15px #0003"><a href="../independent-outlet-day1-v1/index.html" style="color:white;text-decoration:none">新增：查看 24 小时动力演化 ▶</a></div>'
   assert '</body>' in before;updated=before.replace('</body>',link+'</body>',1)
  assert updated!=before;path.write_text(updated,encoding='utf-8');saved[str(path.relative_to(R))]=digest(path)
 html=(REPORT/'index.html').read_text(encoding='utf-8')
 links=re.findall(r'(?:href|src)="([^"]+)"',html)
 for link in links:
  if not link.startswith(('http:','https:','#','data:')):assert (REPORT/link).resolve().exists(),link
 for k in range(25):assert (REPORT/f'hour-{k:02d}.png').is_file()
 write_json(REPORT/'navigation.json',dict(updated_pages=saved,local_links_verified=links,hourly_frames_exist=25))
 # Refresh only the display artifact manifest; all numerical outputs remain sealed.
 write_json(REPORT/'navigation-delivery.json',dict(files={p.name:digest(p) for p in REPORT.iterdir() if p.is_file() and p.name!='delivery.json'},numerical_result_sha256=read(REPORT/'summary.json')['source_result_sha256']))
 print('Day-one report linked;25frames and local links exist.')

if __name__=='__main__':main()
