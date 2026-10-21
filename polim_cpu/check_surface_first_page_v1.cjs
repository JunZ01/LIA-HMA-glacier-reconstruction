const fs=require('fs'),vm=require('vm'),path=require('path');
const root=path.resolve(__dirname,'reports/surface-first-geometry-v1');
const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
const elements=new Map();let tick=null;
const ctx={window:{},document:{getElementById(id){if(!elements.has(id))elements.set(id,{value:'0',textContent:'',src:'',addEventListener(k,fn){this[k]=fn}});return elements.get(id)}},setInterval(fn){tick=fn;return 1},clearInterval(){tick=null}};
vm.createContext(ctx);vm.runInContext(fs.readFileSync(path.join(root,'data.js'),'utf8'),ctx);
for(const m of html.matchAll(/<script>([\s\S]*?)<\/script>/g))vm.runInContext(m[1],ctx);
const slider=elements.get('time'),play=elements.get('play');
function check(i){if(elements.get('clock').textContent!==i+' 小时'||elements.get('map').src!==`frames/frame-${String(i).padStart(3,'0')}.png`)throw Error('Frame '+i);if(elements.get('newSpeed').textContent!==ctx.window.SURFACE_DATA.new[i].max_speed_m_year.toFixed(1)+' 米/年')throw Error('Speed '+i)}
check(0);for(const i of [12,24]){slider.value=String(i);slider.input();check(i)}
play.onclick();check(0);if(!tick)throw Error('Replay did not start');tick();check(1);play.onclick();if(tick)throw Error('Pause failed');
slider.value='23';slider.input();play.onclick();tick();check(24);if(tick||play.textContent!=='重新播放')throw Error('Playback did not stop');
for(const m of html.matchAll(/(?:src|href)="([^"]+)"/g)){if(/^https?:/.test(m[1]))continue;if(!fs.existsSync(path.resolve(root,m[1])))throw Error('Missing '+m[1])}
for(const name of ctx.window.SURFACE_DATA.images)if(!fs.existsSync(path.join(root,name)))throw Error('Missing '+name);
const result={status:'pass',frames:25,checked_times:[0,12,24],replay_pause_and_end:true,display_matches_saved_metrics:true,local_links_and_images_exist:true,scope:'Delivered JavaScript in minimal Node DOM; not native browser automation'};
fs.writeFileSync(path.join(root,'page-check.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
