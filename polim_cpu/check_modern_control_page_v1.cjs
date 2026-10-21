const fs=require('fs'),vm=require('vm'),path=require('path');
const root=path.resolve(__dirname,'reports/modern-raw-f19-bp-control-v1');const html=fs.readFileSync(path.join(root,'index.html'),'utf8');const nodes=new Map();
const ctx={window:{},document:{getElementById(id){if(!nodes.has(id))nodes.set(id,{textContent:'',src:''});return nodes.get(id)}}};vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(root,'data.js'),'utf8'),ctx);for(const m of html.matchAll(/<script>([\s\S]*?)<\/script>/g))vm.runInContext(m[1],ctx);
for(const [id,name] of [['speed','velocity.png'],['geometry','geometry.png'],['branches','branches.png']]){nodes.get(id).onclick();if(nodes.get('map').src!==name||!nodes.get('viewTitle').textContent)throw Error(id)}
if(!nodes.get('outcome').textContent.includes(ctx.window.MODERN_DATA.summary.peak.speed_m_year.toFixed(1)))throw Error('Saved peak mismatch');
for(const m of html.matchAll(/(?:src|href)="([^"]+)"/g)){if(/^https?:/.test(m[1]))continue;if(!fs.existsSync(path.resolve(root,m[1])))throw Error('Missing '+m[1])}
const result={status:'pass',three_views_switch:true,metrics_match_saved_result:true,local_assets_and_links_exist:true,scope:'Minimal Node DOM,not native browser automation'};fs.writeFileSync(path.join(root,'page-check.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
