const fs=require('fs'),vm=require('vm'),path=require('path');
const root=path.resolve(__dirname,'reports/observed-velocity-v1');
const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
const selector={value:'0',addEventListener(k,fn){this[k]=fn}},profile={src:'branch-00.png'};
const ctx={document:{getElementById(id){if(id==='branch')return selector;if(id==='profile')return profile;throw Error(id)}}};vm.createContext(ctx);
for(const m of html.matchAll(/<script>([\s\S]*?)<\/script>/g))vm.runInContext(m[1],ctx);
for(let i=0;i<8;i++){selector.value=String(i);selector.change();if(profile.src!==`branch-${String(i).padStart(2,'0')}.png`||!fs.existsSync(path.join(root,profile.src)))throw Error('Branch switch '+i)}
for(const m of html.matchAll(/(?:src|href)="([^"]+)"/g)){if(/^https?:/.test(m[1]))continue;if(!fs.existsSync(path.resolve(root,m[1])))throw Error('Missing '+m[1])}
const result={status:'pass',all_eight_branch_switches:true,local_links_and_images_exist:true,scope:'Delivered JavaScript in minimal Node DOM,not native browser automation'};
fs.writeFileSync(path.join(root,'page-check.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
