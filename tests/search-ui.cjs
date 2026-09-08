const {readFileSync}=require('node:fs');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const nodes=new Map();
function el(id){if(!nodes.has(id))nodes.set(id,{value:'',checked:true,innerHTML:'',textContent:'',addEventListener(){},querySelectorAll(){return []}});return nodes.get(id)}
const pending=[];
const context=vm.createContext({document:{querySelector:el,addEventListener(){},body:{classList:{add(){},remove(){},toggle(){}}}},console,Map,JSON,Date,Intl,setInterval:()=>0,clearInterval(){},requestAnimationFrame:f=>f(0),matchMedia:()=>({matches:true}),performance,
  setTimeout,clearTimeout,AbortController,URL,performance,location:{href:'https://example.com/',origin:'https://example.com'},
  fetch:(url,opts)=> url==='/api/stats'? Promise.resolve({json:async()=>({total:100})}):new Promise(resolve=>pending.push({url,resolve})),
});
const html=readFileSync('server/static/index.html','utf8');
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
vm.runInContext('render = () => {}',context);
const result=title=>({ok:true,json:async()=>({results:[{title}],ms:1})});
(async()=>{
  el('#q').value='forest';
  let p=vm.runInContext('run()',context);pending.shift().resolve(result('filtered'));await p;
  el('#commercial').checked=false;
  p=vm.runInContext('run()',context);
  assert.equal(pending.length,1,'filter changes must not reuse the old cache');
  pending.shift().resolve(result('unfiltered'));await p;
  el('#q').value='mountain';
  const old=vm.runInContext('run()',context);const req=pending.shift();
  el('#q').value='forest';await vm.runInContext('run()',context);
  req.resolve(result('stale mountain'));await old;
  assert.equal(vm.runInContext('results[0].title',context),'unfiltered','stale response must not replace cached results');
  el('#q').value='another query';p=vm.runInContext('run()',context);
  pending.shift().resolve({ok:false});await p;
  assert.match(el('#lat').textContent,/unavailable/);
  console.log('UI: filter cache, stale responses, and errors OK');
})().catch(e=>{console.error(e);process.exitCode=1});
