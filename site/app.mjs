import {recommend, validateBundle} from './scorer.mjs';
const $ = id => document.getElementById(id);
let bundle, sandbox, selected = new Set();
const fmt = n => (n * 100).toFixed(2) + '%';
const labels = {popularity:'Popularity', item_cf:'Item–item CF', bpr:'PyTorch BPR'};
function element(tag, text, cls) { const e = document.createElement(tag); e.textContent = text; if(cls)e.className=cls; return e; }
function showChoices() {
  const q = $('search').value.toLocaleLowerCase();
  $('choices').replaceChildren();
  const matches = bundle.movies.filter(m => m.title.toLocaleLowerCase().includes(q));
  for(const m of matches.slice(0,60)) {
    const b = element('button', m.title, 'choice'); b.id = 'movie-' + m.id; b.setAttribute('aria-pressed', selected.has(m.id));
    b.onclick = () => { selected.has(m.id) ? selected.delete(m.id) : selected.add(m.id); render(); document.getElementById('movie-' + m.id)?.focus({preventScroll:true}); };
    $('choices').append(b);
  }
  if(!matches.length)$('choices').append(element('p','No matching movies. Try another title.','hint'));
}
function render() {
  showChoices(); $('selected').replaceChildren();
  for(const id of selected) {
    const title = bundle.movies.find(m=>m.id===id).title;
    const chip = element('button',title+' ×','chip'); chip.setAttribute('aria-label','Remove '+title);
    chip.onclick=()=>{selected.delete(id);render();}; $('selected').append(chip);
  }
  const t = performance.now(), recs = recommend(bundle,[...selected],6), ms = performance.now()-t;
  $('results').replaceChildren(...recs.map(m=>{const li=document.createElement('li');li.append(element('span',m.title),element('span',selected.size?m.score.toFixed(3):String(m.score),'score'));return li;}));
  $('count').textContent = selected.size ? `${selected.size} favorite${selected.size===1?'':'s'} selected` : 'Training popularity';
  $('status').textContent = recs.length ? `${selected.size?'Personalized':'Popularity'} ranking · ${ms.toFixed(1)} ms in this browser` : 'All movies selected. Remove a favorite to see recommendations.';
}
function useBundle(b) {
  bundle = validateBundle(b); selected.clear(); $('search').value='';
  $('source-label').textContent = b.restricted ? 'PRIVATE MOVIELENS MODEL · loaded locally' : 'FICTIONAL SANDBOX · synthetic preferences';
  $('source-description').textContent = b.restricted ? 'Real movie titles and trained item embeddings from your local MovieLens run. This file is held in memory and never uploaded. Do not redistribute it without permission.' : 'Original fictional titles and embeddings trained on synthetic preferences. This demonstrates live scoring, not MovieLens quality. Load a private bundle below to use real movies.';
  $('search').placeholder = b.restricted ? 'Search your MovieLens catalog…' : 'Search the fictional catalog…'; render();
}
$('search').oninput=()=>{if(bundle)showChoices();};
$('clear').onclick=()=>{if(bundle){selected.clear();render();}};
$('sandbox').onclick=()=>{if(sandbox){useBundle(sandbox);$('bundle').value='';}};
$('bundle').onchange=async e=>{
  const f=e.target.files[0];if(!f)return;
  try {if(f.size>25_000_000)throw Error('Model file must be smaller than 25 MB.');useBundle(JSON.parse(await f.text()));}
  catch(err){$('status').textContent='Could not load model: '+err.message;}
};
async function loadResults() {
  const r=await fetch('./metrics.json');if(!r.ok)throw Error('Results unavailable'); const data=await r.json();
  const models=data.models, names=Object.keys(labels), winner=names.reduce((a,b)=>models[a].ndcg>=models[b].ndcg?a:b);
  $('takeaway').textContent=`${labels[winner]} has the highest test NDCG@10 point estimate. Neither personalized model establishes an improvement over popularity here. ${models.popularity.by_history['0'].users} of ${data.protocol.test_exclusions.evaluated_users} evaluated users have no training history.`;
  for(const [value,label] of [[data.protocol.train_rows.toLocaleString(),'training ratings'],[data.protocol.eligible_catalog.toLocaleString(),'eligible movies'],[data.protocol.test_exclusions.evaluated_users,'evaluated test users'],['100%','eligible catalog ranked']]) {
    const div=element('div','','fact');div.append(element('strong',String(value)),element('span',label));$('facts').append(div);
  }
  for(const name of names) {
    const m=models[name],tr=document.createElement('tr');if(name===winner)tr.className='best';
    [labels[name],fmt(m.recall),fmt(m.ndcg),fmt(m.coverage),data.latency[name].p95_ms.toFixed(2)+' ms'].forEach(v=>tr.append(element('td',v)));$('metrics').append(tr);
    const g=element('div','','bar-group');g.append(element('p',labels[name]));
    for(const [key,label] of [['ndcg','NDCG'],['coverage','Coverage']]){const row=element('div','','bar-row '+key),track=element('div','','track'),fill=element('div','','fill');fill.style.width=(m[key]*100)+'%';track.append(fill);row.append(element('span',label),track,element('span',(m[key]*100).toFixed(1)+'%'));g.append(row);}$('bars').append(g);
  }
  for(const [depth,cohort] of Object.entries(models.popularity.by_history)){
    const tr=document.createElement('tr');[depth,cohort.users,...names.map(n=>models[n].by_history[depth].ndcg===null?'—':fmt(models[n].by_history[depth].ndcg))].forEach(v=>tr.append(element('td',String(v))));$('cohorts').append(tr);
  }
  for(const [key,label] of [['item_cf_minus_popularity','Item CF − popularity'],['bpr_minus_popularity','BPR − popularity'],['bpr_minus_item_cf','BPR − item CF']]) {
    const c=data.paired_bootstrap_95_ci[key].ndcg,div=element('div','','ci'),pp=n=>(n*100).toFixed(2);
    div.append(element('span',label+' · NDCG'),element('strong',`${pp(c.delta)} pp [${pp(c.low)}, ${pp(c.high)}]`));$('intervals').append(div);
  }
}
try {const r=await fetch('./sandbox.json');if(!r.ok)throw Error('Sandbox unavailable');sandbox=await r.json();useBundle(sandbox);}catch(e){$('status').textContent=e.message;}
try {await loadResults();}catch(e){$('takeaway').textContent='Results could not load. Serve this site over HTTP and retry.';}
