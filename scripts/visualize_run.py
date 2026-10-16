"""Build an offline explanation from real logs; optional PNG needs matplotlib.

Run from the repository root: python scripts/visualize_run.py
No model requests are made. Hidden evaluator labels are revealed only to the viewer.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edison.composition import CompositionWorld


def extract(run, baseline):
    assert run['profile'] == baseline['profile'] == 'combined'
    assert run['seed'] == baseline['seed']
    assert run['probe_messages'] == baseline['probe_messages']
    assert run['holdout_messages'] == baseline['holdout_messages']
    assert run['budget'] == baseline['budget'] == len(run['observations'])
    world = CompositionWorld(run['profile'], run['seed'])
    node_names = {n: ('A' if world._links['source', n][0] == .02 else 'B')
                  for n in world.primitives.nodes}
    transforms = {t: ('H' if m == 'hamming' else 'I') for t, m in world._transforms.items()}
    model_ids = {p['idea_id'] for e in run['generator_events'] for p in e.get('proposals', []) if p['new']}

    def construction(c):
        return {'path': [node_names[n] for n in c['path']],
                'transforms': [transforms[t] for t in c['transforms']],
                'repetitions': c['repetitions'], 'flip': c['flip']}

    observations = []
    for index, o in enumerate(run['observations'], 1):
        m = o['outcome']['measurement']
        observations.append({'trial': index, 'id': o['idea_id'], 'construction': construction(o['construction']),
                             'ber': m['bit_errors'] / m['bits'] * 100 if m else None,
                             'errors': m['bit_errors'] if m else None, 'bits': m['bits'] if m else None,
                             'cost': m['cost'] if m else None, 'model': o['idea_id'] in model_ids,
                             'selected': o['construction'] == run['selected']})
    selected = next(o for o in observations if o['selected'])
    # First-discovery ancestry is defined by the first incoming edge of each new node.
    incoming = {}
    for edge in run['graph']['edges']:
        incoming.setdefault(edge['child'], edge)
    ancestry = [selected['id']]
    while ancestry[-1] != 0:
        ancestry.append(incoming[ancestry[-1]]['parent'])
        assert len(ancestry) <= len(run['graph']['nodes'])
    ancestry.reverse()
    by_id = {o['id']: o for o in observations}
    lineage = [dict(by_id[i], operation=incoming[i]['operation'] if i else 'root') for i in ancestry]

    def first_pass(r):
        return next(i for i, o in enumerate(r['observations'], 1)
                    if (m := o['outcome']['measurement']) and m['cost'] <= r['public']['max_cost']
                    and m['bit_errors'] / m['bits'] <= r['public']['target_ber'])

    holdout = run['holdout']['measurement']
    return {'profile': run['profile'], 'seed': run['seed'], 'budget': run['budget'],
            'target_percent': run['public']['target_ber'] * 100,
            'max_cost': run['public']['max_cost'], 'observations': observations,
            'lineage': lineage, 'selected_trial': selected['trial'],
            'model_calls': run['model_calls_succeeded'], 'model_new_tested': run['model_origin_tested'],
            'graph_nodes': len(run['graph']['nodes']), 'graph_edges': len(run['graph']['edges']),
            'pending': len(run['graph']['pending']), 'holdout': holdout,
            'holdout_percent': holdout['bit_errors'] / holdout['bits'] * 100,
            'first_pass_model': first_pass(run), 'first_pass_baseline': first_pass(baseline),
            'baseline_holdout': baseline['holdout']['measurement'],
            'model_trials': [o['trial'] for o in observations if o['model']],
            'selected_has_model_ancestor': any(i in model_ids for i in ancestry[:-1]),
            'selected_direct_model_proposal': selected['id'] in model_ids}


HTML = r'''<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Эдисон · разбор реального запуска</title>
<style>
:root{--bg:#f4f4ef;--ink:#172c3b;--muted:#526572;--line:#d9dfda;--green:#087d62;--purple:#7454bc;--red:#b44a43;--gold:#d3a33e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,sans-serif}main{max-width:1160px;margin:auto;padding:30px 22px 60px}h1{font-size:clamp(28px,5vw,48px);line-height:1.1;margin:10px 0 16px}h2{font-size:25px;margin:0 0 14px}h3{font-size:18px;margin:0 0 8px}p{margin:8px 0 16px}.eyebrow{font-size:12px;letter-spacing:1.8px;text-transform:uppercase;color:var(--green);font-weight:800}.muted{color:var(--muted)}.panel{background:white;border:1px solid var(--line);border-radius:18px;padding:25px;margin:20px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat{padding:18px;background:white;border:1px solid var(--line);border-radius:14px}.stat strong{font-size:30px;display:block;line-height:1.25}.stat span{font-size:13px;color:var(--muted)}.small{font-size:13px}.callout{border-left:4px solid var(--gold);padding:10px 15px;background:#fff8e7;border-radius:0 8px 8px 0}.good{color:var(--green)}.purple{color:var(--purple)}.bad{color:var(--red)}.badge{display:inline-block;border-radius:6px;background:#e9edf0;padding:2px 7px;font-size:12px;font-weight:700}.flow{display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin:15px 0}.box{border:1px solid var(--line);border-radius:10px;padding:10px 12px;background:#f7faf8;min-width:70px;text-align:center}.arrow{color:#788b97}.network{width:100%;height:auto}.network text{font-family:system-ui,sans-serif;fill:var(--ink)}.network .label{font-size:14px}.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:16px 0}button,select{font:inherit;border:1px solid var(--line);background:white;color:var(--ink);padding:8px 12px;border-radius:8px;cursor:pointer}button:hover{background:#e9f3ef}input[type=range]{width:100%;accent-color:var(--green)}.chart{width:100%;height:auto;display:block}.chart circle{cursor:pointer}.chart text{font:12px system-ui,sans-serif;fill:var(--muted)}.metrics{display:flex;gap:25px;flex-wrap:wrap;margin:12px 0}.metrics strong{font-size:28px;display:block}.history{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.history button{padding:16px;text-align:left;border-top:4px solid var(--green);background:#fbfcfa}.history button.model{border-top-color:var(--purple)}.history .num{font-size:22px;font-weight:800}.bars{display:grid;gap:15px}.track{background:#edf0ec;border-radius:8px;height:24px;overflow:hidden}.bar{background:var(--green);height:100%;min-width:1px}.bar.model{background:var(--purple)}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid var(--line)}.scroll{overflow:auto;max-height:360px}.active-row{background:#edf6f1}a{color:var(--green)}.legend{display:flex;gap:15px;flex-wrap:wrap;font-size:13px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}.final-line{font-size:20px;font-weight:650}svg{overflow:visible}footer{color:var(--muted);font-size:13px;margin-top:25px}@media(max-width:720px){.grid,.history{grid-template-columns:1fr}.stats{grid-template-columns:1fr 1fr}.panel{padding:17px}.history button{padding:12px}.network .label{font-size:17px}}
</style><main>
<div class="eyebrow">Эдисон · фактический журнал · combined / seed 0</div>
<h1>Как доставить сообщение,<br>если канал его портит?</h1>
<p class="muted">Один реальный запуск: 96 опытов, два запроса к LLM. Здесь можно пройти журнал по шагам. Объясняющие названия A, B, H, I раскрыты для вас; генератор видел только случайные идентификаторы.</p>
<div class="stats"><div class="stat"><strong>≤ 2,5%</strong><span>допустимая доля ошибочных битов</span></div><div class="stat"><strong>≤ 4,0</strong><span>стоимость одного исходного бита</span></div><div class="stat"><strong id="stat-ber"></strong><span>ошибки выбранного решения на контроле</span></div><div class="stat"><strong>96</strong><span>уникальных испытаний</span></div></div>
<section class="panel"><h2>1. Что было в среде</h2>
<svg class="network" viewBox="0 0 1000 260" role="img" aria-label="Прямой канал шумный. Маршрут через узел A надёжнее, но второй участок инвертирует сигнал.">
<defs><marker id="tip" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#526572"/></marker></defs>
<path d="M190 130 L805 130" stroke="#b44a43" stroke-width="2.5" marker-end="url(#tip)"/><text x="500" y="113" text-anchor="middle" class="label">Напрямую: вероятность ошибки каждого бита 28%</text>
<path d="M190 120 L438 48 M560 48 L805 120" fill="none" stroke="#087d62" stroke-width="3" marker-end="url(#tip)"/>
<text x="290" y="55" text-anchor="middle" class="label">2% шума</text><text x="730" y="43" text-anchor="middle" class="label">2% шума</text><text x="730" y="66" text-anchor="middle" class="label">+ переворот 0 ↔ 1</text>
<path d="M190 142 L438 222 M560 222 L805 142" fill="none" stroke="#a8b2b6" stroke-width="2" marker-end="url(#tip)"/><text x="500" y="258" text-anchor="middle" class="label">Через B: оба участка сильно шумят</text>
<rect x="25" y="96" width="165" height="65" rx="13" fill="#eaf1f5"/><text x="108" y="125" text-anchor="middle" font-size="18">Отправитель</text><text x="108" y="147" text-anchor="middle" font-size="14">например, 1011</text>
<rect x="438" y="17" width="122" height="59" rx="13" fill="#e1f3e9"/><text x="499" y="53" text-anchor="middle" font-size="19">Узел A</text>
<rect x="438" y="191" width="122" height="54" rx="13" fill="#ecefee"/><text x="499" y="225" text-anchor="middle" font-size="19">Узел B</text>
<rect x="810" y="96" width="165" height="65" rx="13" fill="#eaf1f5"/><text x="893" y="125" text-anchor="middle" font-size="18">Получатель</text><text x="893" y="147" text-anchor="middle" font-size="14">нужны те же биты</text>
</svg><div class="grid"><div><h3>Доступные действия</h3><p>Добавлять промежуточные узлы, выстраивать цепочки преобразований, повторять передачу, переворачивать полученные биты.</p></div><div><h3>Неизвестные преобразователи</h3><p><b>H</b> — код Хэмминга: 4 бита превращаются в 7 и можно исправить одну ошибку в блоке. <b>I</b> — тождественное преобразование: ничего не меняет. Их свойства генератору не сообщались.</p></div></div>
<div class="callout">Стоимость = число участков × число повторов × расширение сообщения. Много повторений может снизить ошибки, но сделать конструкцию слишком дорогой.</div></section>
<section class="panel"><h2>2. Как были разделены роли</h2><div class="flow"><div class="box purple">LLM<br><small>предлагает комбинации</small></div><span class="arrow">→</span><div class="box">Реестр идей<br><small>объединяет точные повторы</small></div><span class="arrow">→</span><div class="box">Очередь + ответвления<br><small>ветви сохраняются</small></div><span class="arrow">→</span><div class="box good">Испытатель<br><small>ошибки и стоимость</small></div></div>
<p class="small muted">Обратной передачи оценок в LLM нет. Она не узнавала, что идея дала 99% ошибок. Структурный генератор создавал соседние варианты независимо от результатов. Итог выбирался по пробным измерениям и один раз проверялся на отдельной выборке.</p></section>
<section class="panel"><h2>3. Реальная ветвь, давшая итоговое решение</h2><p class="muted">Это происхождение конструкции, а не четыре подряд выполненных опыта. Между этими узлами испытывались другие ветви.</p><div class="history" id="lineage"></div>
<p class="small muted">Последний переход — перебор варианта с инверсией, а не вывод LLM из ошибки. I в выбранной цепочке лишний: он не улучшает канал. Небольшие различия ошибок эквивалентных конструкций обусловлены разными реализациями случайного шума.</p></section>
<section class="panel"><h2>4. Пройди все 96 опытов</h2><div class="controls"><button id="start">Сначала</button><button id="play">▶ Пройти</button><button id="winner">К найденному решению</button><label><input id="zoom" type="checkbox"> Масштаб 0–10%</label></div>
<label for="step">Показан опыт <b id="step-label"></b></label><input id="step" type="range" min="1" max="96" value="59">
<div class="legend"><span><i class="dot" style="background:var(--green)"></i>структурная идея, допустимая стоимость</span><span><i class="dot" style="background:var(--purple)"></i>новая идея LLM</span><span><i class="dot" style="background:var(--red)"></i>стоимость выше лимита</span></div>
<svg id="chart" class="chart" viewBox="0 0 1000 300" role="img" aria-label="Ошибки конструкций по порядку реальных испытаний"></svg>
<div class="grid"><div class="box" style="text-align:left"><span id="origin" class="badge"></span><h3 id="candidate" style="margin-top:12px"></h3><div class="metrics"><div><strong id="ber"></strong><span class="small">ошибки на пробной выборке</span></div><div><strong id="cost"></strong><span class="small">стоимость / предел 4,0</span></div></div><p id="verdict"></p></div><div><h3>Что значит наблюдение</h3><p id="explanation"></p><p class="small muted">Каждый такой опыт использовал 2048 исходных битов. Это измерение конкретной конструкции, не универсальный вывод о всём направлении.</p></div></div>
<details><summary>Таблица всех испытаний</summary><div class="scroll"><table><thead><tr><th>Опыт</th><th>Идея</th><th>Ошибка</th><th>Стоимость</th><th>Источник</th></tr></thead><tbody id="rows"></tbody></table></div></details></section>
<section class="panel"><h2>5. Что получилось — и что пока не доказано</h2><div class="grid"><div><h3>Первый проходящий вариант</h3><div class="bars"><div>Без LLM: опыт <b id="base-pass"></b><div class="track"><div id="base-bar" class="bar"></div></div></div><div>С LLM: опыт <b id="model-pass"></b><div class="track"><div id="model-bar" class="bar model"></div></div></div></div><p class="small muted">Одинаковые мир, seed, 96 опытов, размеры выборок. Это одно сравнение, не статистическая оценка метода. Здесь подключение LLM не ускорило поиск.</p></div><div><h3>Независимая проверка после поиска</h3><p id="holdout" class="final-line"></p><p id="final-cost"></p><p class="small">96 идей испытано; <b id="pending"></b> идей остаются в очереди. Ветки не удалены из-за неудачи.</p></div></div>
<div class="callout">Финальную комбинацию LLM напрямую не предложила. Но её предок пришёл от LLM, а структурный генератор добавил инверсию. Подключение и совместная работа проверены. Преимущество LLM, самостоятельное изобретение кодирования и zero-shot открытие этим запуском не доказаны.</div></section>
<footer>Источник: artifacts/openrouter-verified.json и artifacts/process-baseline.json. Показаны фактические измерения; повторных запросов к модели эта страница не делает. Свойства скрытого мира раскрыты ретроспективно для объяснения. Страница работает без интернета.</footer>
</main><script>
const D=__DATA__;
const $=s=>document.getElementById(s), fmt=(x,n=2)=>x.toLocaleString('ru-RU',{maximumFractionDigits:n,minimumFractionDigits:n});
let timer=null;
function desc(c){return `${c.path.length?'через '+c.path.join(' → '):'напрямую'} · ${c.transforms.length?c.transforms.join(' → '):'без преобразования'} · ×${c.repetitions}${c.flip?' · инверсия':''}`}
function explain(o){let c=o.construction;if(o.selected)return 'Эта конструкция выбрана по пробным измерениям. Узел A уменьшает шум, инверсия компенсирует переворот, H исправляет часть оставшихся ошибок. I ничего не меняет.';
if(o.cost>D.max_cost)return 'Даже если бы точность была достаточной, стоимость превышает ограничение. Идея остаётся в графе, но не проходит условия задачи.';
if(o.ber>D.target_percent&&o.ber>90)return 'Биты в основном перевёрнуты. Для нас это видно ретроспективно; генератору оценка не передавалась. Соседняя конструкция с инверсией будет создана структурным правилом.';
if(o.ber<=D.target_percent)return 'На пробной выборке конструкция проходит оба ограничения. Это ещё не отдельная контрольная проверка.';
return 'Стоимость допустима, но ошибок слишком много. Это не закрывает ветвь: её ответвления остаются в очереди.'}
function color(o){return o.cost>D.max_cost?'#b44a43':o.model?'#7454bc':'#087d62'}
function draw(){let step=Number($('step').value), ymax=$('zoom').checked?10:100,W=1000,H=300,L=53,R=18,T=24,B=40;
let x=i=>L+(i-1)/(D.budget-1)*(W-L-R),y=v=>H-B-v/ymax*(H-T-B);
let s='<defs><clipPath id="plotclip"><rect x="53" y="24" width="929" height="236"/></clipPath></defs>';
for(let v=0;v<=ymax;v+=ymax/4)s+=`<line x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}" stroke="#e1e6e2"/><text x="${L-9}" y="${y(v)+4}" text-anchor="end">${v}%</text>`;
for(let i of [1,24,48,72,96])s+=`<text x="${x(i)}" y="${H-15}" text-anchor="middle">${i}</text>`;
s+=`<line x1="${L}" y1="${y(D.target_percent)}" x2="${W-R}" y2="${y(D.target_percent)}" stroke="#172c3b" stroke-dasharray="5 5"/><text x="${W-R}" y="${y(D.target_percent)-8}" text-anchor="end">предел 2,5%</text><g clip-path="url(#plotclip)">`;
let best=Infinity,points=[];for(let o of D.observations){if(o.trial>step)break;if(o.cost<=D.max_cost&&o.ber!==null)best=Math.min(best,o.ber);if(Number.isFinite(best))points.push(`${x(o.trial)},${y(best)}`)}
s+=`<polyline points="${points.join(' ')}" stroke="#172c3b" stroke-width="1.6" fill="none"/>`;
for(let o of D.observations){if(o.ber===null)continue;s+=`<circle data-trial="${o.trial}" cx="${x(o.trial)}" cy="${y(o.ber)}" r="${o.trial===step?7:o.model?5:3.7}" fill="${color(o)}" opacity="${o.trial<=step?1:.12}" stroke="${o.trial===step?'#172c3b':'white'}" stroke-width="1.5"><title>Опыт ${o.trial}: ${fmt(o.ber)}%, стоимость ${fmt(o.cost)}</title></circle>`}
s+='</g>';s+=`<text x="500" y="298" text-anchor="middle">Порядковый номер испытания · тёмная линия — лучшая точность при допустимой стоимости</text>`;$('chart').innerHTML=s;
$('chart').querySelectorAll('[data-trial]').forEach(c=>c.onclick=()=>select(Number(c.dataset.trial)));
}
function select(i){$('step').value=i;render()}
function render(){let i=Number($('step').value),o=D.observations[i-1];$('step-label').textContent=`${i} из ${D.budget}`;$('origin').textContent=o.model?'Новое предложение LLM':'Структурное ответвление';$('candidate').textContent=desc(o.construction);$('ber').textContent=o.ber===null?'не измерено':fmt(o.ber)+'%';$('cost').textContent=o.cost===null?'—':fmt(o.cost);let pass=o.ber!==null&&o.ber<=D.target_percent&&o.cost<=D.max_cost;$('verdict').textContent=pass?'Оба ограничения выполнены':o.cost>D.max_cost?'Не проходит по стоимости':'Не проходит по точности';$('verdict').className=pass?'good':'bad';$('explanation').textContent=explain(o);draw()}
$('stat-ber').textContent=fmt(D.holdout_percent)+'%';$('step').max=D.budget;$('step').value=D.selected_trial;
$('lineage').innerHTML=D.lineage.map((o,i)=>`<button data-step="${o.trial}" class="${o.model?'model':''}"><div class="badge">Опыт ${o.trial}</div><p>${i===0?'Исходная передача':o.model?'LLM: новая комбинация':'Структурное изменение'}</p><div class="small">${desc(o.construction)}</div><div class="num">${fmt(o.ber)}%</div><div class="small">ошибок · стоимость ${fmt(o.cost)}</div></button>`).join('');
$('lineage').querySelectorAll('[data-step]').forEach(b=>b.onclick=()=>{select(Number(b.dataset.step));$('step').scrollIntoView({behavior:'smooth',block:'center'})});
$('rows').innerHTML=D.observations.map(o=>`<tr data-step="${o.trial}"><td><button>${o.trial}</button></td><td>${desc(o.construction)}</td><td>${o.ber===null?'—':fmt(o.ber)+'%'}</td><td>${o.cost===null?'—':fmt(o.cost)}</td><td>${o.model?'LLM':'структура'}</td></tr>`).join('');$('rows').querySelectorAll('[data-step]').forEach(b=>b.onclick=()=>select(Number(b.dataset.step)));
$('base-pass').textContent=D.first_pass_baseline;$('model-pass').textContent=D.first_pass_model;$('base-bar').style.width=D.first_pass_baseline/D.budget*100+'%';$('model-bar').style.width=D.first_pass_model/D.budget*100+'%';$('holdout').textContent=`${D.holdout.bit_errors} ошибок / ${D.holdout.bits.toLocaleString('ru-RU')} бит = ${fmt(D.holdout_percent)}%`;$('final-cost').textContent=`Стоимость ${fmt(D.holdout.cost)} при пределе ${fmt(D.max_cost)}. Контроль выполнен после всех 96 опытов.`;$('pending').textContent=D.pending;
$('step').oninput=render;$('zoom').onchange=draw;$('start').onclick=()=>select(1);$('winner').onclick=()=>select(D.selected_trial);$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='▶ Пройти';return}if(Number($('step').value)>=D.budget)select(1);$('play').textContent='⏸ Пауза';timer=setInterval(()=>{let i=Number($('step').value);if(i>=D.budget){clearInterval(timer);timer=null;$('play').textContent='▶ Пройти'}else select(i+1)},300)};render();
</script></html>'''


def plot(data, output):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'edison-matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'edison-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(16, 11), facecolor='#f4f4ef')
    fig.text(.055, .955, 'ЭДИСОН  /  РЕАЛЬНЫЙ ЗАПУСК', color='#087d62', fontsize=12, weight='bold')
    fig.text(.055, .91, 'Доставить биты: ошибки ≤ 2,5%, стоимость ≤ 4,0', fontsize=23, weight='bold', color='#172c3b')
    fig.text(.055, .878, 'Составная среда · seed 0 · 96 опытов · 2 запроса к LLM', color='#526572', fontsize=13)
    ax = fig.add_axes([.055, .62, .89, .21]); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for i, o in enumerate(data['lineage']):
        x = i * .255
        color = '#7454bc' if o['model'] else '#087d62'
        ax.add_patch(FancyBboxPatch((x, .15), .225, .73, boxstyle='round,pad=0.008',
                                   facecolor='white', edgecolor=color, linewidth=2))
        c = o['construction']
        label = ('напрямую' if not c['path'] else 'через ' + ' → '.join(c['path']))
        label += '\n' + (' → '.join(c['transforms']) if c['transforms'] else 'без преобразования')
        label += '\n' + ('с инверсией' if c['flip'] else 'без инверсии')
        ax.text(x + .015, .76, f"Опыт {o['trial']} · {'LLM' if o['model'] else 'структура'}", color=color, fontsize=12, weight='bold')
        ax.text(x + .015, .58, label, fontsize=11, va='top', linespacing=1.6)
        ax.text(x + .015, .23, f"{o['ber']:.2f}% ошибок · цена {o['cost']:g}", fontsize=12, weight='bold')
        if i < 3:
            ax.annotate('', (x + .25, .5), (x + .23, .5), arrowprops={'arrowstyle': '->', 'color': '#526572'})
    fig.text(.055, .61, 'Реальная ветвь происхождения. Между этими узлами испытывались другие варианты.', fontsize=11, color='#526572')
    ax = fig.add_axes([.07, .285, .54, .26], facecolor='white')
    obs = data['observations']
    for name, color, pred in [('Структура', '#087d62', lambda o: not o['model'] and o['cost'] <= 4),
                              ('Новые идеи LLM', '#7454bc', lambda o: o['model'] and o['cost'] <= 4),
                              ('Дороже лимита', '#b44a43', lambda o: o['cost'] > 4)]:
        points = [o for o in obs if o['ber'] is not None and pred(o)]
        ax.scatter([o['trial'] for o in points], [o['ber'] for o in points], color=color, s=23, label=name, alpha=.85)
    best = float('inf'); xs, ys = [], []
    for o in obs:
        if o['ber'] is not None and o['cost'] <= 4:
            best = min(best, o['ber'])
        if best < float('inf'):
            xs.append(o['trial']); ys.append(best)
    ax.step(xs, ys, where='post', color='#172c3b', lw=1.6)
    ax.axhline(2.5, color='#172c3b', ls='--', lw=1)
    ax.set(xlim=(0, 98), ylim=(-3, 103), xlabel='Порядковый номер испытания', ylabel='Ошибка на пробной выборке')
    ax.yaxis.set_major_formatter(PercentFormatter(100)); ax.grid(alpha=.15)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('Все измерения; линия — лучшее при допустимой цене', fontsize=12, loc='left', pad=12)
    ax = fig.add_axes([.68, .285, .26, .26]); ax.set_axis_off()
    ax.text(0, .93, 'Проверка на новых битах', fontsize=14, weight='bold')
    ax.text(0, .72, f"{data['holdout_percent']:.2f}%", fontsize=36, weight='bold', color='#087d62')
    ax.text(0, .57, f"{data['holdout']['bit_errors']} ошибок / {data['holdout']['bits']:,} бит".replace(',', ' '), fontsize=12)
    ax.text(0, .41, f"Стоимость {data['holdout']['cost']:g} из {data['max_cost']:g}", fontsize=14)
    ax.text(0, .24, f"Первый успех без LLM: опыт {data['first_pass_baseline']}\nС LLM: опыт {data['first_pass_model']}", fontsize=12, linespacing=1.8)
    fig.text(.055, .19, 'Что именно сработало', fontsize=15, weight='bold', color='#172c3b')
    fig.text(.055, .15, 'Узел A уменьшил шум → инверсия компенсировала переворот → H исправил оставшиеся ошибки.', fontsize=12)
    fig.text(.055, .118, 'H: код Хэмминга 4 → 7 бит. I: ничего не меняет; в выбранной конструкции он лишний.', fontsize=11, color='#526572')
    fig.text(.055, .075, 'LLM не получала оценок. Финальное решение — структурное ответвление её идеи.', fontsize=12, weight='bold')
    fig.text(.055, .045, 'В этом одном сравнении LLM не ускорила поиск. Преимущество модели и zero-shot открытие не доказаны.', fontsize=11, color='#526572')
    fig.savefig(output, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='artifacts/openrouter-verified.json')
    parser.add_argument('--baseline', default='artifacts/process-baseline.json')
    parser.add_argument('--data', help='Use the committed, sanitized visualization data')
    parser.add_argument('--no-png', action='store_true')
    args = parser.parse_args()
    if args.data:
        data = json.loads(Path(args.data).read_text())
    elif Path(args.run).exists() and Path(args.baseline).exists():
        data = extract(json.loads(Path(args.run).read_text()), json.loads(Path(args.baseline).read_text()))
    elif Path('reports/edison-run-data.json').exists():
        data = json.loads(Path('reports/edison-run-data.json').read_text())
    else:
        parser.error('Provide original logs or --data reports/edison-run-data.json')
    Path('docs').mkdir(exist_ok=True)
    Path('reports').mkdir(exist_ok=True)
    serialized = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')
    Path('reports/edison-run-data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    Path('docs/edison-run.html').write_text(HTML.replace('__DATA__', serialized))
    if not args.no_png:
        plot(data, 'docs/edison-run.png')
    print('Built real-run visualization: docs/edison-run.html')


if __name__ == '__main__':
    main()
