"""Illustrate actual Stalker/scientist records. No API calls; PNG needs matplotlib."""
import json
import os
from pathlib import Path
import tempfile


def load():
    report = json.loads(Path('reports/discovery-run.json').read_text())
    knowledge = json.loads(Path('reports/discovery-knowledge.json').read_text())
    entities = report['openrouter_discovery_run']['discovery_events']
    facts = knowledge['facts']
    transforms = []
    nodes = []
    for index, entity in enumerate(entities, 1):
        item = dict(entity, number=index)
        item['facts'] = {f['property']: f for f in facts if f['entity'] == entity['entity_id']}
        item['probe_count'] = sum(e['entity'] == entity['entity_id'] for e in knowledge['evidence'])
        (transforms if entity['kind'] == 'transform' else nodes).append(item)
    correcting = next(t for t in transforms
                      if t['facts']['single_bit_flip_recovery']['value']['recovered'] > 0)
    node = min(nodes, key=lambda n: n['facts']['incoming_link_observation']['value']['mismatch_fraction'])
    example = next(e for e in knowledge['evidence'] if e['entity'] == correcting['entity_id']
                   and e['experiment'] == {'input': [1, 0, 1, 1], 'flip_positions': [2]})
    return {'report': report, 'knowledge': knowledge, 'entities': entities,
            'transforms': transforms, 'nodes': nodes, 'correcting': correcting,
            'node': node, 'example': example}


def bits(value):
    return ''.join(map(str, value))


def png(data):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'edison-matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'edison-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})
    fig = plt.figure(figsize=(16, 14.5), facecolor='#f4f4ef')
    ax = fig.add_axes([.035, .03, .93, .94]); ax.set_axis_off()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ink, muted, green, purple, red = '#172c3b', '#526572', '#087d62', '#7454bc', '#b44a43'
    def text(x, y, s, size=12, color=ink, weight='normal', **kw):
        ax.text(x, y, s, fontsize=size, color=color, weight=weight, va='top', **kw)
    def card(x, y, w, h, color=green):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008',
                                   facecolor='white', edgecolor=color, linewidth=1.5))
    def arrow(x1, y1, x2, y2, color=muted):
        ax.annotate('', (x2, y2), (x1, y1), arrowprops={'arrowstyle': '->', 'color': color, 'lw': 2})
    text(.01, .985, 'ЭДИСОН  /  ОБНАРУЖЕНИЕ И ИЗУЧЕНИЕ', 12, green, 'bold')
    text(.01, .946, 'Сталкер находит. Учёный проверяет.', 27, weight='bold')
    text(.01, .904, 'Фактический запуск: пустой каталог → 4 сущности → 218 опытов → 14 свойств', 14, muted)
    card(.01, .725, .26, .14, muted)
    text(.025, .847, 'ИНФОПОЛЕ', 15, weight='bold')
    text(.025, .816, 'Сейчас — локальный источник.\nДва узла и два преобразователя.\nСвойства не выданы заранее.', 12, muted, linespacing=1.7)
    arrow(.278, .79, .338, .79)
    card(.345, .725, .285, .14)
    text(.36, .847, 'СТАЛКЕР', 15, green, 'bold')
    text(.36, .816, 'Просматривает источник\nПроверяет доступный интерфейс\nПодключает новую сущность', 12, linespacing=1.7)
    arrow(.637, .79, .697, .79)
    card(.705, .725, .28, .14)
    text(.72, .847, 'КАТАЛОГ СУЩНОСТЕЙ', 14, weight='bold')
    text(.72, .816, 'Идентификатор + как вызвать\nИсточник + ревизия\nОценок полезности здесь нет', 12, muted, linespacing=1.7)
    text(.025, .702, 'Обнаружения: перед испытаниями комбинаций № 1, 3, 5, 7. По одной сущности за просмотр.', 11, muted)
    arrow(.845, .72, .845, .653)
    text(.55, .676, 'Сведения об интерфейсе → отдельному учёному', 11, purple)
    card(.01, .367, .61, .278, purple)
    text(.028, .627, 'УЧЁНЫЙ-ТЕСТЕР: ПРЕОБРАЗОВАТЕЛЬ …2e691e', 14, purple, 'bold')
    text(.028, .594, 'Вопрос: что он делает с сообщением и искажениями?', 12)
    e = data['example']
    stages = [('Вход', bits(e['experiment']['input'])), ('Кодирование', bits(e['result']['encoded'])),
              ('Портим бит №3', bits(e['result']['received'])), ('Декодирование', bits(e['result']['decoded']))]
    for i, (label, value) in enumerate(stages):
        x = .028 + i * .148
        text(x, .549, label, 10, muted)
        text(x, .522, value, 16, red if i == 2 else green, 'bold', fontfamily='DejaVu Sans Mono')
        if i < 3:
            arrow(x + .113, .511, x + .139, .511)
    text(.028, .477, 'Наблюдение: исходные 1011 восстановлены.', 13, green, 'bold')
    text(.028, .443, 'Проверены все 16 входов длиной 4 бита\nи каждое одиночное искажение: 112 / 112 восстановлений.', 11, linespacing=1.6)
    text(.028, .391, 'Это свойство для проверенного класса входов и ошибок.', 10, muted)
    card(.65, .367, .335, .278, purple)
    text(.67, .627, 'УЧЁНЫЙ-ТЕСТЕР: УЗЕЛ …d1ecc0', 13, purple, 'bold')
    text(.67, .59, 'Проверяет участки по отдельности', 12)
    incoming = data['node']['facts']['incoming_link_observation']['value']
    outgoing = data['node']['facts']['outgoing_link_observation']['value']
    text(.67, .55, f"Источник → узел\n{incoming['mismatches']} / {incoming['bits']} несовпадений", 12, linespacing=1.7)
    text(.67, .483, f"Узел → получатель\n{outgoing['mismatches']} / {outgoing['bits']} несовпадений", 12, linespacing=1.7)
    text(.67, .416, 'На выходе преобладает инверсия.\nВывод ограничен условиями опыта.', 11, green, 'bold', linespacing=1.6)
    arrow(.315, .36, .315, .315, purple)
    arrow(.818, .36, .818, .315, purple)
    card(.01, .16, .975, .145, green)
    text(.028, .287, 'БАЗА ЗНАНИЙ: СВОЙСТВО СО СВИДЕТЕЛЬСТВАМИ', 15, green, 'bold')
    text(.028, .253, 'Что измерено', 11, muted, 'bold')
    text(.028, .228, '4 → 7 бит\nОдиночные ошибки исправлены', 12, linespacing=1.7)
    text(.37, .253, 'Границы применимости', 11, muted, 'bold')
    text(.37, .228, '16 четырёхбитных входов\nОдин испорченный бит за раз', 12, linespacing=1.7)
    text(.705, .253, 'Откуда знаем', 11, muted, 'bold')
    text(.705, .228, 'Сохранены входы и результаты\nИсточник, ревизия, ID опыта', 12, linespacing=1.7)
    text(.015, .128, 'Генератор комбинаций получает только каталог. Оценки и изученные свойства ему не передаются.', 12, weight='bold')
    text(.015, .094, 'Повторная загрузка той же базы: 14 свойств сохранены, 0 новых опытов над компонентами.', 12, green)
    text(.015, .058, 'Сталкер пока работает с локальным провайдером. Семейства опытов учёного заданы программно.', 11, muted)
    text(.015, .031, 'Знания сейчас используются для сохранения, повторного изучения и объяснения; выбором комбинаций они не управляют.', 10, muted)
    fig.savefig('docs/stalker-scientist.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


HTML = r'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Сталкер и учёный-тестер Эдисона</title><style>
*{box-sizing:border-box}body{background:#f4f4ef;color:#172c3b;font:16px/1.55 system-ui,sans-serif;margin:0}main{max-width:1080px;margin:auto;padding:30px 20px 60px}h1{font-size:clamp(28px,5vw,44px);line-height:1.1}h2{font-size:23px}section{background:white;border:1px solid #d9dfda;border-radius:16px;padding:24px;margin:20px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}.box{border:1px solid #d9dfda;border-radius:10px;padding:15px;background:#f9faf7}.green{color:#087d62}.purple{color:#7454bc}.muted{color:#526572}.small{font-size:13px}.flow{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.controls{display:flex;flex-wrap:wrap;gap:15px;margin:20px 0}select{font:inherit;max-width:100%;padding:7px;border:1px solid #d9dfda;border-radius:6px}label{display:flex;gap:6px;flex-direction:column}.bit{display:inline-block;border-radius:5px;background:#edf3f0;padding:5px 9px;margin:2px;font:20px monospace}.changed{background:#f9d7d4;color:#a6352e}.good{background:#dcf3e6;color:#087d62}.metric{font-size:24px;font-weight:700}.callout{background:#fff8e7;border-left:4px solid #d3a33e;padding:12px 16px;margin:18px 0}.raw{overflow-wrap:anywhere;font:12px monospace}.samples{overflow-wrap:anywhere;font:14px monospace;letter-spacing:2px}.timeline{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}button{font:inherit;cursor:pointer;padding:8px 12px;border:1px solid #d9dfda;border-radius:6px;background:white}details{margin-top:15px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px monospace;max-height:300px;overflow:auto}@media(max-width:720px){.grid,.timeline{grid-template-columns:1fr}.bit{padding:4px 7px}section{padding:16px}}
</style><main><p class="green small">ЭДИСОН · ФАКТИЧЕСКИЕ ОПЫТЫ · COMBINED / SEED 0</p>
<h1>Сталкер находит.<br>Учёный проверяет.</h1><p class="muted">На этой странице можно выбрать сохранённый опыт и увидеть его вход, вмешательство и результат. Это воспроизведение журнала, а не новый вызов модели.</p>
<section><h2>1. Сталкер расширяет среду задачи</h2><div class="grid"><div class="box"><b>Локальное инфополе</b><p>Сущности находятся в отдельном источнике. Их свойства не выдаются заранее.</p></div><div class="box"><b class="green">Сталкер</b><p>Просматривает страницы, распознаёт интерфейс, подключает сущность. Известные записи объединяются.</p></div><div class="box"><b>Каталог начинается пустым</b><p>В каталог попадают идентификатор, интерфейс вызова, источник и ревизия. Новые инструменты получают ответвления и у старых конструкций.</p></div></div><h3>Как рос каталог</h3><div id="timeline" class="timeline"></div></section>
<section><h2>2. Учёный проверяет преобразователь</h2><p>Одна гипотеза: после изменения одного передаваемого бита исходное сообщение удастся восстановить. Учёный проверяет каждый четырёхбитный вход и каждую позицию искажения.</p>
<div class="controls"><label>Сущность<select id="transform"></select></label><label>Исходные биты<select id="word"></select></label><label>Вмешательство<select id="flip"></select></label></div>
<div class="flow"><div><b>Вход</b><div id="input"></div></div><span>→</span><div><b>Закодировано</b><div id="encoded"></div></div><span>→</span><div><b>После вмешательства</b><div id="received"></div></div><span>→</span><div><b>Декодировано</b><div id="decoded"></div></div></div>
<p id="result" class="metric"></p><p id="aggregate"></p><p class="small muted">Выбираются только реально выполненные опыты. Одиночная ошибка — изменение одного закодированного бита, без другого шума.</p><details><summary>Свидетельство именно этого опыта</summary><pre id="evidence"></pre></details>
<div class="callout">Сравни две сущности: одна восстанавливает сообщение после одиночного искажения, другая — нет. Название реализации учёному не сообщалось; он сохраняет измеренные свойства.</div></section>
<section><h2>3. Учёный изучает промежуточный узел</h2><p>Проверяются отдельно источник→узел и узел→получатель. Два разных сбалансированных 512-битных шаблона используют независимые потоки шума.</p><div class="controls"><label>Узел<select id="node"></select></label><label>Участок<select id="direction"><option value="incoming">Источник → узел</option><option value="outgoing">Узел → получатель</option></select></label><label>Шаблон<select id="replicate"><option value="0">Чередование 0101…</option><option value="1">256 нулей, затем 256 единиц</option></select></label></div>
<div class="grid"><div class="box"><b>Отправлено, первые 32 бита</b><p id="sent-node" class="samples"></p></div><div class="box"><b>Принято, первые 32 бита</b><p id="received-node" class="samples"></p></div><div class="box"><b>Этот опыт, все 512 бит</b><p id="node-one" class="metric"></p></div></div><p id="node-aggregate"></p><p id="node-ci" class="small muted"></p><div class="callout">Почти все несовпадающие биты могут означать устойчивый переворот сигнала. Учёный фиксирует наблюдение «преобладает инверсия», а не объявляет сущность бесполезной.</div></section>
<section><h2>4. Что остаётся в базе</h2><div class="grid"><div class="box"><b>Свойство</b><p>Измеренное значение: длина выхода, число восстановлений или доля несовпадений.</p></div><div class="box"><b>Область проверки</b><p>Какие входы, размеры сообщений и вмешательства проверялись. Остальные условия остаются неизвестными.</p></div><div class="box"><b>Свидетельства</b><p>Входы, полученные биты, результат, источник, ревизия и идентификаторы опытов.</p></div></div><p id="totals" class="metric green"></p><p>После загрузки той же базы сохранены 14 свойств, новых опытов над компонентами — 0. Совпадать должны сущность, источник и ревизия.</p><div class="callout">Генератор комбинаций получает только каталог. Оценки и база свойств ему не передаются. Сейчас знания используются в сохранении, повторном изучении и объяснении результатов; порядок комбинаций они не меняют.</div><p class="small muted">Это локальный конечный источник. Семейства опытов учёного заданы программно. Интернет-поиск сущностей и самостоятельное изобретение исследовательских протоколов ещё не реализованы.</p></section>
<footer class="small muted">Источник: reports/discovery-knowledge.json и reports/discovery-run.json. Страница автономна, работает без сети и ничего не отправляет в API.</footer></main><script>
const D=__DATA__, $=s=>document.getElementById(s), evidence=D.knowledge.evidence;
const fmt=(x,n=2)=>x.toLocaleString('ru-RU',{minimumFractionDigits:n,maximumFractionDigits:n});
function renderBits(id,bits,changed=[],ok=false){$(id).innerHTML=bits.map((b,i)=>`<span class="bit ${changed.includes(i)?'changed':ok?'good':''}">${b}</span>`).join('')}
function transform(){let t=D.transforms[Number($('transform').value)],word=$('word').value.split('').map(Number),position=Number($('flip').value),positions=position<0?[]:[position];let e=evidence.find(e=>e.entity===t.entity_id&&JSON.stringify(e.experiment.input)===JSON.stringify(word)&&JSON.stringify(e.experiment.flip_positions)===JSON.stringify(positions));if(!e)throw Error('Unknown recorded experiment');let ok=JSON.stringify(e.result.decoded)===JSON.stringify(word);renderBits('input',word);renderBits('encoded',e.result.encoded);renderBits('received',e.result.received,positions);renderBits('decoded',e.result.decoded,[],ok);$('result').textContent=ok?'Исходное сообщение восстановлено':'Исходное сообщение не восстановлено';$('result').style.color=ok?'#087d62':'#b44a43';let f=t.facts.single_bit_flip_recovery.value;$('aggregate').textContent=`Во всей серии одиночных искажений: ${f.recovered} восстановлений из ${f.tested}. Исследование сущности: ${t.probe_count} отдельных опытов.`;$('evidence').textContent=JSON.stringify(e,null,2)}
function resetFlips(){let t=D.transforms[Number($('transform').value)],length=t.facts.encoded_lengths_for_4_bits.value[0];$('flip').innerHTML='<option value="-1">Без искажения</option>'+Array.from({length},(_,i)=>`<option value="${i}">Перевернуть бит №${i+1}</option>`).join('');$('flip').value=String(Math.min(2,length-1));transform()}
function node(){let t=D.nodes[Number($('node').value)],direction=$('direction').value,replicate=Number($('replicate').value),e=evidence.find(e=>e.entity===t.entity_id&&e.experiment.direction===direction&&e.experiment.replicate===replicate);let n=e.experiment.input.reduce((sum,b,i)=>sum+(b!==e.result.received[i]),0),f=t.facts[direction+'_link_observation'].value;$('sent-node').textContent=e.experiment.input.slice(0,32).join('');$('received-node').textContent=e.result.received.slice(0,32).join('');$('node-one').textContent=`${n} / 512 несовпадений`;$('node-aggregate').textContent=`Оба опыта вместе: ${f.mismatches} / ${f.bits} несовпадений (${fmt(f.mismatch_fraction*100)}%). Преобладает ${f.dominant_mapping==='inverted'?'инверсия':f.dominant_mapping==='preserving'?'сохранение сигнала':'неопределённое поведение'}.`;$('node-ci').textContent=`95% интервал Уилсона для доли несовпадений: ${fmt(f.wilson_95[0]*100)}–${fmt(f.wilson_95[1]*100)}%. Другие соседи и условия не проверены.`}
$('timeline').innerHTML=D.entities.map((e,i)=>`<div class="box"><b>Сущность №${i+1}</b><p>${e.kind==='node'?'Узел':'Преобразователь'}<br><span class="raw">…${e.entity_id.slice(-6)}</span></p><p class="small">Подключена перед испытанием комбинации №${e.before_trial}.</p></div>`).join('');
$('transform').innerHTML=D.transforms.map((t,i)=>`<option value="${i}">№${t.number}: 4 → ${t.facts.encoded_lengths_for_4_bits.value[0]} бит (…${t.entity_id.slice(-6)})</option>`).join('');$('word').innerHTML=Array.from({length:16},(_,i)=>`<option>${i.toString(2).padStart(4,'0')}</option>`).join('');$('word').value='1011';$('transform').value=String(D.transforms.findIndex(t=>t.entity_id===D.correcting.entity_id));$('node').innerHTML=D.nodes.map((t,i)=>`<option value="${i}">№${t.number}: …${t.entity_id.slice(-6)}</option>`).join('');$('node').value=String(D.nodes.findIndex(t=>t.entity_id===D.node.entity_id));$('direction').value='outgoing';$('totals').textContent=`${D.entities.length} сущности · ${evidence.length} опытов · ${D.knowledge.facts.length} свойств`;
$('transform').onchange=resetFlips;$('word').onchange=transform;$('flip').onchange=transform;$('node').onchange=node;$('direction').onchange=node;$('replicate').onchange=node;resetFlips();node();
</script></html>'''


if __name__ == '__main__':
    data = load()
    Path('docs').mkdir(exist_ok=True)
    serialized = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')
    Path('docs/stalker-scientist.html').write_text(HTML.replace('__DATA__', serialized))
    png(data)
    print('Built docs/stalker-scientist.png and docs/stalker-scientist.html from recorded evidence.')
