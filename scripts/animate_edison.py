"""Render the Russian Edison explainer with Pillow and FFmpeg; no model calls."""
import argparse
import json
import math
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS = 1280, 720, 24
BG = '#0b1220'
PANEL = '#152237'
EDGE = '#2a3d56'
WHITE = '#eff5fc'
MUTED = '#a0b2ca'
CYAN = '#58d8ee'
GREEN = '#65e1b0'
AMBER = '#ffca78'
PURPLE = '#b5a0ff'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONTS = {}
SCENES = [
    (6, 'ЭДИСОН', 'Как система обнаруживает свойства'),
    (8, '01 / СТАЛКЕР', 'Найти то, что можно исследовать'),
    (8, '02 / УЧЁНЫЙ', 'Не одно объяснение, а конкурирующие гипотезы'),
    (8, '03 / КОНТРАКТ', 'Превратить вопрос в исполнимый опыт'),
    (8, '04 / ИСПОЛНИТЕЛЬ', 'Ответ даёт среда'),
    (9, '05 / ПАМЯТЬ', 'Измерение и объяснение хранятся отдельно'),
    (9, '06 / НЕЗАВИСИМОСТЬ', 'Два контура с разными контекстами'),
    (8, '07 / ВЕТВЛЕНИЕ', 'Неудача не стирает остальные пути'),
    (10, '08 / ЖИВАЯ ПРОВЕРКА', 'Что уже получилось на самом деле'),
    (6, 'ЭДИСОН / СЕЙЧАС', 'Вопросы выбирает LLM. Измерения даёт среда.'),
]


def font(size, bold=False):
    key = (size, bold)
    if key not in FONTS:
        FONTS[key] = ImageFont.truetype(BOLD if bold else FONT, size)
    return FONTS[key]


def ease(x):
    x = max(0, min(1, x))
    return x*x*(3-2*x)


def txt(d, xy, text, size=24, color=WHITE, bold=False):
    d.text(xy, text, font=font(size, bold), fill=color)


def wrap(d, text, x, y, width, size=22, color=MUTED, gap=9, bold=False):
    words = text.split()
    line = ''
    for word in words:
        trial = (line+' '+word).strip()
        if d.textlength(trial, font=font(size, bold)) > width and line:
            txt(d, (x,y), line, size, color, bold)
            y += size+gap
            line = word
        else:
            line = trial
    if line:
        txt(d, (x,y), line, size, color, bold)
    return y+size+gap


def box(d, rect, title, body='', accent=CYAN, title_size=25):
    x,y,r,b = rect
    d.rounded_rectangle(rect, radius=18, fill=PANEL, outline=EDGE, width=2)
    d.rounded_rectangle((x+1,y+18,x+5,b-18), radius=2, fill=accent)
    txt(d, (x+22,y+19), title, title_size, accent, True)
    if body:
        wrap(d, body, x+22, y+60, r-x-42, 21)


def arrow(d, a, b, progress=1, color=CYAN, dot=0):
    d.line([a,b], fill=EDGE, width=3)
    ex=a[0]+(b[0]-a[0])*progress
    ey=a[1]+(b[1]-a[1])*progress
    d.line([a,(ex,ey)], fill=color, width=3)
    angle=math.atan2(b[1]-a[1], b[0]-a[0])
    if progress>.98:
        points=[b, (b[0]-12*math.cos(angle-.45),b[1]-12*math.sin(angle-.45)),
                (b[0]-12*math.cos(angle+.45),b[1]-12*math.sin(angle+.45))]
        d.polygon(points,fill=color)
    if dot:
        q=dot%1
        px=a[0]+(b[0]-a[0])*q
        py=a[1]+(b[1]-a[1])*q
        d.ellipse((px-6,py-6,px+6,py+6),fill=color)


def caption(d, text, small=''):
    d.rounded_rectangle((48,584,1232,672),radius=16,fill='#111d2e')
    wrap(d,text,70,601,1140,23,WHITE,gap=5)
    if small:
        txt(d,(70,642),small,15,MUTED)


def frame(scene, t, duration, absolute):
    im=Image.new('RGB',(W,H),BG)
    d=ImageDraw.Draw(im)
    for x in range(48,W,48):
        for y in range(40,580,48):
            d.ellipse((x,y,x+1,y+1),fill='#1a2a40')
    label,title=SCENES[scene][1:]
    txt(d,(48,28),label,17,CYAN,True)
    txt(d,(48,68),title,34,WHITE,True)
    txt(d,(1132,30),f'{scene+1:02d} / 10',16,MUTED)
    p=ease(t/1.5)
    flow=t*.3
    if scene==0:
        labels=[('Инфополе','Доступные сущности'),('Сталкер','Находит и подключает'),
                ('Учёный','Ставит вопросы'),('Опыт','Получает измерения'),('Память','Сохраняет свидетельства')]
        for i,(name,body) in enumerate(labels):
            x=48+i*244
            box(d,(x,244,x+210,416),name,body,CYAN if i<3 else GREEN,23)
            if i<4:
                arrow(d,(x+214,325),(x+240,325),ease((t-i*.3)/.7),dot=flow)
        txt(d,(48,155),'СВОБОДНЫЙ ВОПРОС → ПРОВЕРЯЕМЫЙ ОПЫТ',23,PURPLE,True)
        caption(d,'LLM предлагает объяснение. Оно должно выдержать встречу со средой.')
    elif scene==1:
        d.rounded_rectangle((48,154,695,546),radius=20,outline=EDGE,width=2,fill='#101d30')
        txt(d,(72,174),'ИНФОПОЛЕ',18,MUTED,True)
        for i in range(9):
            x=78+(i%3)*192; y=232+(i//3)*92
            selected=i==4
            d.rounded_rectangle((x,y,x+160,y+62),radius=12,fill=PANEL,
                                outline=GREEN if selected else EDGE,width=3 if selected else 1)
            txt(d,(x+15,y+17),'сущность '+str(i+1),17,GREEN if selected else MUTED)
        sx=70+int((t/5%1)*570)
        d.line((sx,216,sx,526),fill=CYAN,width=2)
        box(d,(780,217,1232,420),'Новая сущность',
            'Известен интерфейс: encode / decode. Внутренний механизм скрыт.',GREEN)
        arrow(d,(686,351),(776,318),p,dot=flow)
        txt(d,(804,454),'СТАЛКЕР НЕ ОЦЕНИВАЕТ ПОЛЬЗУ',18,CYAN,True)
        caption(d,'Сначала система узнаёт, что можно вызвать и как с этим взаимодействовать.',
                'Сейчас источник Сталкера — локальное инфополе, не весь интернет.')
    elif scene==2:
        box(d,(48,236,415,460),'Чёрный ящик','Можно передать биты и получить ответ. Что происходит внутри?',CYAN)
        hs=[('Гипотеза A','Копирует вход без изменений.'),('Гипотеза B','Добавляет избыточные биты.'),
            ('Гипотеза C','Восстанавливает данные после повреждения.')]
        for i,(name,body) in enumerate(hs):
            y=159+i*130
            box(d,(560,y,1232,y+113),name,body,PURPLE,23)
            arrow(d,(420,345),(553,y+55),ease((t-i*.35)/1.2),PURPLE,flow+i*.2)
        caption(d,'Какая проба даст разный результат при разных объяснениях?',
                'Иллюстрация вопроса. Гипотезы и входы выбирает LLM; интерфейс задаёт адаптер.')
    elif scene==3:
        box(d,(48,160,560,529),'Структурированный ответ','',PURPLE)
        lines=['{','  "question": "Что изменится?",','  "hypotheses": [ ... ],',
               '  "experiment": "Python-код",','  "interpretations": [ ... ]','}']
        for i,line in enumerate(lines):
            txt(d,(76,239+i*37),line,22,WHITE if i!=3 else GREEN)
        arrow(d,(566,342),(660,342),p,dot=flow)
        checks=[('JSON Schema','Поля и типы ответа'),('Синтаксис Python','Код можно разобрать'),
                ('Ссылки на измерения','Нельзя сослаться на выдуманный опыт')]
        for i,(name,body) in enumerate(checks):
            y=160+i*124
            box(d,(674,y,1232,y+109),name,body,GREEN,23)
        caption(d,'Ошибка формата → точная диагностика → одна попытка исправления.',
                'Корректная структура ещё не доказывает ни правильность программы, ни истинность гипотезы.')
    elif scene==4:
        box(d,(48,235,345,470),'Вопрос модели','Вызвать неизвестную функцию на [2, 3].',PURPLE)
        box(d,(477,190,808,505),'Изолированный опыт','Python вызывает инструмент среды. Без сети, ключей и файлов проекта.',CYAN)
        box(d,(937,235,1232,470),'Ответ среды','Вход: [2, 3] Результат: 13',GREEN)
        arrow(d,(350,345),(470,345),p,PURPLE,flow)
        arrow(d,(814,345),(931,345),ease((t-.6)/1.3),GREEN,flow)
        txt(d,(502,441),'ИЗМЕРЕНИЕ ЗАПИСЫВАЕТ',16,GREEN,True)
        txt(d,(502,466),'ДОВЕРЕННЫЙ ИНСТРУМЕНТ',16,GREEN,True)
        caption(d,'LLM не может заменить реальное измерение своим рассказом.',
                'Пример [2, 3] → 13 взят из живого журнала функции v3.')
    elif scene==5:
        box(d,(48,167,605,382),'НАБЛЮДЕНИЕ','На вход [2, 3] инструмент вернул 13. Это результат конкретного записанного опыта.',GREEN)
        box(d,(675,167,1232,382),'ИНТЕРПРЕТАЦИЯ','Возможно, это сумма квадратов: 2² + 3² = 13. На других входах ещё нужно проверять.',AMBER)
        txt(d,(430,423),'ИЗМЕРЕНИЕ ≠ ЗАКОН',30,WHITE,True)
        arrow(d,(1195,411),(1195,535),p,AMBER,flow)
        arrow(d,(1195,535),(78,535),p,CYAN,flow*.5)
        txt(d,(125,495),'Новый опыт ищет контрпример → объяснение пересматривается',23,CYAN)
        caption(d,'Память хранит исходные ответы, область проверки и непроверенные объяснения.',
                'Точный повтор запроса берётся из памяти; независимой проверки обобщений пока нет.')
    elif scene==6:
        txt(d,(50,151),'КОНТУР ИССЛЕДОВАНИЯ СВОЙСТВ',18,CYAN,True)
        for rect,name,body in [((48,194,360,325),'Учёный','Выбирает гипотезы'),
                               ((484,194,796,325),'Опыт','Возвращает измерения'),
                               ((920,194,1232,325),'Память','Хранит свидетельства')]:
            box(d,rect,name,body,CYAN,24)
        arrow(d,(366,260),(478,260),p,CYAN,flow)
        arrow(d,(802,260),(914,260),p,GREEN,flow)
        d.line((48,365,1232,365),fill=EDGE,width=2)
        txt(d,(50,383),'КОНТУР ПОИСКА КОМБИНАЦИЙ',18,PURPLE,True)
        for rect,name,body in [((48,425,360,551),'Генератор','Предлагает сочетания'),
                               ((484,425,796,551),'Ветви','Сохраняют варианты'),
                               ((920,425,1232,551),'Испытатель','Измеряет результат')]:
            box(d,rect,name,body,PURPLE,24)
        arrow(d,(366,488),(478,488),p,PURPLE,flow)
        arrow(d,(802,488),(914,488),p,PURPLE,flow)
        caption(d,'Генератор не получает историю опытов, накопленные свойства и оценки.',
                'Доступные примитивы общие; исследовательские контексты разделены. Планировщика-агента нет.')
    elif scene==7:
        points=[(100,350),(335,236),(335,463),(600,183),(600,302),(600,419),(600,527),
                (874,155),(874,223),(874,296),(874,365),(874,444),(874,521),
                (1170,172),(1170,254),(1170,401),(1170,513)]
        links=[(0,1),(0,2),(1,3),(1,4),(2,5),(2,6),(3,7),(3,8),(4,9),(4,10),
               (5,11),(6,12),(7,13),(9,14),(11,15),(12,16)]
        for j,(a,b) in enumerate(links):
            arrow(d,points[a],points[b],ease((t-j*.12)/1.0),GREEN if j%3 else PURPLE)
        for j,(x,y) in enumerate(points):
            if t>j*.1:
                d.ellipse((x-13,y-13,x+13,y+13),fill=AMBER if j==4 else GREEN,outline=BG,width=3)
        txt(d,(386,328),'неудачный опыт',17,AMBER)
        txt(d,(48,153),'ВЕТВИ ОСТАЮТСЯ В ОЧЕРЕДИ',18,CYAN,True)
        caption(d,'Неудача не вычёркивает ветвь. Ответвления растут и от прежних узлов.',
                'Чередование предложений LLM и очереди FIFO сохраняет широту в заданной грамматике.')
    elif scene==8:
        data=json.loads((ROOT/'reports/research-v3-summary.json').read_text())
        names=['Функция','Битовый код','Физика']
        for i,(row,name) in enumerate(zip(data,names)):
            x=48+i*403
            box(d,(x,169,x+378,446),name,'',CYAN if i!=1 else GREEN)
            n=round(row['observations']*ease(t/2.2))
            txt(d,(x+24,240),str(n),64,WHITE,True)
            txt(d,(x+24,320),'измерений',20,MUTED)
            txt(d,(x+24,381),'Все этапы прошли' if i==1 else 'Обрыв 2-го раунда',20,GREEN if i==1 else AMBER,True)
        txt(d,(48,473),'38 / 38',38,GREEN,True)
        txt(d,(267,484),'ответов совпали при воспроизведении',26,WHITE)
        caption(d,'Все три обзора завершены. Два ответа оборвались даже на 8192 токенах.',
                'Это результат проверки v3. Воспроизведение измерений не доказывает истинность объяснений.')
    else:
        txt(d,(48,175),'НАЙТИ → СПРОСИТЬ → ИЗМЕРИТЬ → ПЕРЕСМОТРЕТЬ',29,CYAN,True)
        box(d,(48,253,605,490),'УЖЕ РАБОТАЕТ','LLM выбирает вопросы и пишет исполняемые опыты. Измерения имеют проверяемое происхождение.',GREEN)
        box(d,(675,253,1232,490),'ЕЩЁ НУЖНО ДОВЕСТИ','Обрывы длинных ответов и независимую проверку новых прогнозов. Универсальность не доказана.',AMBER)
        caption(d,'Мы задаём инструменты. Учёный выбирает, что выяснять.',
                'EdisonResearch • схема текущего прототипа • без утверждения о доказанной zero-shot способности')
    # Progress rail and scene dots remain stable through transitions.
    total=sum(s[0] for s in SCENES)
    d.rectangle((0,710,W,719),fill=EDGE)
    d.rectangle((0,710,int(W*absolute/total),719),fill=CYAN)
    for i in range(10):
        x=543+i*21
        d.ellipse((x,688,x+7,695),fill=CYAN if i==scene else EDGE)
    alpha=min(ease(t/.45),ease((duration-t)/.45))
    if alpha<1:
        im=Image.blend(Image.new('RGB',(W,H),BG),im,alpha)
    return im


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args()
    out=ROOT/'docs'
    times=[]; base=0
    for duration,*_ in SCENES:
        times.append(base); base+=duration
    # Contact sheet used to inspect all scenes before/after encoding.
    sheet=Image.new('RGB',(1280,360*5),BG)
    for i,(duration,*_) in enumerate(SCENES):
        tile=frame(i,duration*.55,duration,times[i]+duration*.55).resize((640,360))
        sheet.paste(tile,((i%2)*640,(i//2)*360))
    sheet.save(out/'edison-animation-storyboard.jpg',quality=90)
    frame(0,3,6,3).save(out/'edison-animation-poster.jpg',quality=94)
    if args.preview:
        return
    output=out/'edison-explained.mp4'
    command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
             '-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264',
             '-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for i,(duration,*_) in enumerate(SCENES):
            for number in range(duration*FPS):
                process.stdin.write(frame(i,number/FPS,duration,times[i]+number/FPS).tobytes())
            print(f'Scene {i+1}/10 rendered',flush=True)
    finally:
        process.stdin.close()
    if process.wait()!=0:
        raise SystemExit('FFmpeg failed')
    print(output,flush=True)


if __name__=='__main__':
    main()
