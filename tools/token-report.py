#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目成本核算（通用模板）—— 从本机 Hermes 的 state.db【只读】统计本项目的 token 与估算花费。

用法：
    python token-report.py                       # 屏幕上看明细 + 合计
    python token-report.py --md                  # 输出 markdown 片段（贴 README 用）
    python token-report.py --doc docs/成本账.md    # 直接重写成本账（活文档，含口径与读法）
    python token-report.py --readme README.md      # 刷新 README 标记块里的成本摘要（改 md 时顺手跑）
    # 临时核算别的项目（不改 CONFIG）：
    python token-report.py --label "别的项目" --project D:/other \
        --session 20260925_1634 --keyword other --partial 20260925_1238 --partial-hint 悬浮

归属规则（A 全额 / B 按消息行数折算）与口径提醒见 docs/成本账.md —— 那份文档就由本脚本 --doc 生成。
"""
import sqlite3
import sys
import datetime
import io

CONFIG = {
    'db': 'D:/insane-robot/state.db',
    'label': 'empirical-kit',
    'project_dir': 'D:/empirical-kit',
    'main_sessions': ['local-6299a52', 'local-5cb468a2'],   # 上一轮规划 + 本轮补齐
    'keywords': ['empirical-kit', 'epkit', '实证工具'],
    'partial_sessions': [],     # 例：['20260925_1238']
    'partial_hints': [],        # 例：['拉黑', '悬浮']（不给则用 keywords）
}


def apply_cli(cfg):
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        i += 1
        if a == '--md':
            cfg['md'] = True
            continue
        if i >= len(args):
            break
        v = args[i]
        i += 1
        if a == '--db':
            cfg['db'] = v
        elif a == '--label':
            cfg['label'] = v
        elif a == '--project':
            cfg['project_dir'] = v
        elif a == '--session':
            cfg['main_sessions'].append(v)
        elif a == '--keyword':
            cfg['keywords'].append(v)
        elif a == '--partial':
            cfg['partial_sessions'].append(v)
        elif a == '--partial-hint':
            cfg['partial_hints'].append(v)
        elif a == '--doc':
            cfg['doc'] = v
        elif a == '--readme':
            cfg['readme'] = v
    cfg.setdefault('md', False)
    cfg.setdefault('doc', None)
    cfg.setdefault('readme', None)
    return cfg


CFG = apply_cli(dict(CONFIG))
CFG['partial_hints'] = CFG['partial_hints'] or CFG['keywords']

con = sqlite3.connect('file:%s?mode=ro' % CFG['db'], uri=True)
cur = con.cursor()

SESS_COLS = '''id, source, title, started_at, message_count, tool_call_count, input_tokens, output_tokens,
               cache_read_tokens, reasoning_tokens, estimated_cost_usd, cost_status, cwd'''


def bj(ts):
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M') if ts else '-'


def sess(sid_or_prefix):
    if len(sid_or_prefix) < 32:
        r = cur.execute('select %s from sessions where id like ?' % SESS_COLS, (sid_or_prefix + '%',)).fetchall()
        return r[0] if len(r) == 1 else (r[0] if r else None)
    return cur.execute('select %s from sessions where id = ?' % SESS_COLS, (sid_or_prefix,)).fetchone()


def first_user_msg(sid):
    r = cur.execute("select content from messages where session_id=? and role='user' order by rowid limit 1", (sid,)).fetchone()
    return str(r[0]) if r and r[0] else ''


def hit_lines(sid, hints):
    if not hints:
        return 0, 0
    tot = cur.execute('select count(*) from messages where session_id=?', (sid,)).fetchone()[0]
    like = ' or '.join(['content like ?'] * len(hints))
    hit = cur.execute('select count(*) from messages where session_id=? and (%s)' % like,
                      (sid,) + tuple('%%%s%%' % h for h in hints)).fetchone()[0]
    return hit, tot


def matched(r):
    return any(k and k in (r[2] or '') for k in CFG['keywords']) or any(k and k in first_user_msg(r[0]) for k in CFG['keywords'])


rows = []          # (label, row, weight)
seen = set()

# A① cwd 命中
if CFG['project_dir']:
    for r in cur.execute('select %s from sessions where cwd like ?' % SESS_COLS, ('%%%s%%' % CFG['project_dir'],)).fetchall():
        if r[0] not in seen:
            seen.add(r[0]); rows.append(('cwd 命中项目目录', r, 1.0))
# A② 指定的主会话
for p in CFG['main_sessions']:
    r = sess(p)
    if r and r[0] not in seen:
        seen.add(r[0]); rows.append(('指定主会话', r, 1.0))
# A③ 相关 subagent（独立审查）
for (sid,) in cur.execute("select id from sessions where source='subagent'").fetchall():
    if sid in seen:
        continue
    f = first_user_msg(sid)
    if any(k and k in f for k in CFG['keywords']):
        r = sess(sid)
        if r:
            seen.add(r[0]); rows.append(('子 agent（独立审查等）', r, 1.0))
# A④ 其余关键词命中的会话（标题或首条消息）
if CFG['keywords']:
    for r in cur.execute('select %s from sessions' % SESS_COLS).fetchall():
        if r[0] in seen:
            continue
        if matched(r):
            seen.add(r[0]); rows.append(('关键词命中', r, 1.0))
# B 部分相关
for p in CFG['partial_sessions']:
    r = sess(p)
    if not r or r[0] in seen:
        continue
    seen.add(r[0])
    hit, tot = hit_lines(r[0], CFG['partial_hints'])
    rows.append(('部分相关（按消息行数折算）', r, (hit / float(tot)) if tot else 0.0))
    CFG.setdefault('_share_note', []).append((hit, tot))

rows.sort(key=lambda x: x[1][3])

T = dict(msg=0, tool=0, i=0, o=0, cr=0, rz=0, cost=0.0)
table, plain = [], []
for label, r, w in rows:
    (sid, source, title, st, mc, tc, i, o, cr, rz, cost, status, cwd) = r
    mc, tc = int((mc or 0) * w), int((tc or 0) * w)
    i, o, cr, rz = int((i or 0) * w), int((o or 0) * w), int((cr or 0) * w), int((rz or 0) * w)
    cost = (cost or 0.0) * w
    T['msg'] += mc; T['tool'] += tc; T['i'] += i; T['o'] += o; T['cr'] += cr; T['rz'] += rz; T['cost'] += cost
    note = '全额' if w == 1.0 else '折算 %.0f%%' % (w * 100)
    plain.append('%-13s | %-9s | %s | 消息 %-4s 工具 %-4s | 输入 %-9s 输出 %-8s 缓存读 %-10s 推理 %-8s | $%.4f | %s'
                 % (sid[:13], source, bj(st), mc, tc, i, o, cr, rz, cost, note))
    table.append('| %s（`%s…`）| %s | %s | %s | %s | %s | %s | %s | %.4f | %s |'
                 % (label, sid[:13], bj(st)[5:], format(mc, ','), format(tc, ','), format(i, ','),
                    format(o, ','), format(cr, ','), format(rz, ','), cost, note))

reason_note = '推理可能已含在输出口径里'
now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
total4 = T['i'] + T['o'] + T['cr'] + T['rz']
cny = T['cost'] * 7.1

# 参照：当天全部会话
today = datetime.datetime.now().strftime('%Y-%m-%d')
t0 = int(datetime.datetime.strptime(today, '%Y-%m-%d').timestamp())
day = cur.execute('''select count(*), sum(coalesce(message_count,0)), sum(coalesce(tool_call_count,0)),
                     sum(coalesce(estimated_cost_usd,0)) from sessions
                     where coalesce(last_activity_at, started_at) >= ?''', (t0,)).fetchone()

md_body = '| 会话 | 开始 | 消息 | 工具调用 | 输入 | 输出 | 缓存读 | 推理 | 估算(USD) | 计入 |\n|---|---|---|---|---|---|---|---|---|---|\n%s\n\n' % '\n'.join(table)
md_body += '- 消息 **%s** · 工具调用 **%s**\n' % (format(T['msg'], ','), format(T['tool'], ','))
md_body += '- 输入 **%s** · 输出 **%s** · 缓存读 **%s** · 推理 **%s**\n' % (
    format(T['i'], ','), format(T['o'], ','), format(T['cr'], ','), format(T['rz'], ','))
md_body += '- 四项相加 = **%s**（%s，别当独立增量）\n' % (format(total4, ','), reason_note)
md_body += '- 估算花费 **$%.4f**（按 1 USD≈7.1 粗算约 %.0f 元人民币）\n' % (T['cost'], cny)

# README 用的摘要块（贴进标记区，由 --readme 自动维护）
def _cat(keys):
    return sum(r[2] * (r[1][10] or 0.0) for r in rows if any(k in r[0] for k in keys))

_cost_main = _cat(['主会话', 'cwd 命中', '关键词命中'])
_cost_sub = _cat(['子 agent'])
_cost_part = _cat(['部分相关'])
readme_body = ('- 截至 %s（北京时间）：估算花费 **$%.4f**（≈%.0f 元人民币）· 消息 %s · 工具调用 %s\n'
               '- 结构：主开发会话 $%.2f ／ 子 agent 独立审查 $%.2f ／ 部分相关折算 $%.2f（明细见 [`docs/成本账.md`](docs/成本账.md)）\n'
               '- 口径：`estimated_cost_usd` 是**估算不是账单**；`reasoning_tokens` 通常已含在输出口径里；缓存读占 ~98%%，所以「总 token 近亿」不等于贵。\n'
               '- 复现：`python token-report.py`（屏幕）· `--doc docs/成本账.md`（重写成本账）· `--readme README.md`（刷新本段）'
               % (now_str, T['cost'], cny, format(T['msg'], ','), format(T['tool'], ','),
                  _cost_main, _cost_sub, _cost_part))

READ_BEGIN = '<!-- COST:BEGIN 由 tools/token-report.py --readme 生成，别手改 -->'
READ_END = '<!-- COST:END -->'

if CFG['readme']:
    txt = io.open(CFG['readme'], encoding='utf-8').read()
    assert READ_BEGIN in txt and READ_END in txt, '目标文件里缺成本标记块（先手工放一对标记）'
    i, j = txt.index(READ_BEGIN), txt.index(READ_END) + len(READ_END)
    io.open(CFG['readme'], 'w', encoding='utf-8').write(txt[:i] + READ_BEGIN + '\n' + readme_body + '\n' + READ_END + txt[j:])
    print('已刷新 %s 的成本摘要块 · 合计 $%.4f' % (CFG['readme'], T['cost']))

if CFG['doc']:
    share_rows = '\n'.join(['- `%s…`：%d/%d 行命中关键词 → 折算 %.0f%%' % (CFG['partial_sessions'][i], h, t, 100.0 * h / t)
                            for i, (h, t) in enumerate(CFG.get('_share_note', []))]) or '- （无部分相关会话）'
    doc = u'''# 成本账：本项目花了多少 token / 多少钱

**生成时间**：%s（北京时间；由 `tools/token-report.py` 直查库生成，不手抄）
**数据来源**：本机 Hermes 会话库 `%s`（**只读**查询）
**复现**：`python token-report.py`（屏幕明细）· `--md`（markdown 片段）· `--doc <路径>`（重写本文件）

## 口径

1. **全额计入**：本项目主会话（`%s`）+ `source=subagent` 且首条用户消息提到本项目的子 agent 会话（独立审查）+ cwd 命中项目目录 / 标题关键词命中的会话。
2. **按占比折算**：跨话题会话（手机端那场里夹杂别的任务）按含关键词的**消息行数占比**折算 —— 粗略近似，仅供横向比较：
%s
3. `estimated_cost_usd` 是 Hermes 按当时价目表算的**估算值，不是账单**；`reasoning_tokens` 通常已含在输出口径里，故四项分开列、不重复相加。

## 明细

%s
## 合计

- 消息 **%s** · 工具调用 **%s**
- 输入 **%s** · 输出 **%s** · 缓存读 **%s** · 推理 **%s**
- 四项相加 = **%s**（%s，别当独立增量）
- 估算花费 **$%.4f**（按 1 USD≈7.1 粗算约 %.0f 元人民币）

## 怎么读这些数

- **缓存读通常占 ~98%%**：每轮都要带上长上下文，命中缓存的部分单价远低于新输入 —— 所以「总 token 近亿」不等于「很贵」。
- **子 agent 是独立会话**、token 单独记账，已并入合计；实测每轮独立审查 $0.04–0.06，是主会话的十分之一量级。
- **主会话还在跑**，数字会涨；要当时准确值就重跑 `python token-report.py`。
- 真实花费以 provider 后台账单为准；本表用于比较**各阶段开销结构**（读规范 / 写代码 / 测试 / 审查各占多少）。

## 参照：当天（%s）全部会话

当天 %d 场会话合计：消息 %s · 工具调用 %s · 估算 $%.4f。
''' % (now_str, CFG['db'], (CFG['main_sessions'][0] + '…') if CFG['main_sessions'] else '（未指定）', share_rows,
       md_body, format(T['msg'], ','), format(T['tool'], ','),
       format(T['i'], ','), format(T['o'], ','), format(T['cr'], ','), format(T['rz'], ','),
       format(total4, ','), reason_note, T['cost'], cny,
       today, day[0], format(day[1] or 0, ','), format(day[2] or 0, ','), day[3] or 0.0)
    io.open(CFG['doc'], 'w', encoding='utf-8').write(doc)
    print('已重写 %s（%d 字节）· 合计 $%.4f' % (CFG['doc'], len(doc.encode('utf-8')), T['cost']))
elif CFG['md']:
    print(md_body)
else:
    print('== %s · 成本核算（来源：%s，只读）==' % (CFG['label'], CFG['db']))
    print('\n'.join(plain))
    print()
    print('合计：消息 %s · 工具调用 %s' % (format(T['msg'], ','), format(T['tool'], ',')))
    print('      输入 %s · 输出 %s · 缓存读 %s · 推理 %s'
          % (format(T['i'], ','), format(T['o'], ','), format(T['cr'], ','), format(T['rz'], ',')))
    print('      四项相加 %s（%s）· 估算花费 $%.4f' % (format(total4, ','), reason_note, T['cost']))
    for hit, tot in CFG.get('_share_note', []):
        print('      部分相关折算依据：%d/%d 行命中关键词' % (hit, tot))
