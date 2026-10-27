"""Compare generation, measured token use and saved test paths; no model calls."""
import argparse
from collections import Counter
from datetime import datetime
from html import escape
import json
from pathlib import Path
import re
import sqlite3

import numpy as np
import pandas as pd

from quantscale.evaluation import digest
from .build_html_report import table, section, fmt, pct
from .report_topologies import topology_html

ORDER = ['single', 'independent', 'centralized', 'decentralized', 'hybrid']
NAMES = dict(single='单 Agent', independent='独立并行', centralized='中心化', decentralized='去中心化', hybrid='混合')


def completion_chart(old, new):
    elements=[]
    y=lambda v:242-v*205
    for value in [0,.25,.5,.75,1]:
        elements.append(f'<line x1="54" y1="{y(value)}" x2="876" y2="{y(value)}" stroke="#dce3e7"/><text x="44" y="{y(value)+4}" text-anchor="end" font-size="12" fill="#65727d">{value*100:.0f}%</text>')
    for i,(a,b) in enumerate(zip(old['aggregates'],new['aggregates'])):
        x=136+i*163
        for row,dx,color in [(a,-20,'#9aabb2'),(b,20,'#176275')]:
            rate=row['valid_rate']
            if rate is None:
                elements.append(f'<text x="{x+dx}" y="230" font-size="10" text-anchor="middle">无尝试</text>')
                continue
            elements.append(f'<rect x="{x+dx-14}" y="{y(rate)}" width="28" height="{rate*205}" fill="{color}"><title>{escape(row["label"])}：{row["valid"]}/{row["attempts"]}，完成率{pct(rate)}</title></rect>')
            elements.append(f'<text x="{x+dx}" y="{y(rate)-7}" font-size="11" text-anchor="middle" fill="#41545f">{rate*100:.0f}%</text>')
        elements.append(f'<text x="{x}" y="270" font-size="13" text-anchor="middle" fill="#41545f">{a["label"]}</text>')
    return '<div class="chart-box"><h3>各结构的训练与验证完成率</h3><p class="chart-label">横轴：结构 · 纵轴：完成训练与验证的候选 / 尝试数，百分比</p><svg viewBox="0 0 900 290" role="img" aria-label="修复前后各结构有效候选完成率">'+''.join(elements)+'</svg><div class="legend"><span><i class="gray"></i>V2 修复前</span><span><i></i>V3 修复后</span></div><p class="source">来源：全部研究轨迹；完成回测不保证有持仓或盈利，相关数量在表中另列。</p></div>'


def collect(root):
    root = Path(root)
    completed = json.loads((root / 'completed.json').read_text())
    protocol = json.loads((root / 'pilot-protocol.json').read_text())
    roster = json.loads((root / 'all-strategies-frozen.json').read_text())
    evaluation_errors = {x['run']: x['error'] for x in completed['evaluation_failures']}
    records = []
    fingerprints = set()
    for item in sorted(protocol['schedule'], key=lambda x: (ORDER.index(x['topology']), x['seed'])):
        name = '%s-n%d-s%d' % (item['topology'], item['fleet_size'], item['seed'])
        folder = root / name
        events = [json.loads(x) for x in (folder / 'trajectory.jsonl').read_text().splitlines()]
        trials = [x for x in events if x['kind'] == 'trial']
        responses = [x for x in events if x['kind'] == 'response']
        researchers = [x for x in responses if x['role'] == 'researcher']
        usage = [x['response'].get('_provenance', {}).get('usage', {}) for x in responses]
        pending = []
        for event in events:
            if event['kind'] == 'request':
                pending.append(event['request'])
            elif event['kind'] == 'response':
                assert pending
                pending.pop(0)
            elif (event['kind']=='budget_exhausted' and pending
                  and 'conservative DeepSeek input reservation' in event.get('reason','')):
                # The adapter rejected this request locally before HTTP.
                pending.pop()
        code_ids = [x['strategy_id'] for x in trials if 'strategy_id' in x]
        valid = [x for x in trials if 'validation' in x]
        tokens = sum(x['tokens'] for x in responses)
        assert len(trials) <= protocol['config']['backtests']
        for response in responses:
            p = response['response'].get('_provenance', {})
            fingerprints.add((str(p.get('served_model')), str(p.get('system_fingerprint'))))
        aborted = [x['error'] for x in events if x['kind'] == 'aborted']
        row = dict(**item, run=name, label=NAMES[item['topology']]+' · '+str(item['seed']),
                   attempts=len(trials), researcher_calls=len(researchers), calls=len(responses),
                   submitted=sum(isinstance(x['response'].get('strategy_code'), str) and bool(x['response']['strategy_code'].strip()) for x in researchers),
                   accepted_code=len(code_ids), valid=len(valid), unique_valid=len({x['strategy_id'] for x in valid}),
                   active_valid=sum(x['train'].get('active_days',0)>0 and x['validation'].get('active_days',0)>0 for x in valid),
                   cash_validation=sum(x['validation'].get('active_days',0)==0 for x in valid),
                   unique_code=len(set(code_ids)), tokens=tokens,
                   input_tokens=sum(x.get('prompt_tokens', 0) for x in usage),
                   output_tokens=sum(x.get('completion_tokens', 0) for x in usage),
                   reasoning_tokens=sum(x.get('completion_tokens_details', {}).get('reasoning_tokens', 0) for x in usage),
                   cached_input_tokens=sum(x.get('prompt_cache_hit_tokens', 0) for x in usage),
                   input_reservation_total=sum(x['response'].get('_provenance',{}).get('input_reservation',0) for x in responses),
                   manager_tokens=sum(x['tokens'] for x in responses if x['role']=='manager'),
                   repair_calls=sum(x['kind']=='request' and 'format_repair' in x['request'] for x in events),
                   truncated=sum('did not finish: length' in x['response'].get('adapter_error','') for x in researchers),
                   truncated_output_caps=[x['response'].get('_provenance',{}).get('applied_max_output_tokens',
                                          x['response'].get('_provenance',{}).get('usage',{}).get('completion_tokens'))
                                          for x in researchers if 'did not finish: length' in x['response'].get('adapter_error','')],
                   normalized_broadcasts=sum(x['kind']=='manager_broadcast' and bool(x.get('normalization')) for x in events),
                   manager_failed_rounds=sum(x['kind']=='manager_round_failed' for x in events),
                   execution_errors=dict(Counter(x['error'] for x in trials if 'error' in x)),
                   unconfirmed_requests=[dict(role=q['role'], max_total_tokens=q['max_total_tokens']) for q in pending],
                   status='no_submission', failure='; '.join(aborted))
        row['budget_stop_reason'] = '; '.join(x['reason'] for x in events if x['kind']=='budget_exhausted')
        row['token_budget_breached'] = tokens > protocol['config']['tokens']
        row['manager_terminal_failure'] = (item['topology'] in ('centralized','hybrid')
            and ('directions' in row['failure'] or 'Manager must return' in row['failure']
                 or any(x['kind']=='manager_protocol_exhausted' for x in events)))
        if (folder / 'protocol.json').exists():
            row['backend_configuration'] = json.loads((folder / 'protocol.json').read_text()).get('backend_configuration', {})
        if (folder / 'selection.json').exists():
            selection = json.loads((folder / 'selection.json').read_text())
            row.update(strategy_id=selection['strategy_id'], hypothesis=selection['hypothesis'],
                       strategy_source=selection['strategy']['source'], validation=selection['validation'],
                       winner_worker=selection['agent'], winner_round=selection['round'])
            req = next(x['request'] for x in events if x['kind']=='request' and x['request']['role']=='researcher'
                       and x['request']['agent']==selection['agent'] and x['request']['round']==selection['round'])
            row['winner_peer_results'] = len(req['peer_results'])
            artifact = json.loads((folder/'frozen.json').read_text())
            assert digest(artifact['payload']) == roster['submissions'][name] == artifact['artifact_sha256']
            assert artifact['payload']['implementation_sha256'] == protocol['implementation_sha256']
            row['status'] = 'evaluation_failed' if name in evaluation_errors else 'frozen'
            row['failure'] = evaluation_errors.get(name, row['failure'])
        evaluation = root / 'hidden-evaluation' / name / 'evaluation.json'
        if evaluation.exists():
            result = json.loads(evaluation.read_text())
            assert datetime.fromisoformat(result['evaluated_at']) > datetime.fromisoformat(roster['frozen_before_any_evaluation_utc'])
            assert result['artifact_sha256'] == roster['submissions'][name]
            daily = pd.read_csv(evaluation.parent/'hidden_daily.csv',parse_dates=['date'])
            assert np.allclose(daily.net_return, daily.gross_return-daily.cost-daily.borrow_cost)
            assert len(daily)==500
            nav = np.cumprod(1+daily.net_return.to_numpy())
            assert np.isclose(nav[-1]-1,result['hidden']['total_return'])
            monthly = daily.assign(nav=nav).groupby(daily.date.dt.to_period('M')).tail(1)
            row.update(status='evaluated', test=result['hidden'], years=result['by_calendar_year'],
                       nav=dict(dates=['2024-01-02']+monthly.date.dt.strftime('%Y-%m-%d').tolist(),values=[1.]+monthly.nav.tolist()))
        records.append(row)
    totals = {key:sum(x[key] for x in records) for key in ['attempts','researcher_calls','calls','submitted','accepted_code',
              'valid','unique_valid','active_valid','cash_validation','tokens','input_tokens','output_tokens','reasoning_tokens','cached_input_tokens','input_reservation_total',
              'manager_tokens','repair_calls','truncated','normalized_broadcasts','manager_failed_rounds']}
    totals['tested'] = sum(x['status']=='evaluated' for x in records)
    totals['manager_terminal_failures'] = sum(x['manager_terminal_failure'] and x['status']=='no_submission' for x in records)
    totals['frozen'] = len(roster['submissions'])
    totals['active_tested'] = sum(x['status']=='evaluated' and x['test']['active_days']>0 for x in records)
    totals['profitable_tested'] = sum(x['status']=='evaluated' and x['test']['total_return']>0 for x in records)
    totals['unconfirmed_requests'] = sum(len(x['unconfirmed_requests']) for x in records)
    totals['token_budget_breaches'] = sum(x['token_budget_breached'] for x in records)
    totals['unconfirmed_allowance'] = sum(q['max_total_tokens'] for x in records for q in x['unconfirmed_requests'])
    assert totals['tokens']==completed['actual_model_tokens']
    assert totals['calls']==completed['actual_model_calls']
    assert totals['attempts']==completed['actual_candidate_trials']
    aggregates = []
    for topology in ORDER:
        rows=[x for x in records if x['topology']==topology]
        evaluated=[x for x in rows if x['status']=='evaluated']
        a=dict(topology=topology,label=NAMES[topology],planned=len(rows),tested=len(evaluated))
        for key in ['attempts','submitted','accepted_code','valid','unique_valid','active_valid','cash_validation','tokens','input_tokens','output_tokens','reasoning_tokens','truncated','manager_tokens','repair_calls']:
            a[key]=sum(x[key] for x in rows)
        a['valid_rate']=a['valid']/a['attempts'] if a['attempts'] else None
        a['tokens_per_valid']=a['tokens']/a['valid'] if a['valid'] else None
        if evaluated:
            a.update(test_sharpe_mean=float(np.mean([r['test']['sharpe'] for r in evaluated])),
                     test_annualized_return_mean=float(np.mean([r['test']['annualized_return'] for r in evaluated])),
                     validation_sharpe_mean=float(np.mean([r['validation']['sharpe'] for r in evaluated])))
        aggregates.append(a)
    registry=root/'operator-data/.quantscale-evaluations.sqlite3'
    registry_states=[]
    additional_registry_entries=0
    if registry.exists():
        connection=sqlite3.connect('file:'+str(registry.resolve())+'?mode=ro',uri=True)
        registered=connection.execute('SELECT artifact,status FROM evaluations').fetchall()
        connection.close()
        expected=set(roster['submissions'].values())
        registry_states=sorted(Counter(status for artifact,status in registered if artifact in expected).items())
        additional_registry_entries=sum(artifact not in expected for artifact,_ in registered)
        assert sum(n for _,n in registry_states)==len(expected)
    return dict(root=str(root.resolve()),protocol=protocol,completed=completed,totals=totals,runs=records,
                aggregates=aggregates,fingerprints=sorted(fingerprints),registry_states=registry_states,
                additional_registry_entries=additional_registry_entries)


def report(old, new):
    o,n=old['totals'],new['totals']
    ratio=lambda x,k:x[k]/x['attempts'] if x['attempts'] else None
    submitted_rate=lambda x:x['submitted']/x['researcher_calls'] if x['researcher_calls'] else None
    reasoning_share=lambda x:x['reasoning_tokens']/x['output_tokens'] if x['output_tokens'] else None
    efficiency=lambda x:x['tokens']/x['valid'] if x['valid'] else None
    sections=[]
    summary=(f'<div class="finding"><span>交付</span><p>完整代码提交率从 <strong>{pct(submitted_rate(o))}</strong> 变为 <strong>{pct(submitted_rate(n))}</strong>；完成训练与验证的候选从 <strong>{o["valid"]}</strong> 个变为 <strong>{n["valid"]}</strong> 个。最终冻结提交为 {o["frozen"]}/10 → {n["frozen"]}/10，这与单次代码提交率是两个指标。</p></div>'
             +f'<div class="finding"><span>token</span><p>已确认总量 {o["tokens"]:,} → <strong>{n["tokens"]:,}</strong>（变化 {(n["tokens"]/o["tokens"]-1)*100:+.1f}%）。每个有效候选的已确认token为 {fmt(efficiency(o),0)} → <strong>{fmt(efficiency(n),0)}</strong>。本轮有 {n["unconfirmed_requests"]} 个请求无用量回执，不能把已确认量当成完整账单。</p></div>'
             +'<div class="finding"><span>解读</span><p>更多可运行策略意味着更高研究覆盖量，不保证测试盈利。代码生成、运行正确性、组级提交和最终收益在下文分别统计；同一预算上限不代表实际总消耗相等。</p></div>')
    sections.append(section('overview','01','先看：完整提交、有效实验和费用分别怎么变了？',
        summary+table(['指标','修复前 V2','修复后 V3'],[
        ['最终冻结提交 / 计划',f"{o['frozen']} / 10",f"{n['frozen']} / 10"],
        ['最终测试完成 / 计划',f"{o['tested']} / 10",f"{n['tested']} / 10"],
        ['其中测试期有持仓的策略',o['active_tested'],n['active_tested']],
        ['其中测试期复合收益为正的策略',o['profitable_tested'],n['profitable_tested']],
        ['候选尝试',o['attempts'],n['attempts']],
        ['完整代码提交 / 研究者回执',f"{o['submitted']} / {o['researcher_calls']}（{pct(submitted_rate(o))}）",f"{n['submitted']} / {n['researcher_calls']}（{pct(submitted_rate(n))}）"],
        ['训练与验证均完成 / 尝试',f"{o['valid']} / {o['attempts']}（{pct(ratio(o,'valid'))}）",f"{n['valid']} / {n['attempts']}（{pct(ratio(n,'valid'))}）"],
        ['每组内去重后的有效策略数之和',o['unique_valid'],n['unique_valid']],
        ['训练与验证都有持仓的有效候选',o['active_valid'],n['active_valid']],
        ['验证期全程空仓的候选',o['cash_validation'],n['cash_validation']],
        ['候选输出截断',o['truncated'],n['truncated']],
        ['已确认总 token',f"{o['tokens']:,}",f"{n['tokens']:,}"],
        ['每个有效候选的已确认token（含已确认失败用量）',fmt(efficiency(o),0),fmt(efficiency(n),0)],
        ['输出中推理占比',pct(reasoning_share(o)),pct(reasoning_share(n))]])
        +'<p class="source">来源：两轮全部 trajectory.jsonl。完整提交指研究者返回非空策略代码；有效候选指训练和验证均完成，不代表盈利。重复代码可命中缓存，仍计尝试与模型费用，因此同时展示组内去重数量，并另列有持仓与全程空仓的数量。token是实际返回用量，不是预算上限。</p>'
        +'<div class="note"><strong>提交率、单位有效候选费用和最终收益需要分开看。</strong><p>提交更完整不保证策略盈利。若每个候选更便宜，实验也可能因完成更多候选而消耗相近总token；反之，提前结束造成的低总消耗并不一定代表高效率。</p></div>'))
    sections.append(section('productivity','02','每种结构的有效产出和 token 效率',
        completion_chart(old,new)+table(['结构','V2有效/尝试','V3有效/尝试','V2每有效候选token','V3每有效候选token','V3完整代码','V3组内唯一有效'],[
          [b['label'],f"{a['valid']}/{a['attempts']}",f"{b['valid']}/{b['attempts']}",fmt(a['tokens_per_valid'],0),fmt(b['tokens_per_valid'],0),b['submitted'],b['unique_valid']]
          for a,b in zip(old['aggregates'],new['aggregates'])])
        +'<p>单位有效候选费用的分子包含整个结构两次运行的研究者、管理者、格式修复和失败调用。有效数为零时不报告有限费用。该指标衡量端到端产出，不等于独立经济假设的数量。</p>'))
    sections.append(section('tokens','03','token用在哪里？包括管理者与修复费用',
        table(['用量类别','V2','V3'],[[label,f"{o[key]:,}",f"{n[key]:,}"] for label,key in [
          ('总调用数','calls'),('输入token','input_tokens'),('其中输入缓存命中token','cached_input_tokens'),
          ('输出token（含推理）','output_tokens'),('其中推理token','reasoning_tokens'),('管理者总token','manager_tokens'),
          ('格式修复调用数','repair_calls'),('确定性规范化成功轮数','normalized_broadcasts'),('因管理者协议失败未提交的组数','manager_terminal_failures')]])
        +f'<p class="source">V3另记录 {n["manager_failed_rounds"]} 个管理者失败轮次；V2未单独记录此事件，不能把旧版缺少事件当作失败轮数为零。</p>'
        +table(['结构','V2总token','V3总token','V3管理者token','V3格式修复调用'],[[b['label'],f"{a['tokens']:,}",f"{b['tokens']:,}",f"{b['manager_tokens']:,}",b['repair_calls']] for a,b in zip(old['aggregates'],new['aggregates'])])
        +f'<p class="source">V3有 {n["unconfirmed_requests"]} 个请求没有用量回执；其已预留请求额度合计 {n["unconfirmed_allowance"]:,} token。零表示本轮没有此类缺口。若存在缺口，已确认用量不能当作完整账单。输入缓存命中包含在输入总量中，推理包含在输出总量中，不能重复相加。</p>'
        +'<p>本报告比较 token 数量，不按未核实的网关单价估算金额。缓存命中、输出计价、角色设置会使同样的token数对应不同费用；最终支付金额以实际服务账单为准。</p>'))
    sections[-1]=sections[-1].replace('</section>',
        f'<div class="note"><strong>推理减少了，但输入上下文变大了。</strong><p>输入token从 {o["input_tokens"]:,} 增至 {n["input_tokens"]:,}；输出中的推理占比从 {pct(reasoning_share(o))} 降至 {pct(reasoning_share(n))}。管理者token从 {o["manager_tokens"]:,} 增至 {n["manager_tokens"]:,}。本轮有更多有效结果、更长的代码上下文及更多管理者调用，这些变化共同影响用量；不能把差额全当作纯通信费用。</p></div></section>')
    sections[-1]=sections[-1].replace('</section>','<p class="source">单位候选费用使用已确认token除以观察到的有效候选数；有未确认请求时，它不是包含全部账单费用的精确值。V2仅统计正式重跑，不合并此前中止准备运行的72,341个已确认token及3个未知请求；原始账本仍另存。</p></section>')
    sections[-1] = sections[-1].replace('</section>', table(
        ['运行','实际token','预算剩余（非费用）','预算停止原因'],
        [[r['label'],f"{r['tokens']:,}",f"{max(0,new['protocol']['config']['tokens']-r['tokens']):,}",r['budget_stop_reason'] or '未因预算预检停止'] for r in new['runs']])
        +'<p class="source">输入预留按消息UTF-8字节数+1024估计，不是供应商精确token计数，也不是已支付费用。带更多代码上下文的结构可能在实际token未达到上限时因预留不足提前停止；这也是本轮固定资源上限比较的限制。</p></section>')
    sections.append(section('returns','04','最终测试收益：修复是否同时改善了策略？',
        table(['结构','V2完成/计划','V3完成/计划','V2测试Sharpe均值','V3测试Sharpe均值','V2年化收益均值','V3年化收益均值'],[
          [b['label'],f"{a['tested']}/2",f"{b['tested']}/2",fmt(a.get('test_sharpe_mean')),fmt(b.get('test_sharpe_mean')),pct(a.get('test_annualized_return_mean')),pct(b.get('test_annualized_return_mean'))]
          for a,b in zip(old['aggregates'],new['aggregates'])])
        +'<p class="source">来源：冻结策略的单次历史测试。均值仅包括完成测试的组，缺失不填零，逐次Sharpe均值不是策略组合Sharpe。两轮候选不同，不是同一策略的前后收益。</p>'))
    sections[-1]=sections[-1].replace('</section>','<p>有效候选变多后，验证期有更多选优机会；验证成绩改善不自动意味着泛化改善。测试收益单独列示，未根据测试表现再次选择策略。</p></section>')
    valid=[r for r in new['runs'] if r['status']=='evaluated']
    if valid:
        sections.append(section('details','05','V3逐次结果及逐年收益',table(['运行','验证Sharpe','测试Sharpe','测试年化','最大回撤','2024累计','2025累计','有效/尝试','token'],[
           [r['label'],fmt(r['validation']['sharpe'],4),fmt(r['test']['sharpe']),pct(r['test']['annualized_return']),pct(r['test']['max_drawdown']),pct(r['years']['2024']['total_return']),pct(r['years']['2025']['total_return']),f"{r['valid']}/{r['attempts']}",f"{r['tokens']:,}"] for r in valid])))
        code=''
        for r in valid:
            code+=f'<details class="strategy"><summary>{escape(r["label"])} · 测试Sharpe {fmt(r["test"]["sharpe"])}<small>胜出worker {r["winner_worker"]} / 第{r["winner_round"]}轮 / 可见同伴有效结果 {r["winner_peer_results"]} 条</small></summary><p>{escape(r["hypothesis"])}</p><details><summary>完整Python策略代码</summary><pre><code>{escape(r["strategy_source"])}</code></pre></details></details>'
        sections.append(section('strategies','06','V3最终策略与实际信息来源',code+'<p class="source">假设为模型原始表述，不代替代码审计。完整代码已内嵌。不能仅凭组别名称认为胜出策略使用了协作信息。</p>'))
    failures=[r for r in new['runs'] if r['status']!='evaluated' or r['execution_errors'] or r['manager_failed_rounds']]
    if failures:
        sections.append(section('failures','07','所有失败都保留在统计里',table(['运行','最终状态','终止原因','候选错误与次数','失败管理者轮数'],[
            [r['label'],r['status'],r['failure'] or '—','；'.join(f'{v} × {k}' for k,v in r['execution_errors'].items()) or '—',r['manager_failed_rounds']] for r in failures])))
        truncated=[r for r in new['runs'] if r['truncated_output_caps']]
        if truncated:
            sections[-1]=sections[-1].replace('</section>',table(['出现截断的运行','该次实际输出上限（含推理）'],
                [[r['label'],', '.join(str(x) for x in r['truncated_output_caps'])] for r in truncated])
                +'<p class="source">剩余预算减去保守输入预留后，实际输出额度可能远小于配置的32,768。预算收尾截断与完整额度下仍无法完成代码，应分开解读。</p></section>')
    sections.append(section('protocol','08','这次改了什么，哪些设定保持一致？',
        table(['项目','V2','V3'],[['研究者','high / 输出16,384','low / 输出32,768'],['管理者','high / 输出16,384','关闭推理 / 输出8,192'],
        ['管理者协议','严格包装；格式失败终止整组','无歧义规范化；有限格式修复计费；连续2轮失败结束，保留已有候选'],
        ['预算','每组17万token、最多12次候选','相同'],['数据、交易与成本','2016–2021训练、2022–2023验证、2024–2025测试；交易5bp、借券年化30bp','相同'],
        ['重复与调度','五种结构各2次；组内顺序调用；组间并发4','相同']])
        +'<details><summary>展开五种拓扑结构与信息流图</summary>'+topology_html()+'</details>'
        +f'<p class="source">V3供应商返回的模型 / 指纹组合：{escape(str(new["fingerprints"]))}；V2：{escape(str(old["fingerprints"]))}。模型别名不是不可变快照，重复标签不控制服务端随机数。</p>'
        +'<p>两轮同时改变推理强度、输出额度和协议处理方式，因此结果是组合修复前后的描述性比较，不能分离每项修改的因果贡献。测试年份已重复使用，不是新盲测。当前成分股仍有幸存者偏差，且每种结构只有两次重复。</p>'
        +f'<p>每轮所有计划研究先结束并冻结，才统一开启测试；本报告验证冻结内容、实现版本和评估引用一致，并检查每份500个收益区间与净收益分解。V3有 {n["token_budget_breaches"]} 组已知token超预算；超出时系统中止，不自动重跑。没有依据本轮测试结果再调整候选。</p>'))
    template=Path(__file__).with_name('open_report_template.html').read_text()
    topology_path=Path(new['root'])/'topology-analysis.json'
    if topology_path.exists():
        from .topology_report_sections import sections as topology_sections
        extra=topology_sections(json.loads(topology_path.read_text()),new)
        sections[1:1]=extra[:2]
        returns_index=next(i for i,s in enumerate(sections) if 'id="returns"' in s)
        sections.insert(returns_index+1,extra[2])
    template=template.replace('<title>QuantScale · 开放策略搜索实验报告</title>','<title>QuantScale · V2与V3提交、用量及收益对比</title>')
    header='''<header><div class="eyebrow">QUANTSCALE / V2 → V3</div><h1>修复之后，<br>提交率与 token 效率如何变化？</h1><p class="intro">相同数据和资源上限下，对比策略交付、研究有效产出、模型用量与最终测试收益。</p><div class="meta"><span>DeepSeek Flash</span><span>500只股票</span><span>每组17万token</span><span>五种结构各两次</span></div><div class="actions"><button onclick="window.print()">打印 / 保存PDF</button></div></header>'''
    template=re.sub(r'<header>.*?</header>',lambda _:header,template,flags=re.S)
    template=re.sub(r'<script id="report-data".*?</script>\s*<script>.*?</script>','',template,flags=re.S)
    footer='<footer>来源：两轮冻结记录、trajectory.jsonl及hidden-evaluation。<p><a href="rerun-comparison.json">完整对比数据</a><a href="comparison.csv">V3逐次收益</a><a href="all-strategies-frozen.json">V3冻结记录</a></p></footer>'
    template=re.sub(r'<footer>.*?</footer>',lambda _:footer,template,flags=re.S)
    sections=[re.sub(r'<div class="section-no">\d+</div>',f'<div class="section-no">{i:02d}</div>',s,count=1) for i,s in enumerate(sections,1)]
    navigation=[]
    short_titles=dict(overview='对比总览',productivity='有效提交',tokens='token消耗',returns='测试收益',
                      details='逐次结果',strategies='最终策略',failures='失败记录',protocol='设定与边界',
                      **{'topology-mechanisms':'拓扑机制','search-dynamics':'搜索改进过程','topology-risk':'风险与相似性'})
    for i,s in enumerate(sections,1):
        identifier=re.search(r'id="([^"]+)"',s).group(1)
        title=re.search(r'<h2>(.*?)</h2>',s).group(1)
        navigation.append(f'<a href="#{identifier}">{i:02d}<span>{short_titles.get(identifier,title)}</span></a>')
    template=re.sub(r'<noscript>.*?</noscript>','',template,flags=re.S)
    return template.replace('@@NAV@@',''.join(navigation)).replace('@@BODY@@','\n'.join(sections)).replace('@@DATA@@','')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('old')
    parser.add_argument('new')
    args=parser.parse_args()
    old,new=collect(args.old),collect(args.new)
    target=Path(args.new)
    (target/'rerun-comparison.json').write_text(json.dumps(dict(old=old,new=new),ensure_ascii=False,indent=2,allow_nan=False))
    (target/'QuantScale-v2-v3-对比报告.html').write_text(report(old,new))
    print(json.dumps(dict(old=old['totals'],new=new['totals'],new_aggregate=new['aggregates']),ensure_ascii=False,indent=2))
