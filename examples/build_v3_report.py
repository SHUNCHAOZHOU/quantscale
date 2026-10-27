"""V3-only results with one explicitly separated supplementary single-agent run."""
from collections import Counter
from datetime import datetime
from html import escape
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from quantscale.evaluation import digest
from .analyze_topologies import analyze
from .build_html_report import table, section, fmt, pct
from .compare_open_runs import collect, ORDER, NAMES
from .report_topologies import topology_html
from .topology_report_sections import sections as mechanism_sections


def supplement_record(base):
    target=base/'supplement-single-n1-s17'
    done=json.loads((target/'completed.json').read_text())
    research=target/'research';evaluation=target/'evaluation'
    events=[json.loads(x) for x in (research/'trajectory.jsonl').read_text().splitlines()]
    trials=[x for x in events if x['kind']=='trial']
    responses=[x for x in events if x['kind']=='response']
    valid=[x for x in trials if 'validation' in x]
    pending=[]
    for e in events:
        if e['kind']=='request':pending.append(e['request'])
        elif e['kind']=='response':pending.pop(0)
        elif e['kind']=='budget_exhausted' and pending and 'conservative DeepSeek input reservation'in e['reason']:pending.pop()
    usages=[x['response']['_provenance']['usage'] for x in responses]
    r=dict(run='single-n1-s17-supplement',condition='single-n1-s17',label='单 Agent · 17（补跑）',
           topology='single',seed=17,fleet_size=1,additional_attempt=True,
           research_dir=str(research.resolve()),evaluation_dir=str(evaluation.resolve()),
           attempts=len(trials),calls=len(responses),
           submitted=sum(isinstance(x['response'].get('strategy_code'),str) for x in responses),
           valid=len(valid),unique_valid=len({x['strategy_id'] for x in valid}),
           active_valid=sum(x['train']['active_days']>0 and x['validation']['active_days']>0 for x in valid),
           cash_validation=sum(x['validation']['active_days']==0 for x in valid),
           tokens=sum(x['tokens'] for x in responses),input_tokens=sum(x['prompt_tokens'] for x in usages),
           output_tokens=sum(x['completion_tokens'] for x in usages),
           reasoning_tokens=sum(x.get('completion_tokens_details',{}).get('reasoning_tokens',0) for x in usages),
           manager_tokens=0,repair_calls=0,
           truncated=sum('did not finish: length'in x['response'].get('adapter_error','') for x in responses),
           unconfirmed_requests=[dict(role=q['role'],max_total_tokens=q['max_total_tokens']) for q in pending],
           execution_errors=dict(Counter(x['error'] for x in trials if 'error'in x)),
           stopping_reason=next((x['reason'] for x in reversed(events) if x['kind']=='budget_exhausted'),'not recorded'),
           trial_history=[dict(number=x['round']+1,validation_sharpe=x.get('validation',{}).get('sharpe'),error=x.get('error')) for x in trials],
           status=done['status'],failure=done.get('error',''))
    assert r['tokens']==done['confirmed_tokens']
    if (research/'selection.json').exists():
        winner=json.loads((research/'selection.json').read_text())
        r.update(validation=winner['validation'],hypothesis=winner['hypothesis'],strategy_source=winner['strategy']['source'],strategy_id=winner['strategy_id'])
    if r['status']=='evaluated':
        artifact=json.loads((research/'frozen.json').read_text());result=json.loads((evaluation/'evaluation.json').read_text())
        assert digest(artifact['payload'])==artifact['artifact_sha256']==result['artifact_sha256']
        assert result==done['evaluation']
        assert datetime.fromisoformat(result['evaluated_at']).timestamp()>(research/'frozen.json').stat().st_mtime
        daily=pd.read_csv(evaluation/'hidden_daily.csv',parse_dates=['date'])
        assert len(daily)==500 and np.allclose(daily.net_return,daily.gross_return-daily.cost-daily.borrow_cost)
        nav=np.cumprod(1+daily.net_return.to_numpy());monthly=daily.assign(nav=nav).groupby(daily.date.dt.to_period('M')).tail(1)
        r.update(test=result['hidden'],years=result['by_calendar_year'],nav=dict(dates=['2024-01-02']+monthly.date.dt.strftime('%Y-%m-%d').tolist(),values=[1.]+monthly.nav.tolist()))
    return r


def build(base):
    original=collect(base)
    supplemental=supplement_record(base)
    original_runs=original['runs']
    for r in original_runs:
        r.update(condition=r['run'],additional_attempt=False,
                 research_dir=str((base/r['run']).resolve()),evaluation_dir=str((base/'hidden-evaluation'/r['run']).resolve()))
    conditions=[r for r in original_runs if r['status']=='evaluated']
    if supplemental['status']=='evaluated':conditions.append(supplemental)
    conditions.sort(key=lambda r:(ORDER.index(r['topology']),r['seed']))
    attempts=original_runs+[supplemental]
    aggregates=[]
    for topology in ORDER:
        completed=[r for r in conditions if r['topology']==topology]
        spent=[r for r in attempts if r['topology']==topology]
        row=dict(topology=topology,label=NAMES[topology],planned=2,tested=len(completed),research_attempts=len(spent))
        for key in ['attempts','submitted','valid','unique_valid','tokens','input_tokens','output_tokens','reasoning_tokens','manager_tokens','repair_calls']:
            row[key]=sum(r[key] for r in spent)
        row['tokens_per_valid']=row['tokens']/row['valid'] if row['valid'] else None
        if completed:
            row.update(validation_sharpe_mean=float(np.mean([r['validation']['sharpe'] for r in completed])),
                       test_sharpe_mean=float(np.mean([r['test']['sharpe'] for r in completed])),
                       test_annualized_return_mean=float(np.mean([r['test']['annualized_return'] for r in completed])),
                       test_max_drawdown_mean=float(np.mean([r['test']['max_drawdown'] for r in completed])))
        aggregates.append(row)
    totals={key:sum(r[key] for r in attempts) for key in ['attempts','calls','submitted','valid','active_valid','tokens','input_tokens','output_tokens','reasoning_tokens','manager_tokens','repair_calls','truncated']}
    totals.update(unconfirmed_requests=sum(len(r['unconfirmed_requests']) for r in attempts),
                  unconfirmed_allowance=sum(q['max_total_tokens'] for r in attempts for q in r['unconfirmed_requests']),
                  tested=len(conditions),logical_conditions=10,research_attempts=len(attempts))
    analysis=analyze(base,conditions,aggregates,output='v3-topology-analysis-with-supplement.json')
    payload=dict(scope='V3 only; original failed attempt retained; supplemental cohort is descriptive, not the original preregistered batch',
                 original_batch_totals=original['totals'],supplement=supplemental,conditions=conditions,aggregates=aggregates,totals=totals)
    (base/'v3-results-with-supplement.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False))
    s=supplemental;parts=[]
    parts.append(section('v3-overview','01','V3结果：补全视图与实际成本分开看',
        f'<p>当前完成测试的逻辑条件为 <strong>{len(conditions)}/10</strong>。原批次有9份测试结果，单Agent标签17因网络失败没有提交；本次单独追加一次研究，状态为 <strong>{escape(s["status"])}</strong>。原始失败记录仍然保留。</p>'
        +table(['统计口径','已确认token','候选尝试','完成训练与验证'],[
            ['原V3批次（包含原网络失败）',f"{original['totals']['tokens']:,}",original['totals']['attempts'],original['totals']['valid']],
            ['本次单Agent补跑',f"{s['tokens']:,}",s['attempts'],s['valid']],
            ['全部V3研究活动合计',f"{totals['tokens']:,}",totals['attempts'],totals['valid']]])
        +f'<p>其中训练与验证都有持仓的有效候选共 {totals["active_valid"]} 个。仍有 {totals["unconfirmed_requests"]} 个请求没有用量回执，对应预留请求额度 {totals["unconfirmed_allowance"]:,} token；已确认token不能当作完整账单。</p>'
        +'<div class="note"><strong>补跑不抹去原失败，也不构成全新盲测。</strong><p>策略结果表每个逻辑条件取一份，单Agent·17使用这次补跑的冻结结果，原失败无收益可替换。成本表则计入全部研究尝试。单Agent因此有一次额外研究尝试，不能把补全视图当作完全同时、严格预算匹配的原始十组实验。</p></div>'))
    body=table(['结构','完成/计划','测试Sharpe均值','测试年化收益均值','测试最大回撤均值','研究尝试数'],[
        [r['label'],f"{r['tested']}/2",fmt(r.get('test_sharpe_mean')),pct(r.get('test_annualized_return_mean')),pct(r.get('test_max_drawdown_mean')),r['research_attempts']] for r in aggregates])
    body+='<p class="source">来源：V3原始冻结评估与本次追加的单次评估。均值是各次指标的算术均值，不是组合策略表现；补跑仅覆盖原失败条件，没有按收益替换其他策略。</p>'
    body+=table(['策略','验证Sharpe','测试Sharpe','年化收益','最大回撤','2024累计','2025累计'],[
        [r['label'],fmt(r['validation']['sharpe'],4),fmt(r['test']['sharpe']),pct(r['test']['annualized_return']),pct(r['test']['max_drawdown']),pct(r['years']['2024']['total_return']),pct(r['years']['2025']['total_return'])] for r in conditions])
    options=''.join(f'<option value="{r["topology"]}"'+(' selected' if r['topology']=='single' else '')+f'>{r["label"]}</option>' for r in aggregates if r['tested'])
    body+='<div class="chart-box"><div class="chart-heading"><h3>V3测试期净值路径</h3><label>选择结构 <select id="nav-select">'+options+'</select></label></div><p class="chart-label">横轴：月末日期 · 纵轴：复合净值，初始1；多空交易费5bp、借券年化30bp</p><div id="nav-chart"></div><div id="nav-legend" class="legend"></div><p class="source">逐日净收益连续复合后月末抽样；回撤指标使用完整日序列，月末图不会显示全部月内波动。</p></div>'
    parts.append(section('v3-results','02','五种拓扑的V3测试结果',body))
    detail=table(['项目','本次单独补跑'],[['重复标签','17；服务端随机数不受标签控制'],['研究方式','重新开始搜索，不导入原失败尝试的候选或测试反馈'],['额外预算上限','170,000总token，最多12个候选'],['生成设置','研究者low推理，最大输出32,768；V3代码与数据接口不变'],['当前状态',s['status']],['实际已确认token',f"{s['tokens']:,}"],['候选尝试 / 有效',f"{s['attempts']} / {s['valid']}"],['错误',s['failure'] or '无终止错误']])
    if s['status']=='evaluated':
        detail+=f'<p>本次选中策略验证Sharpe为 <strong>{fmt(s["validation"]["sharpe"],4)}</strong>，测试Sharpe为 <strong>{fmt(s["test"]["sharpe"],4)}</strong>，测试年化收益 <strong>{pct(s["test"]["annualized_return"])}</strong>，最大回撤 <strong>{pct(s["test"]["max_drawdown"])}</strong>。</p>'
    detail+=table(['候选顺序','验证Sharpe','执行结果'],[[x['number'],fmt(x['validation_sharpe'],4),'单次决策超过10秒' if x['error'] else '完成训练与验证'] for x in s['trial_history']])
    detail+='<p>第2、7个候选发生决策超时，属于策略代码执行失败；本次没有网络终止错误。用量达到134,284 token后，剩余35,716额度不足以覆盖下一请求的保守输入预留，因此正常停止，没有强行用完17万上限。12个候选是上限，并非保证次数。</p>'
    detail+='<p>最终选中第1个候选：用最近完成周五的5日收益排序，买入最多40只跌幅最大的股票，卖空最多40只涨幅最大的股票，两侧各约50%等权敞口。后续6次尝试均未超过它。其测试期2024年收益−5.76%，2025年−12.17%；此次补跑补全了流程，但没有改善为盈利策略。</p>'
    parts.append(section('supplement','03','单Agent标签17的补跑记录',detail+'<p class="source">原网络失败已确认消耗13,343 token，另有一个在途请求用量未知；两者没有从总成本里删除。研究与冻结完成后，新增策略只测试一次。</p>'))
    mechanisms=mechanism_sections(analysis,dict(aggregates=aggregates))
    parts.extend(mechanisms)
    parts.append(section('v3-tokens','04','V3 token消耗与有效产出',
        table(['结构','全部已确认token','输入token','输出token','其中推理token','有效/候选尝试','每有效候选token'],[
            [r['label'],f"{r['tokens']:,}",f"{r['input_tokens']:,}",f"{r['output_tokens']:,}",f"{r['reasoning_tokens']:,}",f"{r['valid']}/{r['attempts']}",fmt(r['tokens_per_valid'],0)] for r in aggregates])
        +'<p>上表计入所有V3研究活动：单Agent包含原失败尝试和追加补跑。输入、输出与推理必须分清，推理包含在输出中，不能重复相加。每个有效候选的费用包含已确认失败用量，但未包含服务端尚未确认的账单费用。</p>'
        +'<p>同样的预算上限不代表实际消耗相等。当前输入预留按消息UTF-8字节数+1024估计，带更多代码上下文的结构可能提前停止；本次追加预算也使单Agent总研究机会不同。不能单凭这张成本表认定某一拓扑长期更经济。</p>'))
    code=''
    for r in conditions:
        code+=f'<details class="strategy"><summary>{escape(r["label"])} · 测试Sharpe {fmt(r["test"]["sharpe"])}</summary><p>{escape(r["hypothesis"])}</p><details><summary>完整Python代码</summary><pre><code>{escape(r["strategy_source"])}</code></pre></details></details>'
    parts.append(section('v3-code','05','最终策略与代码',code+'<p class="source">假设是模型原始表述，应结合代码和实际收益阅读；程序、假设、回测和最终测试不代表相同层次的有效性。</p>'))
    parts.append(section('v3-methods','06','V3实验设定与边界',
        table(['项目','设定'],[['数据','500只当前标普成分股的真实Yahoo行情；存在幸存者和未来成分信息偏差'],['时间划分','2016–2021训练、2022–2023验证、2024–2025测试'],['预算','每次研究上限17万token、12候选；本次补跑另计'],['选优','每次按验证净Sharpe选一份，平分按策略SHA；冻结后各评估一次'],['交易','收盘决策，下个有效开盘执行，随后开盘结算；总敞口≤1、单股≤2%、净敞口绝对值≤1%'],['成本与限制','交易5bp，空头借券年化30bp；决策10秒、每段回测180秒；未模拟冲击、真实借券可得性、保证金融资或现金担保利息'],['数据质量','跳过2021-05-05和2025-11-12不完整会话，执行价格不填补；500个测试区间，仍按252年化']])
        +'<details><summary>五种拓扑结构与信息流图</summary>'+topology_html()+'</details>'
        +'<p>管理者和同伴只拿完成回测的候选，包括亏损策略，不共享执行错误详情；研究者仅保留自身最近3条结果。组内worker按编号顺序调用，组间并发；每组最终选择一个策略，不自动组合多名worker的仓位。</p>'
        +'<p>每种结构最多两份完整测试结果。测试年份已反复使用，补跑也不是新盲测。当前结果适合研究过程分析，不足以确定稳定拓扑排名或可部署投资结论。</p>'))
    template=Path(__file__).with_name('open_report_template.html').read_text()
    template=template.replace('<title>QuantScale · 开放策略搜索实验报告</title>','<title>QuantScale · V3拓扑结果与单Agent补跑</title>')
    header=f'<header><div class="eyebrow">QUANTSCALE / V3</div><h1>V3拓扑实验：<br>单Agent补跑与完整结果</h1><p class="intro">只展示V3。当前完成测试{len(conditions)}/10个逻辑条件；原网络失败与本次追加研究分别记录。</p><div class="meta"><span>DeepSeek Flash</span><span>500只股票</span><span>每次最多17万token</span><span>补跑单独记账</span></div><div class="actions"><button onclick="window.print()">打印 / 保存PDF</button></div></header>'
    template=re.sub(r'<header>.*?</header>',lambda _:header,template,flags=re.S)
    footer='<footer>仅展示V3数据。<p><a href="v3-results-with-supplement.json">V3完整数据与补跑账目</a><a href="v3-topology-analysis-with-supplement.json">拓扑过程与风险分析</a><a href="supplement-single-n1-s17/completed.json">本次补跑结果</a></p></footer>'
    template=re.sub(r'<footer>.*?</footer>',lambda _:footer,template,flags=re.S)
    template=re.sub(r'<noscript>.*?</noscript>','',template,flags=re.S)
    # Use the existing offline SVG net-value renderer, without its old bar chart.
    template=template.replace('bars();navChart();','navChart();')
    d=dict(aggregates=aggregates,runs=[dict(label=r['label'],topology=r['topology'],nav=r['nav']) for r in conditions])
    embedded=json.dumps(d,ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('&','\\u0026')
    labels={'v3-overview':'V3总览','v3-results':'测试结果','supplement':'单Agent补跑',
            'topology-mechanisms':'拓扑机制','search-dynamics':'搜索改进过程','topology-risk':'风险与相似性',
            'v3-tokens':'token账目','v3-code':'策略代码','v3-methods':'实验设定'}
    nav=[]
    for i,part in enumerate(parts,1):
        parts[i-1]=re.sub(r'<div class="section-no">\d+</div>',f'<div class="section-no">{i:02d}</div>',part,count=1)
        identifier=re.search(r'id="([^"]+)"',part).group(1)
        nav.append(f'<a href="#{identifier}">{i:02d}<span>{labels[identifier]}</span></a>')
    html=template.replace('@@NAV@@',''.join(nav)).replace('@@BODY@@','\n'.join(parts)).replace('@@DATA@@',embedded)
    assert 'V2' not in html
    (base/'QuantScale-V3-拓扑实验报告.html').write_text(html)
    return payload


if __name__=='__main__':
    result=build(Path('results/sp500-deepseek-open-pilot-v3'))
    print(json.dumps(dict(supplement={k:result['supplement'][k] for k in ['status','tokens','attempts','valid']},totals=result['totals'],aggregates=result['aggregates']),ensure_ascii=False,indent=2))
