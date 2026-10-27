"""Audit the additional V3 repetition and extend the existing HTML deliverable."""
from html import escape
import json
from pathlib import Path
import re

import numpy as np

from .analyze_topologies import analyze
from .build_html_report import table, section, fmt, pct
from .build_v3_report import supplement_record
from .compare_open_runs import collect, ORDER, NAMES
from .report_topologies import topology_html


def located(batch):
    base=Path(batch['root'])
    for r in batch['runs']:
        r.update(research_dir=str(base/r['run']),evaluation_dir=str(base/'hidden-evaluation'/r['run']))
    return batch['runs']


def build():
    base=Path('results/sp500-deepseek-open-pilot-v3')
    newbase=Path('results/sp500-deepseek-open-pilot-v3-repeat3')
    original=collect(base);new=collect(newbase)
    assert original['protocol']['implementation_sha256']==new['protocol']['implementation_sha256']
    assert original['protocol']['config']==new['protocol']['config']
    oldruns=located(original)+[supplement_record(base)]
    newruns=located(new)
    for r in newruns:
        r['label']=NAMES[r['topology']]+' · 27（第3次）'
        events=[json.loads(x) for x in (Path(r['research_dir'])/'trajectory.jsonl').read_text().splitlines()]
        if any(x['kind']=='manager_protocol_exhausted' for x in events):
            r['termination_note']='管理者连续格式失败达到上限，保留已有最佳策略'
        elif r['budget_stop_reason']:
            r['termination_note']='剩余额度不足以覆盖下一请求的保守输入预留'
        else:
            r['termination_note']=r['failure'] or ('达到12个候选上限' if r['attempts']==12 else '研究结束')
        assert r['backend_configuration']==original['runs'][0]['backend_configuration']
        oldprotocol=json.loads((base/'single-n1-s7/protocol.json').read_text())
        runprotocol=json.loads((Path(r['research_dir'])/'protocol.json').read_text())
        assert runprotocol['research_data_hash']==oldprotocol['research_data_hash']
        if r['status']=='evaluated':
            oldeval=json.loads((base/'hidden-evaluation/single-n1-s7/evaluation.json').read_text())
            neweval=json.loads((Path(r['evaluation_dir'])/'evaluation.json').read_text())
            assert neweval['evaluator_data_sha256']==oldeval['evaluator_data_sha256']
    attempts=oldruns+newruns
    tested=[r for r in attempts if r['status']=='evaluated']
    tested.sort(key=lambda r:(ORDER.index(r['topology']),r['seed']))
    aggregates=[]
    for topology in ORDER:
        complete=[r for r in tested if r['topology']==topology]
        spent=[r for r in attempts if r['topology']==topology]
        previous=[r for r in oldruns if r['topology']==topology and r['status']=='evaluated']
        newest=next(r for r in newruns if r['topology']==topology)
        aggregates.append(dict(topology=topology,label=NAMES[topology],tested=len(complete),
            planned=3,positive=sum(r['test']['total_return']>0 for r in complete),
            test_sharpe_mean=float(np.mean([r['test']['sharpe'] for r in complete])),
            test_sharpe_sd=float(np.std([r['test']['sharpe'] for r in complete],ddof=1)) if len(complete)>1 else None,
            test_annualized_return_mean=float(np.mean([r['test']['annualized_return'] for r in complete])),
            test_max_drawdown_mean=float(np.mean([r['test']['max_drawdown'] for r in complete])),
            previous_sharpe_mean=float(np.mean([r['test']['sharpe'] for r in previous])),
            third_sharpe=newest.get('test',{}).get('sharpe'),
            tokens=sum(r['tokens'] for r in spent),attempts=sum(r['attempts'] for r in spent),
            valid=sum(r['valid'] for r in spent),research_attempts=len(spent)))
    token_total=sum(r['tokens'] for r in attempts)
    unknown=sum(len(r['unconfirmed_requests']) for r in attempts)
    analysis=analyze(base,tested,aggregates,output='v3-three-repetitions-analysis.json')
    payload=dict(scope='V3 only, third historical repetition; previous single seed17 completed by separate supplement',
        new_batch=new,attempts=attempts,aggregates=aggregates,tested=len(tested),
        total_confirmed_tokens=token_total,unconfirmed_requests=unknown)
    (base/'v3-three-repetitions-results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False))
    sections=[]
    newtested=[r for r in newruns if r['status']=='evaluated']
    best=max(aggregates,key=lambda r:r['test_sharpe_mean'])
    sections.append(section('overview','01','第三次重复：先看新增结果，再看结论是否改变',
        f'<p>第三次重复标签为27，五种拓扑各新增一次，完成测试 <strong>{len(newtested)}/5</strong>；累计完成 <strong>{len(tested)}/15</strong> 个逻辑条件。设置与前两次保持一致，全部新增搜索结束、策略冻结后才开启本批测试。</p>'
        +f'<div class="note"><strong>当前测试Sharpe均值最高：{best["label"]}，{fmt(best["test_sharpe_mean"],3)}。</strong><p>每种结构仅三次重复，且共用同一段历史测试期。均值用于描述这批结果，不代表稳定排名或独立样本的显著性结论。单Agent标签17来自此前的追加补跑，原失败成本仍保留。</p></div>'
        +table(['拓扑','新增状态','新增验证Sharpe','新增测试Sharpe','新增年化收益','新增最大回撤'],[
            [r['label'],r['status'],fmt(r.get('validation',{}).get('sharpe'),4),fmt(r.get('test',{}).get('sharpe'),4),pct(r.get('test',{}).get('annualized_return')),pct(r.get('test',{}).get('max_drawdown'))] for r in newruns])
        +'<p class="source">新增五次研究均独立开始，不导入前两次的候选或测试成绩。重复标签不控制DeepSeek服务端随机数；没有因为测试收益不佳自动替换运行。</p>'))
    sections.append(section('comparison','02','从两次到三次：平均成绩与运行波动',
        table(['结构','前两次Sharpe均值','第3次Sharpe','累计Sharpe均值','Sharpe样本标准差','盈利/完成','年化收益均值','最大回撤均值'],[
            [r['label'],fmt(r['previous_sharpe_mean'],3),fmt(r['third_sharpe'],3),fmt(r['test_sharpe_mean'],3),fmt(r['test_sharpe_sd'],3),f"{r['positive']}/{r['tested']}",pct(r['test_annualized_return_mean']),pct(r['test_max_drawdown_mean'])] for r in aggregates])
        +'<p>所有均值均为各次最终策略指标的算术均值，并非组合净值。标准差描述三次搜索结果之间的波动，不是估计市场收益的不确定区间。新增结果无论好坏都纳入，失败条件不记作零收益。</p>'
        +'<h3>第三次改变了什么判断？</h3><p><strong>混合组的重复表现得到加强：</strong>三次测试Sharpe约0.845、0.796、0.603，均盈利，均值0.748，运行间标准差0.128。它没有赢得每次最高收益，却在这三个搜索结果中保持了较小差异。</p>'
        +'<p><strong>独立和去中心化的能力不能由前两次定论：</strong>本次分别获得0.942和1.201的测试Sharpe。去中心化累计均值升至0.215，高于中心化0.049，但其运行间标准差达1.440。第三次独立组胜过中心化，不能继续把管理者存在本身解释为稳定收益优势。</p>'
        +'<p><strong>单Agent仍未显示测试盈利：</strong>三次测试均亏损，均值−0.767。第三次虽在最后一轮把验证提高到0.410，最终测试仍为−0.728。这支持继续检查自我迭代与验证选优的泛化问题，并不证明所有单Agent设计都无效。</p>'))
    options=''.join(f'<option value="{r["topology"]}">{r["label"]}</option>' for r in aggregates)
    sections.append(section('paths','03','三次策略的测试期表现',
        table(['策略','验证Sharpe','测试Sharpe','年化收益','最大回撤','2024累计','2025累计'],[
            [r['label'],fmt(r['validation']['sharpe'],4),fmt(r['test']['sharpe'],3),pct(r['test']['annualized_return']),pct(r['test']['max_drawdown']),pct(r['years']['2024']['total_return']),pct(r['years']['2025']['total_return'])] for r in tested])
        +'<div class="chart-box"><div class="chart-heading"><h3>2024–2025测试期复合净值</h3><label>选择结构 <select id="nav-select">'+options+'</select></label></div><p class="chart-label">横轴：月末日期 · 纵轴：净值（初始1），已扣交易与借券成本</p><div id="nav-chart"></div><div id="nav-legend" class="legend"></div><p class="source">逐日净收益连续复合后取月末值。完整日序列用于计算最大回撤，图不显示全部月内波动。</p></div>'))
    ar={r['run']:r for r in analysis['runs']}
    detail=table(['新增研究','候选/有效/有持仓','验证赢家worker/轮次','赢家可见同伴结果','已确认token','终止原因或错误'],[
        [r['label'],f"{r['attempts']}/{r['valid']}/{r['active_valid']}",f"W{ar[r['run']]['winner_worker']} / {ar[r['run']]['winner_round']}" if r['run'] in ar and 'winner_worker' in ar[r['run']] else '无最终提交',ar.get(r['run'],{}).get('winner_peers','—'),f"{r['tokens']:,}",r['termination_note']] for r in newruns])
    for r in newruns:
        events=[json.loads(x) for x in (Path(r['research_dir'])/'trajectory.jsonl').read_text().splitlines()]
        trials=[x for x in events if x['kind']=='trial']
        detail+=f'<details><summary>{escape(r["label"])}：逐次候选与错误</summary>'+table(['轮次','worker','验证Sharpe','执行错误'],[[x['round'],x['agent'],fmt(x.get('validation',{}).get('sharpe'),4),x.get('error','完成训练与验证')] for x in trials])+'</details>'
    sections.append(section('process','04','第三次搜索过程与有效产出',detail+'<p>有效表示完成训练与验证回测；有持仓表示两个区间均有持仓。空仓候选虽能完成回测，不代表发现交易信号。轮次从0开始，组内worker顺序调用；共享的是上一轮各worker最新的有效回测。最终只选一份验证净Sharpe最高的策略，不把四名worker的仓位自动组合。</p>'
        +'<div class="note"><strong>混合组第三次赢家来自首轮，尚未获得同伴回测反馈。</strong><p>前两次混合赢家来自后续修订，本次则为首轮W1的候选。因此，三次均盈利加强了结果层面的重复证据，却没有证明同伴信息或反馈修订是盈利的必要原因。本次去中心化与独立组赢家也来自首轮；中心化赢家来自后续轮次，却在测试期亏损。</p></div>'
        +'<p>中心化组完成8次候选后，管理者在第2、3轮连续返回不完整JSON，触发预设格式失败上限，随后冻结已有最佳策略。它虽有最终提交，也没有正常用满候选机会；这类运行限制仍会影响拓扑比较。</p>'))
    sections.append(section('tokens','05','追加预算与全部实际用量',
        table(['口径','已确认token','候选尝试','有效候选'],[
            ['前两次及单Agent补跑',f"{sum(r['tokens'] for r in oldruns):,}",sum(r['attempts'] for r in oldruns),sum(r['valid'] for r in oldruns)],
            ['新增第三次重复',f"{new['totals']['tokens']:,}",new['totals']['attempts'],new['totals']['valid']],
            ['全部V3研究活动',f"{token_total:,}",sum(r['attempts'] for r in attempts),sum(r['valid'] for r in attempts)]])
        +table(['结构','研究尝试数','已确认token','有效/候选','每有效候选token'],[
            [r['label'],r['research_attempts'],f"{r['tokens']:,}",f"{r['valid']}/{r['attempts']}",fmt(r['tokens']/r['valid'],0) if r['valid'] else '—'] for r in aggregates])
        +f'<p>第三次总预算上限850,000 token、60个候选，每组最多170,000 token、12个候选。全部研究仍有 {unknown} 个请求用量未知，已确认量不等于完整账单；原单Agent失败尝试没有被删除。推理token包含在输出token中，不重复相加。详细输入、输出、推理及失败用量见完整JSON。</p>'))
    code=''
    for r in newruns:
        if 'strategy_source' in r:
            code+=f'<details class="strategy"><summary>{escape(r["label"])} · 测试Sharpe {fmt(r.get("test",{}).get("sharpe"),3)}</summary><p>{escape(r["hypothesis"])}</p><pre><code>{escape(r["strategy_source"])}</code></pre></details>'
    sections.append(section('code','06','第三次的最终策略代码',code+'<p>这里保留模型的原始假设与实际代码，假设中的收益预期不代表已验证的经济事实。前两次代码保留在原V3报告中。</p>'))
    sections.append(section('methods','07','保持一致的实验设定',
        table(['项目','设定'],[['市场数据','500只当前标普成分股的Yahoo历史行情，存在幸存者与未来成分信息偏差'],['区间','2016–2021训练、2022–2023验证、2024–2025测试'],['模型','DeepSeek Flash；研究者low/最大输出32,768，管理者none/最大输出8,192'],['结构','单Agent为1研究者，其余为4研究者；中心化与混合各有管理者'],['预算','每次170,000总token、12个候选；研究失败不自动替换'],['交易','收盘决策，下一有效开盘交易并持有至随后开盘；交易5bp、借券年化30bp'],['仓位与运行','总敞口≤1，单股≤2%，净敞口≤1%；决策10秒，每段回测180秒，无严格内存上限'],['质量与边界','跳过2021-05-05、2025-11-12；500个测试区间，按252年化。未模拟冲击及真实借券可得性'],['实验完整性','相同实现hash、数据hash与角色设置；新增五组全部冻结后才评估，各评估一次']])
        +'<details><summary>查看五种拓扑及信息流</summary>'+topology_html()+'</details>'
        +'<p>此轮保留了原V3的保守输入预留与固定调度，不包含预算优化。三次重复共用历史测试段，并非三段独立市场样本；新增运行发生在先前测试已被查看之后，属于历史复验，不构成全新盲测。</p>'))
    template=Path(__file__).with_name('open_report_template.html').read_text()
    template=re.sub(r'<title>.*?</title>','<title>QuantScale · V3三次重复实验报告</title>',template)
    template=re.sub(r'<header>.*?</header>',lambda _:f'<header><div class="eyebrow">QUANTSCALE / V3</div><h1>五种拓扑，<br>第三次重复检验</h1><p class="intro">沿用前两次实验设置，新增五次研究。累计完成{len(tested)}/15个逻辑条件。</p><div class="meta"><span>DeepSeek Flash</span><span>500只股票</span><span>重复标签7 / 17 / 27</span></div><div class="actions"><button onclick="window.print()">打印 / 保存PDF</button></div></header>',template,flags=re.S)
    template=re.sub(r'<footer>.*?</footer>','<footer>V3三次重复实验 · 原始记录保留<p><a href="v3-three-repetitions-results.json">全部结果与用量JSON</a><a href="v3-three-repetitions-analysis.json">搜索过程与风险分析JSON</a><a href="QuantScale-V3-拓扑实验报告.html">前两次V3报告</a></p></footer>',template,flags=re.S)
    template=re.sub(r'<noscript>.*?</noscript>','',template,flags=re.S).replace('bars();navChart();','navChart();')
    template=template.replace("j===0?'#176275':'#8798a2'", "['#176275','#8798a2','#a46c32'][j%3]")
    labels=['新增结果','三次对比','测试净值','搜索过程','token账目','新增代码','实验设定']
    links=[]
    for i,(part,label) in enumerate(zip(sections,labels),1):
        identifier=re.search(r'id="([^"]+)"',part).group(1)
        links.append(f'<a href="#{identifier}">{i:02d}<span>{label}</span></a>')
    nav=''.join(links)
    d=dict(aggregates=aggregates,runs=[dict(label=r['label'],topology=r['topology'],nav=r['nav']) for r in tested])
    embedded=json.dumps(d,ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('&','\\u0026')
    html=template.replace('@@NAV@@',nav).replace('@@BODY@@','\n'.join(sections)).replace('@@DATA@@',embedded)
    assert '@@' not in html and 'V2' not in html
    path=base/'QuantScale-V3-三次重复实验报告.html';path.write_text(html)
    print(json.dumps(dict(report=str(path.resolve()),new_totals=new['totals'],aggregates=aggregates),ensure_ascii=False,indent=2))


if __name__=='__main__':build()
