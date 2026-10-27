"""Read-only audit and HTML results for the paired-start study."""
from html import escape
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from quantscale.evaluation import digest
from .paired_research import state_digest
from .build_html_report import table, section, fmt, pct
from .compare_open_runs import NAMES

ROOT=Path('results/sp500-deepseek-paired-start')
OUT=Path('results/sp500-deepseek-open-pilot-v3/paired-start-report')
TOPS=['independent','centralized','decentralized','hybrid']
COLORS=['#8798a2','#a46c32','#6a7b5a','#176275']


def build():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import font_manager
    from matplotlib.ticker import PercentFormatter
    from datetime import datetime

    done=json.loads((ROOT/'completed.json').read_text())
    plan=json.loads((ROOT/'study-protocol.json').read_text())
    roster=json.loads((ROOT/'all-strategies-frozen.json').read_text())
    evaluations={r['run']:r for r in done['results']}
    failures={r['run']:r['error'] for r in done['research_failures']+done['evaluation_failures']}
    seeds=plan['seeds'];records=[];baselines={};initial_pools={}
    for seed in seeds:
        name=f'baseline-s{seed}'
        if name in evaluations:baselines[seed]=evaluations[name]
        state_path=ROOT/name/'common-state.json'
        if state_path.exists():initial_pools[seed]=json.loads(state_path.read_text())
    for seed in seeds:
        for top in TOPS:
            name=f'{top}-s{seed}';folder=ROOT/name
            r=dict(run=name,seed=seed,topology=top,label=f'起点{seeds.index(seed)+1} · {NAMES[top]}',
                status='failed',failure=failures.get(name,'baseline unavailable'),
                tokens=done['usage'].get(name,{}).get('confirmed_tokens',0),
                unknown_requests=done['usage'].get(name,{}).get('unconfirmed_requests',[]),
                added=0,valid=0,curve=[],hypothesis='',code='')
            if (folder/'trajectory.jsonl').exists():
                events=[json.loads(x) for x in (folder/'trajectory.jsonl').read_text().splitlines()]
                initial=json.loads((folder/'initial-state.json').read_text())
                source=json.loads((ROOT/f'baseline-s{seed}/common-state.json').read_text())
                protocol=json.loads((folder/'protocol.json').read_text())
                assert initial==source and state_digest(initial)==protocol['initial_state_sha256']
                assert protocol['study_source_sha256']==plan['study_source_sha256']
                baseline=max(t['validation']['sharpe'] for t in initial if 'validation' in t)
                trials=[x for x in events if x['kind']=='trial']
                best=baseline;curve=[baseline]
                for t in trials:
                    if 'validation' in t:best=max(best,t['validation']['sharpe'])
                    curve.append(best)
                r.update(added=len(trials),valid=sum('validation' in t for t in trials),
                    active_valid=sum(t['validation']['active_days']>0 and t['train']['active_days']>0 for t in trials if 'validation' in t),
                    curve=curve,baseline_validation=baseline,
                    errors=[dict(worker=t['agent'],round=t['round'],error=t['error']) for t in trials if 'error'in t],
                    manager_failures=sum(x['kind']=='manager_round_failed' for x in events),
                    manager_tokens=sum(x['tokens'] for x in events if x['kind']=='response' and x['role']=='manager'),
                    stop_reasons=[x['reason'] for x in events if x['kind'] in ['budget_exhausted','manager_protocol_exhausted']])
                r['stop_reasons'] += [f"Scheduled round {x['round']} skipped after manager format failure; no replacement round" for x in events if x['kind']=='manager_round_failed']
            if name in evaluations:
                result=evaluations[name]['evaluation'];artifact=json.loads((folder/'frozen.json').read_text())
                assert digest(artifact['payload'])==artifact['artifact_sha256']==result['artifact_sha256']==roster['submissions'][name]
                assert datetime.fromisoformat(result['evaluated_at'])>datetime.fromisoformat(roster['frozen_before_any_evaluation_utc'])
                winner=json.loads((folder/'selection.json').read_text())
                baseline_result=baselines[seed]['evaluation']['hidden']
                daily=pd.read_csv(ROOT/'hidden-evaluation'/name/'hidden_daily.csv',parse_dates=['date','execution_date'])
                assert len(daily)==500
                nav=np.cumprod(1+daily.net_return.to_numpy())-1
                assert np.isclose(nav[-1],result['hidden']['total_return'])
                r.update(status='evaluated',failure='',complete=r['added']==8,test=result['hidden'],
                    final_validation=result['validation']['sharpe'],validation_gain=result['validation']['sharpe']-r['baseline_validation'],
                    test_sharpe_change=result['hidden']['sharpe']-baseline_result['sharpe'],
                    test_return_change=result['hidden']['total_return']-baseline_result['total_return'],
                    winner_initial=winner['round']==0,winner_worker=winner['agent'],winner_round=winner['round'],
                    strategy_id=winner['strategy_id'],hypothesis=winner['hypothesis'],code=winner['strategy']['source'],
                    dates=[str(daily.execution_date.iloc[0].date())]+daily.date.dt.strftime('%Y-%m-%d').tolist(),
                    cumulative=[0.]+nav.tolist())
            records.append(r)
    aggregates=[]
    for top in TOPS:
        allrows=[r for r in records if r['topology']==top];rs=[r for r in allrows if r['status']=='evaluated']
        a=dict(topology=top,label=NAMES[top],evaluated=len(rs),complete_eight=sum(r.get('complete',False) for r in rs),
               tokens=sum(r['tokens'] for r in allrows),new_valid=sum(r['valid'] for r in allrows),
               added=sum(r['added'] for r in allrows),improved=sum(r['validation_gain']>1e-10 for r in rs),
               positive_test=sum(r['test']['total_return']>0 for r in rs))
        a['cash_test']=sum(r['test']['active_days']==0 for r in rs)
        for metric in ['validation_gain','test_sharpe_change','test_return_change']:
            a[metric]=float(np.mean([r[metric] for r in rs])) if rs else None
        for metric in ['sharpe','annualized_return','total_return','max_drawdown']:
            a[metric]=float(np.mean([r['test'][metric] for r in rs])) if rs else None
        aggregates.append(a)
    contrasts=[]
    for seed in seeds:
        rs={r['topology']:r for r in records if r['seed']==seed and r['status']=='evaluated'}
        if len(rs)!=4:continue
        v={t:rs[t]['test']['sharpe'] for t in TOPS}
        contrasts.append(dict(seed=seed,manager_without_peers=v['centralized']-v['independent'],
            manager_with_peers=v['hybrid']-v['decentralized'],peers_without_manager=v['decentralized']-v['independent'],
            peers_with_manager=v['hybrid']-v['centralized'],interaction=v['hybrid']-v['centralized']-v['decentralized']+v['independent']))
    OUT.mkdir(parents=True,exist_ok=True)
    audit=dict(protocol=plan,baselines=baselines,initial_pools=initial_pools,records=records,aggregates=aggregates,paired_contrasts=contrasts,
        confirmed_tokens=done['confirmed_tokens'],unconfirmed_requests=done['unconfirmed_requests'],usage=done['usage'],
        research_failures=done['research_failures'],evaluation_failures=done['evaluation_failures'])
    partial_path=ROOT/'partial-recovery/result.json'
    if partial_path.exists():
        partial=json.loads(partial_path.read_text())
        artifact=json.loads((ROOT/'partial-recovery/decentralized-s107/frozen.json').read_text())
        assert digest(artifact['payload'])==artifact['artifact_sha256']==partial['artifact_sha256']
        audit['supplemental_partial']=dict(evaluation=partial,protocol=artifact['payload']['protocol'])
    (OUT/'paired-results.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False))
    font=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,3,figsize=(16,9.8))
    fig.subplots_adjust(left=.055,right=.985,bottom=.13,top=.86,wspace=.27,hspace=.4)
    fig.suptitle('固定起点配对实验：验证演化与最终测试收益',fontsize=20,color='#233d49')
    for col,seed in enumerate(seeds):
        ax=axes[0,col]
        for j,top in enumerate(TOPS):
            r=next(r for r in records if r['seed']==seed and r['topology']==top)
            if r['curve']:ax.step(range(len(r['curve'])),r['curve'],where='post',color=COLORS[j],lw=1.7,marker='o',ms=3)
        ax.set(title=f'共同起点{col+1}：验证最佳Sharpe',xlabel='新增候选编号（0=公共起点）',ylabel='验证净Sharpe',xlim=(-.2,8.2))
        ax.set_xticks([0,2,4,6,8]);ax.grid(axis='y',color='#e4e9ec');ax.axhline(0,color='#b0bac0',ls='--',lw=.7)
        ax=axes[1,col]
        if seed in baselines:
            daily=pd.read_csv(ROOT/'hidden-evaluation'/f'baseline-s{seed}'/'hidden_daily.csv',parse_dates=['date','execution_date'])
            dates=pd.DatetimeIndex([daily.execution_date.iloc[0],*daily.date.tolist()]);values=np.r_[0.,np.cumprod(1+daily.net_return.to_numpy())-1]
            ax.plot(dates,values,color='#263b45',lw=2,ls='--',label='初始最佳')
            ax.set_xlim(dates[0],dates[-1])
        for j,top in enumerate(TOPS):
            r=next(r for r in records if r['seed']==seed and r['topology']==top)
            if r['status']=='evaluated':ax.plot(pd.to_datetime(r['dates']),r['cumulative'],color=COLORS[j],lw=1.4)
        ax.set(title=f'共同起点{col+1}：最终测试累计收益',xlabel='2024–2025日期',ylabel='累计净收益（%）')
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,7]));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.grid(axis='y',color='#e4e9ec');ax.axhline(0,color='#b0bac0',ls='--',lw=.7)
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],color=c,lw=2,label=NAMES[t]) for c,t in zip(COLORS,TOPS)]+[Line2D([0],[0],color='#263b45',ls='--',lw=2,label='初始最佳（测试基线）')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.945),ncol=5,frameon=False)
    fig.text(.5,.035,'来源：3组相同初始状态，四种拓扑各追加最多8次尝试；曲线只画实际完成的范围，不补齐或外推。\n测试策略包含初始候选；重合的曲线可能表示保留同一策略。每个面板纵轴自适应；测试已扣交易及借券费用。',ha='center',fontsize=9,color='#586a73',linespacing=1.7)
    for ext in ['png','svg']:fig.savefig(OUT/f'paired-curves.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    parts=[];labels=[]
    def add(identifier,title,body,label):parts.append(section(identifier,f'{len(parts)+1:02d}',title,body));labels.append(label)
    evaluated=[r for r in records if r['status']=='evaluated']
    complete=[r for r in evaluated if r.get('complete')]
    complete_blocks=sum(all(any(r['seed']==s and r['topology']==t and r.get('complete') for r in evaluated) for t in TOPS) for s in seeds)
    val_improved=sum(r['validation_gain']>1e-10 for r in evaluated)
    test_improved=sum(r['test_sharpe_change']>1e-10 for r in evaluated)
    add('overview','相同起点之后，反馈结构带来了什么增量？',
        f'<p>计划3组公共起点、12条延续分支。实际完成测试{len(evaluated)}/12，完整追加8次候选{len(complete)}/12。公共起点最佳策略另评估一次作为基线，所有研究结束且冻结后才统一评估。</p>'
        +f'<div class="note"><strong>本轮没有复现“混合反馈机制稳定领先”。</strong><p>{val_improved}/{len(evaluated)}条已评估分支改善了验证最佳值，但只有{test_improved}/{len(evaluated)}条提高了相对初始策略的测试Sharpe。混合三组测试Sharpe约−0.599、−0.915、1.194，效果强烈依赖起点。以下不同结构的可用组数不全相同，不能只按总均值排名。</p></div>'
        +table(['结构','完成/计划','完整8次','测试空仓','测试Sharpe均值','测试年化收益均值','测试MDD均值'],[
            [a['label'],f"{a['evaluated']}/3",f"{a['complete_eight']}/3",a['cash_test'],fmt(a['sharpe'],3),pct(a['annualized_return']),pct(a['max_drawdown'])] for a in aggregates])
        +table(['结构','验证改善次数','验证Sharpe增量均值','相对初始测试Sharpe变化均值'],[
            [a['label'],f"{a['improved']}/{a['evaluated']}",fmt(a['validation_gain'],4),fmt(a['test_sharpe_change'],3)] for a in aggregates])
        +f'<p>四分支均有主评估结果的完整起点为{len(contrasts)}/3；四分支都完成8次追加的起点为{complete_blocks}/3。所有均值只基于已评估分支，缺失条件不能当作零收益；不能把不足8次的结果说成机会完全相等。共用同一历史测试段，不提供稳定因果排名或显著性结论。</p>', '配对结果')
    add('design','本轮如何隔离初始候选差异？',
        '<p>三组公共起点分别由无管理者、无共享的4名研究者各生成一份策略并回测，包含失败记录。随后将同一组的代码、训练/验证结果和worker对应关系逐字复制到四个分支；初始状态SHA256逐一核对，原始候选可以一直保留为赢家。</p>'
        +table(['分支','自身历史','管理者分工','同伴有效结果'],[['独立','有','无','无'],['中心化','有','有','无'],['去中心化','有','无','有'],['混合','有','有','有']])
        +'<p>各分支按两轮、每轮W0–W3顺序生成，目标追加8次。管理者失败的轮次不额外补轮；策略错误也消耗候选次数。共享固定使用上一轮快照，自身最近3条结果；管理者与同伴只看最近有效结果，包含亏损和空仓，不共享错误详情。</p>'
        +'<p>与原V3不同：单请求总额度升至131,072；公共起点保护上限27万token、各延续分支40万，整批561万。目标是让候选机会尽量匹配，实际token作为结果，不要求花费相等。模型、行情、仓位、费用、执行时限与验证选优规则保持一致；没有增加新的策略家族限制。</p>'
        +'<p>本轮不检验管理者在最初分工中的价值；它只比较共同起点之后的延续。标签107/117/127不控制服务端随机数。三个起点都在任何本轮隐藏收益揭示前生成，没有按测试成绩挑选。</p>'
        +'<p>保持原V3选优规则：空仓候选允许存在，其Sharpe按回测记0，可能胜过负Sharpe候选。此类结果会单列空仓数量，不能把零回撤解释为发现了优质交易策略，也不会事后换起点或强制替换赢家。</p>', '实验设定')
    add('curves','验证演化与最终测试收益曲线',
        '<img src="paired-curves.svg" style="width:100%;height:auto" alt="三组共同起点的验证演化与测试累计收益"/>'
        +'<p>上排纵轴是历史最佳验证Sharpe；下排纵轴是最终策略的测试累计收益，黑色虚线是同一起点的初始最佳策略。上升的验证曲线不等于测试改进。</p><p><a href="paired-curves.png" download>下载高清PNG</a> · <a href="paired-curves.svg" download>下载SVG</a></p>','演化与收益')
    detail=''
    for i,seed in enumerate(seeds,1):
        detail+=f'<h3>共同起点{i} · 标签{seed}</h3>'
        if seed in initial_pools:
            pool=initial_pools[seed]
            detail+='<details><summary>四份公共初始候选与复制校验</summary>'
            detail+='<p>初始状态SHA256：<code>'+state_digest(pool)+'</code>。所有分支继承相同的代码、指标、错误及worker对应关系。</p>'
            detail+=table(['worker','训练Sharpe','验证Sharpe','验证持仓天数','执行错误'],[[f"W{t['agent']}",fmt(t.get('train',{}).get('sharpe'),3),fmt(t.get('validation',{}).get('sharpe'),3),t.get('validation',{}).get('active_days','—'),t.get('error','完成训练与验证')] for t in pool])
            for t in pool:
                detail+=f'<details><summary>初始W{t["agent"]}：假设与代码</summary><p>{escape(t["hypothesis"])}</p>'
                if 'strategy'in t:detail+='<pre><code>'+escape(t['strategy']['source'])+'</code></pre>'
                detail+='</details>'
            detail+='</details>'
        if seed in baselines:
            b=baselines[seed];m=b['evaluation']['hidden']
            detail+=f'<p>初始最佳：验证Sharpe {fmt(b["validation_sharpe"],4)}，测试Sharpe {fmt(m["sharpe"],3)}，测试累计净收益 {pct(m["total_return"])}，测试持仓天数{m["active_days"]}/500。</p>'
        rs=[r for r in records if r['seed']==seed]
        detail+=table(['结构','新增/有效','验证增量','保留初始策略','测试Sharpe','测试累计收益','测试年化收益','MDD','相对初始累计收益变化'],[
            [NAMES[r['topology']],f"{r['added']}/{r['valid']}",fmt(r.get('validation_gain'),4),'是' if r.get('winner_initial') else ('否' if r['status']=='evaluated' else '未完成'),fmt(r.get('test',{}).get('sharpe'),3),pct(r.get('test',{}).get('total_return')),pct(r.get('test',{}).get('annualized_return')),pct(r.get('test',{}).get('max_drawdown')),pct(r.get('test_return_change'))] for r in rs])
        for r in rs:
            detail+='<details><summary>'+NAMES[r['topology']]+'：最终代码、失败与停止原因</summary>'
            detail+='<p>'+escape(r['failure'] or ('完成8次追加' if r.get('complete') else '未完成8次追加'))+'</p>'
            detail+='<p>'+escape('; '.join(r.get('stop_reasons',[])))+'</p>'
            for e in r.get('errors',[]):detail+=f'<p>r{e["round"]}/W{e["worker"]}：{escape(e["error"])}</p>'
            if r['code']:detail+='<p>'+escape(r['hypothesis'])+'</p><pre><code>'+escape(r['code'])+'</code></pre>'
            detail+='</details>'
    add('blocks','三组起点逐一对照',detail,'逐组明细')
    if 'supplemental_partial' in audit:
        p=audit['supplemental_partial'];m=p['evaluation']['hidden'];v=p['evaluation']['validation'];pr=p['protocol']
        add('partial','补充：网络中断前已观测候选的冻结评估',
            '<p>第一组去中心化在6次追加后网络/TLS中断，最后一个请求没有用量回执。主实验仍记为失败，主均值和配对差值均不填入本补充结果。</p>'
            +f'<p>在任何本轮测试结果揭示前，于{escape(pr["recovery_declared_at_utc"])}确定并冻结补充规则：仅从已观测的初始及追加候选中，按最高验证Sharpe、SHA平分选优，不重试API、不新增候选。此规则与中断前日志哈希一起保存。</p>'
            +table(['已追加次数','验证Sharpe','测试Sharpe','测试累计收益','测试年化收益','测试MDD'],[[pr['observed_additional_attempts'],fmt(v['sharpe'],4),fmt(m['sharpe'],3),pct(m['total_return']),pct(m['annualized_return']),pct(m['max_drawdown'])]])
            +'<p>此项仅说明中断前方案的表现。它不是完成8次的分支，也不是预先约定的主分析，不能用它补成三个完整配对。</p>', '中断补充结果')
    names=dict(manager_without_peers='中心化−独立：无共享时的管理者对比',manager_with_peers='混合−去中心化：有共享时的管理者对比',peers_without_manager='去中心化−独立：无管理者时的共享对比',peers_with_manager='混合−中心化：有管理者时的共享对比',interaction='混合−中心化−去中心化+独立：交互对比')
    body=table(['对比','起点1','起点2','起点3','可用起点均值'],[[label,*[fmt(next((c[key] for c in contrasts if c['seed']==s),None),3) for s in seeds],fmt(float(np.mean([c[key] for c in contrasts])) if contrasts else None,3)] for key,label in names.items()])
    body+='<p>单位为测试Sharpe之差，不是收益百分点。只在四个分支都完成评估的起点上计算完整对比；其中个别分支可能未完成8次追加，差值包含协议可靠性与实际机会差异，不能只归因于反馈内容。初始状态匹配控制了起点，但生成随机性、实际token与后续策略风险仍会影响结果。三组不足以可靠估计总体效应，尤其不能仅根据正负号认定通信有益或有害。</p>'
    add('contrasts','配对对比：管理者与共享的增量',body,'配对差值')
    add('findings','这次预实验允许得出什么结论？',
        '<p><strong>结果层面：</strong>前两组起点中，所有有主评估结果的延续分支，测试Sharpe均低于初始最佳策略；第三组以空仓为基线，独立、去中心化、混合三条分支分别产生0.765、0.382、1.194的测试Sharpe。验证改善并没有稳定带来测试改善，增量具有明显起点依赖。</p>'
        +'<p><strong>机制层面：</strong>在唯一四个分支都完整追加8次的第二组，独立测试Sharpe0.027、去中心化0.004、中心化−0.577、混合−0.915；四份累计净收益均为负。这是一组具体反例，足以反对“混合在任何相同起点上都会更好”的说法，但一组不能估计稳定总体效应。</p>'
        +'<p><strong>补充结果另看：</strong>第一组去中心化在网络中断前6次追加所保留方案，补充测试Sharpe为0.922、年化15.89%、MDD22.01%，高于该组初始Sharpe0.493。它是有价值的延续方案线索，但没有完成8次，不能补入主配对或据此认定去中心化稳定最优。这个例外也说明“主分析未观察到增量”不等于所有中断前方案都没有价值。</p>'
        +'<p><strong>不能推出：</strong>管理者一定有害、共享一定有害、独立结构普遍最优，或者原V3三次结果无效。本轮排除了初始候选差异，却没有研究管理者初始分工的价值；请求额度、保护预算和固定两轮也与原V3不同。原V3的混合盈利结果仍成立，但“优势由管理者加共享造成”的机制解释没有得到稳定支持。</p>'
        +'<p><strong>下一步：</strong>先在独立小测试中提高管理者协议与网络可靠性，再做更多固定起点配对；预先保留新的时间窗口或前向验证，并把验证改进、测试增量和风险变化一起衡量。本次不自动追加新的API实验。</p>', '结论')
    initial_tokens=sum(v['confirmed_tokens'] for k,v in done['usage'].items() if k.startswith('baseline'))
    add('costs','费用、失败与实验边界',
        f'<p>公共起点已确认{initial_tokens:,} token，只计一次；延续分支{done["confirmed_tokens"]-initial_tokens:,}，全部合计 <strong>{done["confirmed_tokens"]:,}</strong>。另有{done["unconfirmed_requests"]}个请求没有用量回执。已确认量不一定是完整账单。</p>'
        +table(['结构','已确认token','新增候选','新增有效','每有效候选token'],[[a['label'],f"{a['tokens']:,}",a['added'],a['new_valid'],fmt(a['tokens']/a['new_valid'],0) if a['new_valid'] else '—'] for a in aggregates])
        +'<p>生成起点与延续机会不同于原V3，不能将本轮绝对均值直接并入原三次统计。市场约束仍为gross≤1、单股≤2%、净敞口≤1%；交易5bp、借券年化30bp。当前成分股历史池存在幸存者与未来成分信息偏差；历史测试期已被查看，不是新盲测。没有完整借券可得性或冲击模拟。</p>'
        +'<p><a href="paired-results.json">下载全部指标、曲线和协议JSON</a></p>','费用与限制')
    template=Path(__file__).with_name('open_report_template.html').read_text()
    template=re.sub(r'<title>.*?</title>','<title>QuantScale · 固定起点配对实验报告</title>',template)
    template=re.sub(r'<header>.*?</header>','<header><div class="eyebrow">QUANTSCALE / PAIRED START</div><h1>相同起点，<br>四种反馈结构继续搜索</h1><p class="intro">三组公共候选池 × 四种拓扑；区分初始策略差异、验证改善与测试增量。</p><div class="actions"><button onclick="window.print()">打印 / 保存PDF</button></div></header>',template,flags=re.S)
    template=re.sub(r'<footer>.*?</footer>','<footer>固定起点配对预实验 · 原V3结果不变 · 来源：独立保存的study-protocol、初始状态、研究日志与冻结评估。</footer>',template,flags=re.S)
    template=re.sub(r'<script\b[^>]*>.*?</script>','',template,flags=re.S)
    links=[]
    for i,(part,label) in enumerate(zip(parts,labels),1):
        identifier=re.search(r'id="([^"]+)"',part).group(1);links.append(f'<a href="#{identifier}">{i:02d}<span>{label}</span></a>')
    html=template.replace('@@NAV@@',''.join(links)).replace('@@BODY@@','\n'.join(parts))
    import base64
    html=html.replace('src="paired-curves.svg"','src="data:image/svg+xml;base64,'+base64.b64encode((OUT/'paired-curves.svg').read_bytes()).decode()+'"')
    assert '@@' not in html
    (OUT/'QuantScale-固定起点配对实验报告.html').write_text(html)
    print(json.dumps(dict(report=str((OUT/'QuantScale-固定起点配对实验报告.html').resolve()),aggregates=aggregates,contrasts=contrasts,confirmed_tokens=done['confirmed_tokens']),ensure_ascii=False,indent=2))


if __name__=='__main__':build()
