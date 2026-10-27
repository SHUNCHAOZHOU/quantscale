"""Synthesize all three V3 repetitions from saved trials; no model or backtest calls."""
from collections import Counter
from html import escape
import json
from pathlib import Path
import re

import numpy as np

from .build_html_report import table, section, fmt, pct
from .compare_open_runs import ORDER, NAMES
from .report_topologies import topology_html
from .v3_evolution_notes import NOTES, TOPOLOGY_NOTES
from .plot_v3_search_sharpe import report_section as search_sharpe_section
from .plot_v3_cumulative_returns import report_section as cumulative_return_section
from .plot_v3_cumulative_returns import validation_report_section
from .overall_report_sections import full_results, constraints, best_portfolio, test_scorecard

BASE=Path('results/sp500-deepseek-open-pilot-v3')
COLORS=['#176275','#8798a2','#a46c32']


def ledger(record):
    root=Path(record['research_dir'])
    events=[json.loads(x) for x in (root/'trajectory.jsonl').read_text().splitlines()]
    requests={}
    cumulative=0;best=None;trials=[]
    for event in events:
        if event['kind']=='request':
            q=event['request']
            if q['role']=='researcher':requests[(q['round'],q['agent'])]=q
        elif event['kind']=='response':cumulative+=event['tokens']
        elif event['kind']=='trial':
            q=requests[(event['round'],event['agent'])]
            value=event.get('validation',{}).get('sharpe')
            if value is not None:best=value if best is None else max(best,value)
            trials.append(dict(index=len(trials)+1,round=event['round'],worker=event['agent'],
                strategy_id=event.get('strategy_id'),hypothesis=event.get('hypothesis',''),
                code=event.get('strategy',{}).get('source'),train=event.get('train'),validation=event.get('validation'),
                validation_sharpe=value,best_so_far=best,error=event.get('error'),tokens_so_far=cumulative,
                own_results=len(q['own_results']),own_valid=sum('validation' in x for x in q['own_results']),
                peers=len(q['peer_results']),direction=q.get('direction')))
    selection=json.loads((root/'selection.json').read_text())
    winning=next(t for t in trials if t['round']==selection['round'] and t['worker']==selection['agent'])
    assert winning['strategy_id']==selection['strategy_id']
    assert np.isclose(best,record['validation']['sharpe'])
    assert cumulative==record['tokens']
    return dict(run=record['run'],label=record['label'],topology=record['topology'],seed=record['seed'],
        trials=trials,winner_index=winning['index'],winner_context=winning,
        tokens=cumulative,post_winner_tokens=cumulative-winning['tokens_so_far'],
        post_winner_trials=len(trials)-winning['index'],
        manager_failed_rounds=sum(x['kind']=='manager_round_failed' for x in events),
        manager_tokens=sum(x['tokens'] for x in events if x['kind']=='response' and x['role']=='manager'),
        manager_exhausted=any(x['kind']=='manager_protocol_exhausted' for x in events),
        stop_reasons=[x['reason'] for x in events if x['kind'] in ['budget_exhausted','manager_protocol_exhausted']])


def curves(topology,records,ledgers):
    runs=[r for r in records if r['topology']==topology]
    values=[t['best_so_far'] for r in runs for t in ledgers[r['run']]['trials'] if t['best_so_far'] is not None]
    low=min(values);high=max(0,max(values));pad=max(.08,(high-low)*.12);low-=pad;high+=pad
    left,right,top,bottom=58,680,22,220
    x=lambda i:left+(i-1)*(right-left)/11
    y=lambda v:bottom-(v-low)*(bottom-top)/(high-low)
    elements=[]
    for v in np.linspace(low,high,5):
        elements.append(f'<line x1="{left}" x2="{right}" y1="{y(v):.2f}" y2="{y(v):.2f}" stroke="#e4e9ec"/><text x="48" y="{y(v)+4:.2f}" text-anchor="end" font-size="11" fill="#586a73">{v:.2f}</text>')
    for i in [1,4,8,12]:elements.append(f'<text x="{x(i):.2f}" y="245" text-anchor="middle" font-size="11" fill="#586a73">{i}</text>')
    elements.append('<text x="370" y="268" text-anchor="middle" font-size="11" fill="#586a73">候选编号（组内实际产生顺序，次）</text>')
    legend=[]
    for j,r in enumerate(runs):
        data=ledgers[r['run']];points=[t for t in data['trials'] if t['best_so_far'] is not None]
        path=' '.join(f"{x(t['index']):.2f},{y(t['best_so_far']):.2f}" for t in points)
        elements.append(f'<polyline points="{path}" fill="none" stroke="{COLORS[j]}" stroke-width="2.4"/>')
        for t in points:
            title=f"{r['label']} · 候选{t['index']} · r{t['round']}/W{t['worker']} · 当前最优{t['best_so_far']:.4f}"
            elements.append(f'<circle cx="{x(t["index"]):.2f}" cy="{y(t["best_so_far"]):.2f}" r="3" fill="{COLORS[j]}"><title>{escape(title)}</title></circle>')
        win=data['winner_context']
        elements.append(f'<circle cx="{x(win["index"]):.2f}" cy="{y(win["best_so_far"]):.2f}" r="6" fill="none" stroke="{COLORS[j]}" stroke-width="2"><title>最终赢家产生处</title></circle>')
        legend.append(f'<span><i style="background:{COLORS[j]}"></i>{escape(r["label"])} · {len(data["trials"])}次</span>')
    return '<div class="chart-box"><h3>'+NAMES[topology]+'：验证最优值的演化</h3><p class="chart-label">纵轴：截至当前候选的最高验证净Sharpe（无量纲）</p><svg viewBox="0 0 720 280" role="img" aria-label="'+NAMES[topology]+'三次搜索的验证最优值变化">'+''.join(elements)+'</svg><div class="legend">'+''.join(legend)+'</div><p class="source">来源：三次trajectory.jsonl；圆环标记最终赢家产生处。失败尝试仍占横轴位置；出现首个有效结果前不画线。不同结构纵轴范围不同，仅在同一图内比较曲线。</p></div>'


def build(overall=False):
    payload=json.loads((BASE/'v3-three-repetitions-results.json').read_text())
    analysis=json.loads((BASE/'v3-three-repetitions-analysis.json').read_text())
    records=[r for r in payload['attempts'] if r['status']=='evaluated']
    records.sort(key=lambda r:(ORDER.index(r['topology']),r['seed']))
    assert len(records)==15 and set(NOTES)=={r['run'] for r in records}
    traces={r['run']:ledger(r) for r in records}
    ar={r['run']:r for r in analysis['runs']}
    later=[r for r in records if ar[r['run']]['winner_round']>ar[r['run']]['first_round']]
    early=[r for r in records if r not in later]
    assert len(later)==10 and sum(r['test']['total_return']>0 for r in later)==3
    assert len(early)==5 and sum(r['test']['total_return']>0 for r in early)==4
    evolution=dict(scope='Saved V3 trials only; strategy code and LLM hypotheses are source evidence, not causal proof',
        source_results='v3-three-repetitions-results.json',runs=list(traces.values()),
        validation_test_correlation=float(np.corrcoef([r['validation']['sharpe'] for r in records],[r['test']['sharpe'] for r in records])[0,1]),
        interpretation_notes=NOTES)
    (BASE/'v3-evolution-ledger.json').write_text(json.dumps(evolution,ensure_ascii=False,indent=2,allow_nan=False))
    parts=[];labels=[]
    def add(identifier,title,body,label):
        parts.append(section(identifier,f'{len(parts)+1:02d}',title,body));labels.append(label)
    add('synthesis','三次运行之后，哪些结论更有把握？',
        '<div class="note"><strong>混合结构的盈利表现最一致；“反馈修订产生泛化收益”的证据仍然不足。</strong><p>三次混合测试Sharpe为0.845、0.796、0.603，均盈利；但前两次赢家来自后续修订，第三次来自首轮。中心化三次都改善了验证最佳值，却只有一次测试盈利。结构表现、演化过程与因果解释需要分别判断。</p></div>'
        +table(['拓扑','三次测试Sharpe','均值 / 样本标准差','盈利次数','年化收益均值','最大回撤均值'],[
            [a['label'],' / '.join(fmt(r['test']['sharpe'],3) for r in records if r['topology']==a['topology']),f"{fmt(a['test_sharpe_mean'],3)} / {fmt(a['test_sharpe_sd'],3)}",f"{a['positive']}/3",pct(a['test_annualized_return_mean']),pct(a['test_max_drawdown_mean'])] for a in payload['aggregates']])
        +'<p>这里的重复性指同一段市场数据上三次策略搜索结果的差异，不能等同于跨市场阶段的稳定性。均值为三份独立策略指标的算术均值，不是组合策略表现。单Agent的Sharpe标准差0.061比混合0.128更小，但三次均亏损；混合的优势是四种多研究者结构中较一致的盈利表现，不是全组方差最小。</p>'
        +'<h3>最重要的三条发现</h3><ol><li><strong>有演化，不等于有泛化。</strong>10次搜索的后续候选超过首个有效轮次的最佳值，最终仅3份测试盈利。验证提升只说明系统更适合已被反复查看的验证窗口。</li><li><strong>结构能影响分工与反馈，却没有锁定策略风格。</strong>不同结构都反复产生动量、反转、波动率和残差组合；同一结构也能产生不同的风险押注。</li><li><strong>实际搜索深度受运行机制约束。</strong>管理者格式、输出截断、策略执行错误、保守token预留和固定worker顺序都会改变有效研究机会，不能把全部差异归于“集体智能”。</li></ol>'
        +'<p class="source">范围：仅V3，标签7/17/27共15份测试策略。标签17单Agent使用用户要求的追加补跑，原网络失败及费用保留。本文为保存日志的事后综合分析，没有新增模型调用或策略回测。</p>', '核心结论')
    if overall:
        add('test-scorecard','测试集统一看板：收益率、Sharpe与MDD',test_scorecard(records,BASE),'测试集指标看板')
        add('all-results','15份最终策略：收益、风险与实际策略机制',full_results(records),'全部测试结果')
    add('design','实验中的evo到底是什么？不同拓扑改了哪条信息边？',
        '<p>这里的evo是<strong>LLM基于回测反馈重新提出假设、改写Python策略，然后按验证成绩保留最佳候选</strong>。模型参数不更新，没有预定义的遗传交叉或突变操作，也没有按收益复制worker。后续候选可以更差，软件最终保留的历史最佳值则不会下降。</p>'
        +'<ol class="experiment-flow"><li><b>提出</b><span>假设与完整策略代码</span></li><li><b>执行</b><span>代码检查、训练、验证</span></li><li><b>反馈</b><span>自身/同伴/管理者信息</span></li><li><b>修订</b><span>新假设或实现修正</span></li><li><b>冻结</b><span>验证最优的一份策略</span></li></ol>'
        +table(['结构','研究者 / 管理者','自身历史','同伴有效结果','管理者分工','主要可观察机制'],[
            ['单Agent','1 / 0','最近3条，含错误','无','无','单一分支的连续自我修订'],
            ['独立多Agent','4 / 0','各自最近3条，含错误','无','无','四条互不通信的探索分支'],
            ['中心化','4 / 1','各自最近3条，含错误','worker不直接读取','有','管理者集中消化证据、重新分工'],
            ['去中心化','4 / 0','各自最近3条，含错误','有','无','各自解释上一轮同伴的最近有效结果'],
            ['混合','4 / 1','各自最近3条，含错误','有','有','统一分工与同伴证据同时进入提案']])
        +topology_html()
        +'<h3>这些实现细节会改变你对“拓扑优势”的理解</h3><ul><li>管理者与同伴收到的是各worker最近一次<strong>完成训练与验证</strong>的结果，包含亏损和空仓候选；执行错误详情不共享。不是完整公共研究库。</li><li>首轮同伴上下文为空；管理者仍能在首轮提供分工。因此首轮混合赢家不可能受同伴回测影响，但可能受初始管理者任务设计影响。</li><li>worker在组内按编号顺序调用，上轮快照在整轮固定；不是实时群聊或组内四路并行。跨组并发才是4路。</li><li>每组最终只选一份策略，结果不能解释成多Agent自动分散投资风险。</li></ul>'
        +table(['设定','实际执行'],[['数据与划分','500只当前标普成分股Yahoo行情；2016–2021训练，2022–2023验证，2024–2025测试'],['搜索空间','自行编写Python特征、信号与仓位；无固定因子菜单。禁止预测模型训练、外部数据和文件访问'],['模型','DeepSeek Flash；研究者low/最大输出32,768，管理者none/最大输出8,192'],['预算','每次上限170,000总token、12候选；管理、上下文、失败与修复都消耗资源'],['交易限制','收盘决策，下一有效开盘执行，再下个开盘结算；交易费5bp、借券年化30bp；gross≤1、单股≤2%、净敞口≤1%'],['执行与数据质量','决策10秒，每段回测180秒，无严格RAM上限。跳过2021-05-05、2025-11-12不完整会话；500测试区间，按252年化']]),'实验与结构')
    if overall:
        add('constraints','搜索开放到了什么程度？2%等限制如何影响解释？',constraints(),'约束与2%')
    add('search-sharpe','Search过程中的Sharpe曲线',search_sharpe_section(),'Sharpe曲线')
    has_validation_paths=(BASE/'validation-curves/endpoints.json').exists()
    if has_validation_paths:
        add('validation-returns','最终选中策略在验证集上的累计净收益',validation_report_section(),'验证集收益')
    add('cumulative-returns','最终策略的测试期累计净收益',cumulative_return_section(),'累计收益')
    overview=table(['结构','后续赢家 / 3','相对首个有效轮次的验证增量','后续赢家测试盈利','含义'],[
        [NAMES[t],f"{sum(ar[r['run']]['winner_round']>ar[r['run']]['first_round'] for r in records if r['topology']==t)}/3",' / '.join(fmt(max(0,ar[r['run']]['winner_validation']-ar[r['run']]['first_best']),4) for r in records if r['topology']==t),f"{sum(r['test']['total_return']>0 for r in later if r['topology']==t)}/{sum(r['topology']==t for r in later)}",TOPOLOGY_NOTES[t][0]] for t in ORDER])
    overview+='<p>“首个有效轮次”是该次研究首次出现完成回测候选的一轮。单Agent一轮1份、多研究者一轮最多4份，起点探索覆盖不同；中心化·17为第1轮、单Agent·27为第1轮，其余为第0轮。增量取最终最佳减首个有效轮次最佳，因此下限为0；这不表示后续没有退步。</p>'
    overview+='<div class="note"><strong>后续赢家测试盈利3/10，首轮保留策略测试盈利4/5。</strong><p>这是对已选策略的事后分组，不是“早停优于进化”的对照实验。首轮与后续赢家不是随机分配，搜索难度与策略风格也不同。不能把这个比例变成每轮成功概率，更不能据此删除后续搜索。</p></div>'
    overview+=f'<p>15份策略的验证Sharpe与测试Sharpe横截面相关系数约 <strong>{fmt(evolution["validation_test_correlation"],3)}</strong>。在这个小样本中，验证排名没有提供明显的测试排序能力；这不是总体相关为零的统计证明。</p>'
    overview+=''.join(curves(t,records,traces) for t in ORDER)
    add('evo-overview','从轨迹看演化：改善、停滞与回退',overview,'演化总览')
    for topology in ORDER:
        rs=[r for r in records if r['topology']==topology]
        body='<p>'+TOPOLOGY_NOTES[topology][1]+'</p>'
        for r in rs:
            a=ar[r['run']];trace=traces[r['run']];note=NOTES[r['run']]
            body+=f'<h3>{escape(r["label"])}：{escape(note[0])}</h3><p>{note[1]}</p>'
            body+=table(['首轮最佳验证','最终验证','赢家轮次 / worker','可见自身有效 / 同伴','测试Sharpe','测试年化收益'],[[fmt(a['first_best'],4),fmt(r['validation']['sharpe'],4),f"r{a['winner_round']} / W{a['winner_worker']}",f"{a['winner_own_valid']} / {a['winner_peers']}",fmt(r['test']['sharpe'],3),pct(r['test']['annualized_return'])]])
            body+='<p><strong>最终代码机制：</strong>'+note[3]+'</p><p><strong>这次演化说明：</strong>'+note[2]+'</p>'
            body+='<details><summary>查看这次的逐候选记录、原始假设与赢家代码</summary>'
            body+=table(['候选','轮次/worker','训练Sharpe','验证Sharpe','当时最优','验证有持仓天数','执行结果'],[
                [t['index'],f"r{t['round']}/W{t['worker']}",fmt((t['train'] or {}).get('sharpe'),3),fmt(t['validation_sharpe'],4),fmt(t['best_so_far'],4),(t['validation'] or {}).get('active_days','—'),t['error'] or ('最终选中' if t['index']==trace['winner_index'] else '完成回测')] for t in trace['trials']])
            for t in trace['trials']:
                body+=f'<details><summary>候选{t["index"]} · r{t["round"]}/W{t["worker"]} · 验证{fmt(t["validation_sharpe"],4)}</summary><p>{escape(t["hypothesis"]) or "模型未提供完整假设"}</p><p>实际上下文：自身最近结果{t["own_results"]}条，其中有效{t["own_valid"]}条；同伴有效结果{t["peers"]}条；累计已确认用量{t["tokens_so_far"]:,} token。</p>'
                if t['direction']:body+='<p><strong>当次管理者分工原文：</strong>'+escape(t['direction'])+'</p>'
                body+='</details>'
            body+='<details class="strategy"><summary>最终冻结的完整Python代码</summary><pre><code>'+escape(r['strategy_source'])+'</code></pre></details></details>'
        body+='<p class="source">阶段方向根据原始hypothesis与逐轮反馈整理；胜出策略的机制已核读Python代码。模型对成本、因子方向或失效原因的解释是待验证诊断，并非因果事实。轮次从0开始。</p>'
        add('evo-'+topology,NAMES[topology]+'：'+TOPOLOGY_NOTES[topology][0],body,NAMES[topology]+'演化')
    add('mechanisms','跨结构看：变化的到底是策略、代码，还是解释？',
        '<h3>1. 常见演化动作是翻转方向、降频和控制风险</h3><p>动量表现不佳后改反转、反转成本高后降频、发现风险暴露后做残差化或Beta控制，在单Agent和无通信的独立组里也会发生。它们是模型在同一行情接口与净敞口约束下的共同搜索习惯，不能直接归因于协作。</p>'
        +'<h3>2. 经济修订与工程修复交织</h3><p>例如单Agent·27先修复标量错误，之后换经济信号，再修订缓存和权重平滑；混合·17的W2先出现仓位违规，下一轮才提出状态波动价差。一次验证改善可能同时受代码恢复可运行、信号变化、交易成本和仓位约束影响。本轮没有保存能把这些因素单独隔离的配对消融。</p>'
        +'<h3>3. 合理的解释未必是正确的诊断</h3><p>日志多次把问题归为“成本过高”“动量反了”或“短腿拖累”。这些解释促成了新代码，但同一方向在不同重复中的结果可以相反。假设里计划的基准、IC研究或额外实验，若没有相应trial记录，就不能算已经做过。</p>'
        +'<h3>4. 没有原样重复，不代表经济上不重复</h3><p>有效代码在各次运行内的AST哈希没有重复，说明没有原封不动提交同一模块。改一个窗口、加入风险过滤器或换调仓实现也会产生新哈希；这不能证明新因子、独立经济机制或新收益来源。尚未进行完整持仓重叠、多因子或信号相似性分析。</p>'
        +'<h3>5. 三次结果如何理解管理者与共享？</h3>'
        +table(['描述性对比','三次组均值差（Sharpe单位）'],[
            ['无共享：中心化−独立',fmt(analysis['factorial']['manager_without_peers'],3)],['有共享：混合−去中心化',fmt(analysis['factorial']['manager_with_peers'],3)],['无管理者：去中心化−独立',fmt(analysis['factorial']['peers_without_manager'],3)],['有管理者：混合−中心化',fmt(analysis['factorial']['peers_with_manager'],3)],['交互对比：混合−中心化−去中心化+独立',fmt(analysis['factorial']['interaction'],3)]])
        +'<p>这里对应2×2信息结构，但不是严格的因果估计。三次后的交互描述值约0.249，前两次约0.029；增加一次即可明显变化。不能据此认定出现了协同红利，也不能用之前接近零证明没有协同。每组n=3，实际预算、有效候选、协议失败和最终风险风格不同。</p>', '机制解释')
    risk=table(['策略','测试Sharpe','年化波动率','平均gross','SPY Beta','日均交易量/NAV','年化算术回归截距'],[
        [r['label'],fmt(r['test']['sharpe'],3),pct(r['test']['volatility']),fmt(ar[r['run']]['mean_gross_exposure'],2),fmt(ar[r['run']]['market_beta'],3),pct(ar[r['run']]['mean_turnover']),pct(ar[r['run']]['arithmetic_intercept_annualized'])] for r in records])
    risk+='<p><strong>混合的三次盈利来自不同风险结构。</strong>标签7的残差动量Beta约0.101、截距5.96%；标签17的状态波动价差Beta约0.407、截距−0.52%；标签27的多周期动量Beta约0.031、截距6.03%。美元净敞口接近零不代表市场Beta为零，不能把三份盈利都直接称为市场无关alpha。</p>'
    risk+='<p>中心化·17年化22.08%伴随Beta约0.573和约20%年化波动。第三次去中心化与独立组Beta约0.242和0.274；它们的高测试成绩也应结合风险看，不能只按收益率给组织结构排名。</p>'
    corr=[]
    for top in ORDER:
        names={r['run'] for r in records if r['topology']==top}
        for p in analysis['correlations']:
            if p['left'] in names and p['right'] in names:
                left=next(r for r in records if r['run']==p['left']);right=next(r for r in records if r['run']==p['right'])
                corr.append([NAMES[top],f"{left['seed']} / {right['seed']}",fmt(p['correlation'],3)])
    risk+='<details><summary>各拓扑三份策略的日净收益相关性</summary>'+table(['结构','标签组合','相关系数'],corr)+'</details>'
    risk+='<p>混合三对相关约0.223、0.378、0.737，具有相似与不同来源；去中心化三对包含负相关，不能仅凭其不稳定就断定发生了羊群效应。不同结构的单Agent·7与中心化·7相关约0.907，说明组织结构不同也可能产生高度相似的经济押注。</p>'
    risk+='<p class="source">风险表与相关性直接使用15份保存的hidden_daily.csv、500个共同观测。SPY以相同execution_date→date的复权开盘收益对齐；OLS含截距，年化算术截距=252×(策略日均收益−Beta×市场日均收益)。未做多因子和显著性检验；截距不等于复合超额收益，也不是可部署alpha的认证。</p>'
    add('risk','不同拓扑的盈利，是否其实来自不同的风险押注？',risk,'风险与重复')
    if overall:
        add('best-portfolio','最高Sharpe策略具体买什么、卖空什么？',best_portfolio(BASE),'最高Sharpe持仓')
    attempts=payload['attempts']
    budgetrows=[]
    for top in ORDER:
        spent=[r for r in attempts if r['topology']==top]
        known=sum(r['tokens'] for r in spent);valid=sum(r['valid'] for r in spent)
        budgetrows.append([NAMES[top],len(spent),f"{known:,}",sum(r['attempts'] for r in spent),valid,sum(r['active_valid'] for r in spent),fmt(known/valid,0),f"{sum(r['manager_tokens'] for r in spent):,}"])
    tailrows=[]
    for r in records:
        t=traces[r['run']]
        tailrows.append([r['label'],f"{t['winner_index']}/{len(t['trials'])}",t['post_winner_trials'],f"{t['post_winner_tokens']:,}",pct(t['post_winner_tokens']/t['tokens'])])
    add('resources','资源与失败：为什么相同上限不等于相同搜索机会？',
        table(['结构','研究尝试数','已确认token','候选尝试','有效候选','有持仓候选','每有效候选token','其中管理者token'],budgetrows)
        +'<p>全部V3活动共2,160,996个已确认token、144次候选尝试、114次有效回测，其中111次在训练与验证均有持仓。单Agent包含一次原网络失败，因此研究尝试数为4，其他各3；原失败另有1个请求没有回执，完整账单仍未知。推理token包含在输出中，不重复相加。</p>'
        +'<h3>运行限制确实进入了比较结果</h3><ul><li>第三次中心化因连续两轮管理者不完整JSON停止，只完成8个候选；有最终提交不代表管理协议完全正常。</li><li>独立组三次各12个候选，合计36个，最多有效候选27个；混合合计25个候选、21个有效，但三份最终策略测试均盈利。探索产量与最终策略质量不能互相替代。</li><li>保守输入预留按UTF-8字节加固定余量估计，含大量代码上下文时可能提前停止；不是实际token恰好用尽。共享与管理者上下文也占总预算。</li><li>worker固定编号顺序意味着额度不足时后面的worker可能没有下一次机会。例如混合·17后续只执行W0，W2的进一步修订未被尝试。</li></ul>'
        +'<details><summary>事后看：最终赢家出现后还花了多少预算？</summary>'+table(['策略','赢家候选编号/总数','之后候选数','之后已确认token','占本次用量'],tailrows)
        +'<p>统计从最终选中候选完成时到研究结束，包含后续管理调用。它说明赢家出现后实际发生了多少探索，不能叫作可节省成本：只有事后才知道后面没有更好的候选，也没有测试这些未选候选的隐藏收益。</p></details>', '预算与失败')
    add('limits','综合判断与下一步最有价值的检验',
        '<p><strong>目前最有依据的结论：</strong>混合实现是这15份已选策略里盈利表现最一致的结构；中心化擅长制造验证改善却没有稳定的测试优势；去中心化两份盈利都来自初始探索；独立多Agent有覆盖优势但选优不稳；单Agent三次均未测试盈利。</p>'
        +'<p><strong>目前证据不足的结论：</strong>混合稳定优于所有结构、管理者或同伴共享具有确定因果收益、三份混合策略具有相同alpha质量、后续演化必然有用或必然无用，均不能由本轮推出。</p>'
        +'<h3>优先检验三个具体问题</h3><ol><li><strong>反馈究竟改变了什么？</strong>预先生成并固定同一批初始候选，在独立、管理者、共享、混合四种反馈下延续搜索。初始人口一致后，后续增量更容易比较；模型仍有随机性，需要重复。此实验与当前“完整系统比较”的目标不同，应单独编号。</li><li><strong>改的是经济信号还是交易实现？</strong>对有代表性的路径做配对消融：固定信号只变调仓/缓冲；固定交易实现只变动量/反转方向。记录持仓与成本，核验模型的失败诊断。</li><li><strong>验证增量能否跨窗口复现？</strong>预先锁定选择规则与预算，换历史窗口并保留新的前向模拟；避免持续围绕2022–2023验证、2024–2025测试来调整设计。条件允许时换历史成分池。</li></ol>'
        +'<h3>阅读边界</h3><p>每个结构只有三份搜索结果，共用同一段测试期；三次不是三段独立市场样本。当前成分股带来幸存者和未来成分信息偏差。标签17单Agent有补跑，第三次实验发生在此前测试结果已被查看之后。实际token和执行机会不完全匹配，也未模拟市场冲击、真实借券可得性与完整融资条件。报告用于研究系统行为，不足以确定稳定组织排名或实盘结论。</p>'
        +'<p class="source">数据来源：pilot-protocol.json、各次protocol/trajectory/selection/frozen、保存的evaluation与hidden_daily，以及已冻结实现的信息流。完整演化账本保留原假设、代码、反馈上下文数量、逐候选指标与累计token，便于复核。</p>', '结论与后续')
    template=Path(__file__).with_name('open_report_template.html').read_text()
    template=re.sub(r'<title>.*?</title>','<title>QuantScale · V3三次运行综合分析：演化与拓扑</title>',template)
    header='<header><div class="eyebrow">QUANTSCALE / V3 / SYNTHESIS</div><h1>策略怎样演化，<br>拓扑改变了什么？</h1><p class="intro">三次运行的综合分析：15条搜索轨迹、最终策略、信息流与资源约束。以逐轮证据区分结果、过程与机制解释。</p><div class="meta"><span>5种拓扑 × 3次</span><span>DeepSeek Flash</span><span>500只股票</span><span>仅分析保存结果</span></div><div class="actions"><button onclick="window.print()">打印 / 保存PDF</button></div></header>'
    template=re.sub(r'<header>.*?</header>',lambda _:header,template,flags=re.S)
    template=re.sub(r'<footer>.*?</footer>','<footer>QuantScale · V3综合分析<p><a href="v3-evolution-ledger.json">完整演化账本JSON</a><a href="v3-three-repetitions-results.json">三次结果与成本</a><a href="v3-three-repetitions-analysis.json">风险与信息流分析</a><a href="QuantScale-V3-三次重复实验报告.html">三次结果汇总与交互净值</a></p></footer>',template,flags=re.S)
    template=re.sub(r'<noscript>.*?</noscript>','',template,flags=re.S)
    # This synthesis uses five static evolution figures; omit the old template's chart initialization.
    template=re.sub(r'<script\b[^>]*>.*?</script>','',template,flags=re.S)
    nav=[]
    for i,(part,label) in enumerate(zip(parts,labels),1):
        identifier=re.search(r'id="([^"]+)"',part).group(1)
        nav.append(f'<a href="#{identifier}">{i:02d}<span>{escape(label)}</span></a>')
    template=template.replace('@@NAV@@',''.join(nav)).replace('@@BODY@@','\n'.join(parts))
    if has_validation_paths:
        template=template.replace('本文为保存日志的事后综合分析，没有新增模型调用或策略回测。','本文为事后综合分析。仅为补充缺失的逐日路径重建了15份策略的验证回测，均核对原指标；未调用模型、重新选优或重跑测试期。')
        template=template.replace('所有分析来自已保存结果。','基于原实验结果，另对验证期逐日路径进行复现核对。')
        template=template.replace('全部V3活动共2,160,996个已确认token','原V3搜索活动共2,160,996个已确认token')
        template=template.replace('<h3>运行限制确实进入了比较结果</h3>','<p>验证收益图的15次路径复现不产生新候选，不计入原搜索次数；它们不调用模型，也不改变原费用统计。</p><h3>运行限制确实进入了比较结果</h3>')
    assert '@@' not in template and 'V2' not in template
    if overall:
        import base64
        template=template.replace('QuantScale · V3三次运行综合分析：演化与拓扑','QuantScale · V3整体实验报告')
        template=template.replace('策略怎样演化，<br>拓扑改变了什么？','V3整体实验报告：<br>结果、演化、策略与边界')
        template=template.replace('三次运行的综合分析：15条搜索轨迹、最终策略、信息流与资源约束。以逐轮证据区分结果、过程与机制解释。','5种拓扑、3次运行、15份最终策略。整合全部测试结果、搜索曲线、累计收益、实际持仓、token消耗与实验限制；所有分析来自已保存结果。')
        scope='<p><strong>统计口径：</strong>15个逻辑条件均已完成测试，其中7份策略累计盈利。成本计入16次实际研究尝试，包含原单Agent网络失败，共144个候选、114次有效回测、2,160,996个已确认token；另有1个请求用量未知。测试成绩保留每个条件的一份冻结策略，失败没有被记成零收益或从成本中删除。</p>'
        template=template.replace('<h3>最重要的三条发现</h3>',scope+'<h3>最重要的三条发现</h3>')
        tags='<p><strong>三次运行如何对应：</strong>第1次标签7、第2次标签17、第3次标签27；标签是seed字段中的重复标识，不是Agent数量、年份或搜索轮次，也没有控制DeepSeek服务端随机数。单Agent第2次来自独立补跑，其他四种结构沿用原第2次；第三次五种结构各新增一次。</p>'
        template=template.replace('<p>这里的evo是',tags+'<p>这里的evo是')
        template=template.replace('支持离线阅读、策略代码展开及打印。','图表已内嵌，支持离线阅读、代码展开和打印。数据下载链接需保留附件目录。')
        template=template.replace('初始人口一致后','初始候选池一致后')
        # Make the full report readable as a single offline HTML file.
        for src in re.findall(r'<img[^>]+src="([^"]+)"',template):
            asset=BASE/src
            assert asset.is_file()
            uri='data:image/svg+xml;base64,'+base64.b64encode(asset.read_bytes()).decode()
            template=template.replace('src="'+src+'"','src="'+uri+'"')
    out=BASE/('QuantScale-V3-整体实验报告.html' if overall else 'QuantScale-V3-三次运行综合分析报告.html');out.write_text(template)
    print(json.dumps(dict(report=str(out.resolve()),sections=len(parts),runs=len(traces),val_test_correlation=evolution['validation_test_correlation']),ensure_ascii=False))


if __name__=='__main__':build()
