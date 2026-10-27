"""Latest overall-report sections, computed from existing V3 artifacts."""
from html import escape
import json
from pathlib import Path

from .build_html_report import table, fmt, pct
from .v3_evolution_notes import NOTES
from .compare_open_runs import ORDER, NAMES


def test_scorecard(records,base):
    import csv
    from pathlib import Path
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    rows=[];means=[]
    for r in records:
        rows.append(dict(topology=r['topology'],run=r['run'],repetition={7:1,17:2,27:3}[r['seed']],
            cumulative_net_return=r['test']['total_return'],annualized_net_return=r['test']['annualized_return'],
            test_sharpe=r['test']['sharpe'],max_drawdown=r['test']['max_drawdown']))
    for top in ORDER:
        subset=[r for r in rows if r['topology']==top]
        assert len(subset)==3
        means.append(dict(topology=top,repetitions=3,**{k:float(np.mean([r[k] for r in subset]))
            for k in ['cumulative_net_return','annualized_net_return','test_sharpe','max_drawdown']}))
    folder=Path(base)/'test-scorecard';folder.mkdir(exist_ok=True)
    for name,data in [('individual',rows),('three-run-means',means)]:
        with (folder/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    headers=['拓扑结构','累计净收益均值','年化净收益均值','Sharpe均值','MDD均值']
    cells=[[NAMES[r['topology']],pct(r['cumulative_net_return']),pct(r['annualized_net_return']),fmt(r['test_sharpe'],3),pct(r['max_drawdown'])] for r in means]
    font=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(12,4.1));ax.axis('off')
    fig.subplots_adjust(left=.025,right=.975,top=.79,bottom=.19)
    fig.suptitle('2024–2025测试集统一对比：每种拓扑三次指标的算术均值',fontsize=17,color='#233d49',y=.96)
    chart=ax.table(cellText=cells,colLabels=headers,cellLoc='center',loc='center',bbox=[0,0,1,1])
    chart.auto_set_font_size(False);chart.set_fontsize(12)
    for (i,j),cell in chart.get_celld().items():
        cell.set_edgecolor('#dce3e7');cell.set_linewidth(.6)
        if i==0:cell.set_facecolor('#f1f4f5');cell.set_text_props(weight='bold',color='#233d49')
        elif i==5:cell.set_facecolor('#edf5f6');cell.set_text_props(weight='bold',color='#176275')
        else:cell.set_facecolor('white')
    fig.text(.5,.065,'来源：15份冻结策略的原测试评估。已扣交易与借券成本；收益与Sharpe越高越好，MDD越低越好。\n均值不是组合策略表现；MDD显示为正的回撤幅度，年化按252日计算。',ha='center',fontsize=10,color='#586a73',linespacing=1.7)
    fig.savefig(folder/'v3-test-scorecard.png',dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)
    body='<p>统一使用<strong>2024–2025测试期、扣除交易与借券成本后的结果</strong>。累计收益覆盖完整测试段；年化收益便于按年比较。MDD为最大回撤，以下以正的跌幅列示，越小越好。</p>'
    body+='<h3>各拓扑三次运行的指标均值</h3>'+table(headers,cells)
    body+='<p>这些是三份策略各自指标的算术均值，不是将三份策略组合后重新计算的收益、Sharpe或MDD。三次共用相同测试时期，不能当作三个独立市场样本。</p>'
    body+='<h3>15份策略逐次对比</h3>'+table(['拓扑','运行','累计净收益','年化净收益','测试Sharpe','MDD'],[
        [NAMES[r['topology']],f"第{r['repetition']}次"+('（补跑）' if r['run']=='single-n1-s17-supplement' else ''),pct(r['cumulative_net_return']),pct(r['annualized_net_return']),fmt(r['test_sharpe'],3),pct(r['max_drawdown'])] for r in rows])
    highest=max(rows,key=lambda r:r['test_sharpe']);least=min(rows,key=lambda r:r['max_drawdown']);most=max(rows,key=lambda r:r['cumulative_net_return'])
    body+=f'<p><strong>单次最高Sharpe：</strong>{NAMES[highest["topology"]]}第{highest["repetition"]}次，{fmt(highest["test_sharpe"],3)}；<strong>单次最高累计收益：</strong>{NAMES[most["topology"]]}第{most["repetition"]}次，{pct(most["cumulative_net_return"])}；<strong>单次最小MDD：</strong>{NAMES[least["topology"]]}第{least["repetition"]}次，{pct(least["max_drawdown"])}。</p>'
    body+='<p><a href="test-scorecard/individual.csv" download>下载15份策略指标CSV</a> · <a href="test-scorecard/three-run-means.csv" download>下载三次均值CSV</a> · <a href="test-scorecard/v3-test-scorecard.png" download>下载汇总表PNG</a></p>'
    body+='<p class="source">来源：原evaluation.json的hidden指标，未重跑测试。Sharpe=日净收益均值÷样本标准差×√252，未扣无风险利率；MDD为复合净值相对此前峰值的最大跌幅。完整单次净值曲线见累计收益章节。</p>'
    return body


def full_results(records):
    best=max(records,key=lambda r:r['test']['sharpe'])
    most=max(records,key=lambda r:r['test']['total_return'])
    text=f'<div class="note"><strong>最高Sharpe与最高收益不是同一份策略。</strong><p>最高测试Sharpe：{escape(best["label"])}，{fmt(best["test"]["sharpe"],3)}；年化{pct(best["test"]["annualized_return"])}、累计{pct(best["test"]["total_return"])}、最大回撤{pct(best["test"]["max_drawdown"])}。最高累计收益：{escape(most["label"])}，{pct(most["test"]["total_return"])}，其Sharpe为{fmt(most["test"]["sharpe"],3)}，年化波动率{pct(most["test"]["volatility"])}。</p></div>'
    text+=table(['最终策略','实际代码机制','验证Sharpe','测试Sharpe','累计净收益','年化净收益','最大回撤'],[
        [r['label'],NOTES[r['run']][3],fmt(r['validation']['sharpe'],4),fmt(r['test']['sharpe'],3),pct(r['test']['total_return']),pct(r['test']['annualized_return']),pct(r['test']['max_drawdown'])] for r in records])
    text+='<details><summary>按2024、2025两个年度拆分</summary>'+table(['策略','2024累计净收益','2025累计净收益'],[[r['label'],pct(r['years']['2024']['total_return']),pct(r['years']['2025']['total_return'])] for r in records])+'</details>'
    text+='<p>每种拓扑每次选出一份策略，15份是分别回测的，不是同一个组合。累计收益是测试区间逐日净收益复合；年化按回测的252日约定换算，二者不可混用。分年度收益来自同一条连续回测路径，不是每年独立重选策略。</p>'
    return text


def constraints():
    return '''<p>当前环境应定义为：<strong>固定股票池、日线行情、近似美元中性、无预测模型训练的开放Python策略搜索</strong>。它开放了特征和信号公式，但没有开放所有投资任务。</p>''' + table(['限制','具体数值或要求','如何实施 / 对结果的影响'],[
        ['单股权重','|wᵢ|≤2%','程序强制检查；不是模型搜索得到的最优值，也不是必须每只都用满'],
        ['总敞口','Σ|wᵢ|≤100%','程序强制检查；不允许超过100%总名义敞口，可以低仓位或空仓'],
        ['净敞口','|Σwᵢ|≤1%','程序强制检查；基本限制为多空平衡，排除常规高仓位纯多头策略'],
        ['当日交易池','股价≥5美元、成交量≥10万股，且可交易','在固定500只历史池上按决策日数据筛选；不是历史逐日标普成员池'],
        ['行情接口','OHLC、复权开收盘、成交量、eligible','仅提供截至决策日的历史；Python模式不限制为300日，当前可读全部已提供过去记录'],
        ['外部信息','无财报、新闻、外部文件或网络','行情规则搜索，没有测试多模态、基本面或另类数据研究能力'],
        ['可选策略','无固定家族、参数网格、信号窗口菜单','不强制动量/反转、不强制等权或25/25持仓，也不强制周/月调仓'],
        ['预测训练','协议禁止训练预测模型','提示词要求并拦截常见fit/train接口；不是能识别所有手写算法的完整语义证明'],
        ['执行接口','generate_weights(history, universe)','返回每只当期股票唯一、有限的权重，包括零；代码≤128KB，限定数值库'],
        ['时间与协议','单决策10秒、每段回测180秒；管理者连续失败2轮停止','错误会减少实际研究机会；1GB是申请值，本批未实施严格RAM上限'],
        ['研究资源','每次17万总token、最多12候选','管理、上下文、失败和修复均计入；保守输入预留可能提前停止'],
        ['选优目标','最高验证扣费Sharpe','不是多目标优化，也不是最高测试Sharpe选优；研究者不接收测试成绩'],
    ]) + '''<h3>2%有必要吗？目前没有检验它是否最合适</h3>
    <p>它是约束集中度的基线。若用满100%总敞口，2%单股上限意味着至少50个非零持仓；对称的25只多头各+2%、25只空头各−2%只是一个实现。5%上限可容纳更集中配置，但放宽不保证提高泛化收益。</p>
    <p>我们尚未做2%/5%/10%的敏感性实验。应分开检验“固定信号、统一配仓器下的集中度影响”和“不同上限下重新evo的搜索变化”。现有代码常把2%写死，只改配置不会自动改变这些仓位；也不应根据已看过的测试表现挑选最好的上限。</p>
    <p>当前没有行业中性、公司合并敞口、Beta中性、换手上限或统一止损要求。美元净敞口为零仍可能有市场Beta；同一公司的不同股票类别也可能同时进入组合。</p>'''


def best_portfolio(base):
    path=Path(base)/'portfolio-snapshots/decentralized-third-target-weights.json'
    data=json.loads(path.read_text())
    text='''<p>去中心化第3次的最高Sharpe策略是<strong>风险调整后的12−1动量多空组合</strong>。计算约一年前至一个月前的价格涨幅，除以60日波动率，买排名前25、卖空后25，每只目标权重为±2%，两侧各50%。它没有直接按行业或公司进行中性化。</p>
    <p>以最近完成周五的数据构造信号；回测器仍每天按目标权重调整，所以“周五更新信号”不等于周内完全不交易。做空收益来自负权重乘股票回报：股票跌10%，−2%仓位贡献约+0.2%的组合毛收益；股票上涨则拖累收益，另扣交易和借券费用。</p>
    <div class="note"><strong>下面是重建的历史目标仓位，不是实盘成交或原始持仓流水。</strong><p>原评估保存了逐日收益和敞口，没有保存逐股持仓。此处用冻结的无状态策略、原数据接口和原股票池，仅重建三个日期的仓位，核对多空敞口。没有重跑收益评估，也没有改变策略。</p></div>'''
    for s in data['snapshots']:
        assert s['long_count']==s['short_count']==25
        assert abs(s['long_weight']-.5)<1e-9 and abs(s['short_weight']+.5)<1e-9
        text+=f'<details><summary>{s["execution_date"]}开盘目标仓位（决策日{s["decision_date"]}）</summary>'
        text+=table(['多头，每只+2%','空头，每只−2%'],[[escape(a),escape(b)] for a,b in zip(s['longs'],s['shorts'])])+'</details>'
    text+='<p>开头的多头包含NVDA、META、MSFT等，后期组合发生明显轮换；不能把最初50只股票当作两年固定持仓。样本使用当前成分股的历史行情，包含当时可能尚未进入标普的股票，不能解释为历史标普成员池的可复制组合。</p>'
    text+='<p><a href="portfolio-snapshots/decentralized-third-target-weights.csv" download>下载三个日期的完整权重CSV</a> · <a href="portfolio-snapshots/decentralized-third-target-weights.json">查看重建说明与冻结策略哈希</a></p>'
    return text
