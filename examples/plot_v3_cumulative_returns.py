"""Plot saved out-of-sample daily returns for the 15 frozen V3 strategies."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from .compare_open_runs import ORDER, NAMES
from .build_html_report import table, pct

BASE=Path('results/sp500-deepseek-open-pilot-v3')
RUN_COLORS=['#176275','#8798a2','#a46c32']
TOPO_COLORS=['#334b58','#8798a2','#a46c32','#6a7b5a','#176275']


def export(validation=False):
    payload=json.loads((BASE/'v3-three-repetitions-results.json').read_text())
    records=[r for r in payload['attempts'] if r['status']=='evaluated']
    records.sort(key=lambda r:(ORDER.index(r['topology']),r['seed']))
    if validation:
        reconstruction=json.loads((BASE/'validation-curves/completed.json').read_text())
        assert reconstruction['all_matched'] and reconstruction['runs']==15
    directory='validation-curves' if validation else 'return-curves'
    stem='v3-validation-cumulative-returns' if validation else 'v3-test-cumulative-returns'
    period='2022–2023验证期' if validation else '2024–2025测试期'
    paths={};dates=None;ends=[]
    for r in records:
        source=(BASE/'validation-curves/daily'/r['run']/'validation_daily.csv') if validation else (Path(r['evaluation_dir'])/'hidden_daily.csv')
        daily=pd.read_csv(source,parse_dates=['date','execution_date'])
        assert len(daily)==500
        assert np.allclose(daily.net_return,daily.gross_return-daily.cost-daily.borrow_cost)
        cumulative=np.r_[0.,np.cumprod(1+daily.net_return.to_numpy())-1]
        time=pd.DatetimeIndex([daily.execution_date.iloc[0],*daily.date.tolist()])
        assert time.is_monotonic_increasing and time.is_unique
        assert np.isclose(cumulative[-1],r['validation' if validation else 'test']['total_return'])
        if dates is None:dates=time
        else:assert dates.equals(time)
        paths[r['run']]=cumulative
        ends.append(dict(run=r['run'],label=r['label'],topology=r['topology'],seed=r['seed'],total_return=float(cumulative[-1])))
    assert len(paths)==15
    font=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.titlesize':13,
                         'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,3,figsize=(16,10.2))
    fig.subplots_adjust(left=.06,right=.99,bottom=.13,top=.88,hspace=.38,wspace=.25)
    fig.suptitle('V3最终策略：'+period+'累计净收益',fontsize=20,color='#233d49')
    lo=min(float(v.min()) for v in paths.values());hi=max(float(v.max()) for v in paths.values());pad=(hi-lo)*.08
    for topology,ax in zip(ORDER,axes.flat):
        rs=[r for r in records if r['topology']==topology]
        for j,r in enumerate(rs):
            ax.plot(dates,paths[r['run']],color=RUN_COLORS[j],lw=1.4)
            ax.scatter([dates[-1]],[paths[r['run']][-1]],color=RUN_COLORS[j],s=17,zorder=3)
        ax.set_title(NAMES[topology],loc='left',fontweight='bold',color='#233d49')
    ax=axes.flat[5]
    for i,topology in enumerate(ORDER):
        mean=np.mean([paths[r['run']] for r in records if r['topology']==topology],axis=0)
        ax.plot(dates,mean,color=TOPO_COLORS[i],lw=1.6,label=NAMES[topology])
    ax.set_title('三次累计净收益的均值（描述性）',loc='left',fontweight='bold',color='#233d49')
    ax.legend(loc='upper left',fontsize=8,frameon=False)
    for ax in axes.flat:
        ax.set(xlabel='日期',ylabel='累计净收益（%）',xlim=(dates[0],dates[-1]),ylim=(lo-pad,hi+pad))
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.axhline(0,color='#9aa7ae',ls='--',lw=.8)
        ax.grid(axis='y',color='#e4e9ec',lw=.7);ax.set_axisbelow(True)
    handles=[Line2D([0],[0],color=c,lw=2,label=label) for c,label in zip(RUN_COLORS,['第1次','第2次（单Agent为补跑）','第3次'])]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False,fontsize=10)
    caption=('来源：15份冻结策略的验证路径重建，全部指标复现原验证结果；' if validation else '来源：15份冻结策略保存的逐日测试收益；')
    caption+='500个共同持有区间，累计收益 = ∏(1 + 日净收益) − 1。\n已扣交易费5bp与借券年化30bp；起点0%，六图同尺度。右下为三次累计收益均值。'
    if validation:caption+=' 验证集参与搜索与选优，并非新的样本外成绩。'
    fig.text(.5,.035,caption,ha='center',fontsize=9,color='#586a73',linespacing=1.8)
    out=BASE/directory;out.mkdir(exist_ok=True)
    for ext in ['png','svg']:fig.savefig(out/f'{stem}.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    pd.DataFrame(paths,index=dates).rename_axis('date').to_csv(out/f'{stem}.csv')
    (out/'endpoints.json').write_text(json.dumps(ends,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(ends,ensure_ascii=False,indent=2))


def report_section(validation=False):
    directory='validation-curves' if validation else 'return-curves'
    stem='v3-validation-cumulative-returns' if validation else 'v3-test-cumulative-returns'
    period='2022–2023验证期' if validation else '2024–2025测试期'
    prefix='下图用原冻结策略与原研究数据重建了15份最终策略的' if validation else '下图直接使用15份最终冻结策略保存的'
    rows=json.loads((BASE/directory/'endpoints.json').read_text())
    body=f'''<p>{prefix}<strong>{period}逐日净收益</strong>。纵轴从0%起，按复利计算累计收益，不是日收益直接相加，也不是年化收益。</p>
    <div class="chart-box"><h3>五种拓扑、三次运行的{period}累计净收益</h3><img src="{directory}/{stem}.svg" style="width:100%;height:auto" alt="五种拓扑各三份最终策略的{period}累计净收益，以及三次均值对比"/><p><a href="{directory}/{stem}.svg" download>下载矢量SVG</a> · <a href="{directory}/{stem}.png" download>下载高清PNG</a> · <a href="{directory}/{stem}.csv" download>下载15条曲线CSV</a></p></div>'''
    body+=table(['结构','第1次累计收益','第2次累计收益','第3次累计收益'],[[NAMES[t],*[pct(r['total_return']) for r in sorted(rows,key=lambda r:r['seed']) if r['topology']==t]] for t in ORDER])
    if validation:
        body+='<div class="note"><strong>验证期参与了反复搜索与最终选优。</strong><p>此图不是新的样本外检验。原研究只保存验证汇总指标，未保存逐日路径；为本图仅重建验证回测，15份策略的全部验证指标均与原记录在数值容差内一致。没有调用模型、重新选优或重跑测试期。</p></div>'
        body+='<p class="source">来源：validation-curves/daily下重建的validation_daily.csv，每条500个共同持有区间；重建核对见completed.json。已扣交易与借券成本，未模拟冲击及完整融资条件。右下三次均值用于描述，未重新组合或再平衡。</p>'
    else:
        body+='<p class="source">来源：保存的hidden_daily.csv，每条500个共同测试持有区间；按相同开盘执行与结算日期对齐，终点逐一核对evaluation.json。已扣交易与借券成本，未模拟冲击及完整融资条件。右下三次均值用于描述，未重新组合、再平衡或选优。未新增API调用或回测。</p>'
    return body


def validation_report_section():
    return report_section(validation=True)


if __name__=='__main__':export()
