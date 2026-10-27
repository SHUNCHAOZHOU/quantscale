"""Export validation Sharpe search curves from saved V3 trajectories."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

from .compare_open_runs import ORDER, NAMES

BASE=Path('results/sp500-deepseek-open-pilot-v3')
RUN_COLORS=['#176275','#8798a2','#a46c32']
TOPO_COLORS=['#334b58','#8798a2','#a46c32','#6a7b5a','#176275']


def export():
    ledger=json.loads((BASE/'v3-evolution-ledger.json').read_text())
    runs=ledger['runs']
    assert len(runs)==15
    font=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.titlesize':13,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','savefig.facecolor':'white'})
    output=BASE/'search-curves';output.mkdir(exist_ok=True)
    averages={}
    for topo in ORDER:
        rs=sorted((r for r in runs if r['topology']==topo),key=lambda r:r['seed'])
        length=min(len(r['trials']) for r in rs)
        avg=[]
        for i in range(length):
            values=[r['trials'][i]['best_so_far'] for r in rs]
            avg.append(float(np.mean(values)) if all(v is not None for v in values) else None)
        averages[topo]=dict(candidate=list(range(1,length+1)),mean_best_sharpe=avg,
                            note='All three runs observed; no extrapolation after the shortest run')
    for mode in ['best','candidate']:
        vals=[t['best_so_far' if mode=='best' else 'validation_sharpe'] for r in runs for t in r['trials']]
        vals=[v for v in vals if v is not None]
        lo=min(vals);hi=max(vals);pad=max(.1,(hi-lo)*.12)
        fig,axes=plt.subplots(2,3,figsize=(16,10.2))
        fig.subplots_adjust(left=.06,right=.99,bottom=.13,top=.88,hspace=.38,wspace=.25)
        fig.suptitle('V3搜索过程：'+('历史最佳验证Sharpe' if mode=='best' else '每个候选的验证Sharpe'),fontsize=20,color='#233d49')
        for topo,ax in zip(ORDER,axes.flat):
            rs=sorted((r for r in runs if r['topology']==topo),key=lambda r:r['seed'])
            for j,r in enumerate(rs):
                x=[t['index'] for t in r['trials']]
                values=[t['best_so_far' if mode=='best' else 'validation_sharpe'] for t in r['trials']]
                y=np.array([np.nan if v is None else v for v in values])
                if mode=='best':ax.step(x,y,where='post',color=RUN_COLORS[j],lw=1.8)
                else:ax.plot(x,y,color=RUN_COLORS[j],lw=1.4,alpha=.85)
                ax.scatter(x,y,color=RUN_COLORS[j],s=18,zorder=3)
                win=r['winner_context']
                ax.scatter([win['index']],[win['validation_sharpe']],s=90,facecolors='none',edgecolors=RUN_COLORS[j],lw=1.6,zorder=4)
                if mode=='candidate':
                    failures=[t['index'] for t in r['trials'] if t['validation_sharpe'] is None]
                    ax.scatter(failures,[-.035-j*.018]*len(failures),transform=ax.get_xaxis_transform(),color=RUN_COLORS[j],marker='x',s=22,clip_on=False)
            ax.set_title(NAMES[topo],loc='left',fontweight='bold',color='#233d49')
            ax.set(xlabel='候选编号（组内产生顺序，次）',ylabel='验证净Sharpe（无量纲）',xlim=(.6,12.4),ylim=(lo-pad,hi+pad))
            ax.set_xticks([1,4,8,12]);ax.axhline(0,color='#9aa7ae',ls='--',lw=.8)
            if mode=='candidate':
                ax.tick_params(axis='x',pad=22)
                ax.xaxis.labelpad=8
            ax.grid(axis='y',color='#e4e9ec',lw=.7);ax.set_axisbelow(True)
        ax=axes.flat[5]
        for j,topo in enumerate(ORDER):
            a=averages[topo];y=[np.nan if v is None else v for v in a['mean_best_sharpe']]
            ax.step(a['candidate'],y,where='post',color=TOPO_COLORS[j],lw=1.7,label=NAMES[topo])
            ax.scatter(a['candidate'],y,color=TOPO_COLORS[j],s=13)
        ax.set_title('结构对比：三次历史最佳值的均值',loc='left',fontweight='bold',color='#233d49')
        ax.set(xlabel='候选编号（仅三次均有观测的区间）',ylabel='平均验证净Sharpe（无量纲）',xlim=(.6,12.4),ylim=(lo-pad,hi+pad))
        ax.set_xticks([1,4,8,12]);ax.axhline(0,color='#9aa7ae',ls='--',lw=.8);ax.grid(axis='y',color='#e4e9ec',lw=.7);ax.legend(fontsize=8,loc='lower right',frameon=False)
        handles=[Line2D([0],[0],color=c,lw=2,marker='o',label=label) for c,label in zip(RUN_COLORS,['第1次 · 标签7','第2次 · 标签17（单Agent为补跑）','第3次 · 标签27'])]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False,fontsize=10)
        caption='来源：V3三次搜索日志；验证期2022–2023。圆环=最终选中候选。组内四名worker按编号顺序产生候选。'
        if mode=='candidate':caption+='\n轴下×=失败，无Sharpe，不计为0；折线在缺失处断开。'
        else:caption+='\n失败尝试占用横轴位置；已有最佳值延续，首个有效值之前留空。'
        fig.text(.5,.035,caption,ha='center',fontsize=9,color='#586a73',linespacing=1.8)
        for ext in ['png','svg']:
            fig.savefig(output/f'v3-validation-sharpe-{mode}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    (output/'mean-best-validation-sharpe.json').write_text(json.dumps(averages,ensure_ascii=False,indent=2,allow_nan=False))
    print(str(output.resolve()))


def report_section():
    return '''<p>横轴是组内实际产生的候选编号，纵轴是<strong>2022–2023验证期、扣除交易与借券成本后的Sharpe</strong>。每种拓扑有三条曲线，颜色对应三次运行。下图可直接比较五种结构；第六格展示三次历史最佳值的均值。</p>
    <div class="chart-box"><h3>历史最佳Sharpe：搜索有没有找到更好的候选？</h3><img src="search-curves/v3-validation-sharpe-best.svg" style="width:100%;height:auto" alt="五种拓扑三次搜索的历史最佳验证Sharpe曲线，以及三次均值对比"/><p>曲线上升表示发现了更高验证Sharpe的候选；横向平台表示继续搜索尚未超过历史最佳。圆环是最终赢家出现的位置。</p><p><a href="search-curves/v3-validation-sharpe-best.svg" download>下载矢量SVG</a> · <a href="search-curves/v3-validation-sharpe-best.png" download>下载高清PNG</a></p></div>
    <details><summary>展开：每个候选本身的Sharpe（查看退步和波动）</summary><div class="chart-box"><h3>候选Sharpe：每一步实际生成了什么成绩？</h3><img src="search-curves/v3-validation-sharpe-candidate.svg" style="width:100%;height:auto" alt="五种拓扑每个候选的验证Sharpe原始曲线"/><p>这条曲线可以下降。失败候选没有Sharpe，保留横轴位置、在轴下以×标注，折线断开，不把失败填成0。验证期空仓的候选按回测记录显示为0，不能视为有效交易信号。</p><p><a href="search-curves/v3-validation-sharpe-candidate.svg" download>下载矢量SVG</a> · <a href="search-curves/v3-validation-sharpe-candidate.png" download>下载高清PNG</a></p></div></details>
    <div class="note"><strong>这是搜索曲线，不是测试期收益曲线。</strong><p>未选中的中间候选没有进行测试期评估，因此不能画它们的测试Sharpe演化。汇总均值只保留三次都实际运行到、且都有历史最佳值的编号；较长运行的尾部不外推，不能将均值曲线末点当成三份最终赢家的平均验证成绩。</p></div>
    <p class="source">五个单结构面板使用相同纵轴尺度，均值面板也同尺度；两张图的纵轴范围随所展示的数据分别确定。相同候选数不代表相同token；多研究者一轮最多4份、单Agent一轮1份，这里按候选顺序而不是轮次对齐。所有图片由保存的逐候选记录生成，没有新增API调用或回测。</p>'''


if __name__=='__main__':export()
