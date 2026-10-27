"""Chinese interpretation of the audited V3 topology traces and saved returns."""
from .build_html_report import table, section, fmt, pct


def sections(data, comparison):
    rows=data['runs']; valid=[r for r in rows if r['status']=='evaluated']
    lookup={r['run']:r for r in rows}
    single_complete=[r for r in valid if r['topology']=='single']
    single_comment=('单Agent：</strong>唯一完成的运行有明显自我修订，但测试失败；另一次因网络中断。它说明独立迭代也能提高验证成绩，不能把混合组的验证改善全部归功于通信；也没有足够完整重复来判断单Agent整体稳定性。'
                    if len(single_complete)==1 else
                    '单Agent：</strong>补跑后可观察两份完整结果，但其中一份来自追加研究尝试，原网络失败的费用和未知用量仍须计入。单Agent·7有明显的验证改善却测试失败，说明自我迭代也能提升验证成绩；补全后的均值仍不足以证明稳定的单Agent能力。')
    pair={(x['left'],x['right']):x['correlation'] for x in data['correlations']}
    def correlation(left,right):
        return pair.get((left,right),pair.get((right,left)))
    means={r['topology']:r['test_sharpe_mean'] for r in comparison['aggregates'] if r['tested']}
    f=data['factorial']
    structural=section('topology-mechanisms','01','把排名拆开：管理者与共享分别对应什么？',
        '<p>四种多研究者结构可以按“有无管理者 × 有无同伴共享”理解；单Agent则是额外的单研究者基线。这样能组织比较，但本轮不是严格的随机因子试验，下面的差值只是已选策略的历史描述。</p>'
        +table(['管理者','同伴共享','结构','测试Sharpe均值'],[
            ['无','无','独立并行',fmt(means['independent'],3)],['有','无','中心化',fmt(means['centralized'],3)],
            ['无','有','去中心化',fmt(means['decentralized'],3)],['有','有','混合',fmt(means['hybrid'],3)]])
        +table(['描述性比较','组均值差（Sharpe单位）'],[
            ['无共享：中心化 − 独立并行',fmt(f['manager_without_peers'],3)],
            ['有共享：混合 − 去中心化',fmt(f['manager_with_peers'],3)],
            ['无管理者：去中心化 − 独立并行',fmt(f['peers_without_manager'],3)],
            ['有管理者：混合 − 中心化',fmt(f['peers_with_manager'],3)],
            ['交互对比：混合 − 中心化 − 去中心化 + 独立并行',fmt(f['interaction'],3)]])
        +'<div class="note"><strong>混合最好，不等于已经发现了额外的“混合协同红利”。</strong><p>本轮交互对比仅约0.029，四组均值的形态接近加性叠加。管理者相关对比约1.07–1.10，共享相关对比约0.55–0.57，但不能将这些数当作因果效应或推广到未来。也不能用接近零的交互对比证明协同不存在；没有进行统计显著性确认。</p></div>'
        +'<p>每组仅两次，实际token和有效候选数不同，管理者还增加一个不同推理设置的角色；最终策略的经济方向也不同。因此，结构差异、预算分配、策略风格和随机性混在一起。这里最有价值的是提出后续消融问题，而不是给管理者或同伴信息“定价”。</p>'
        +'<h3>逐个结构，更具体地理解</h3>'
        +'<p><strong>'+single_comment+'</p>'
        +'<p><strong>独立并行：</strong>20个有效候选、每个约1.34万token，产出便宜且多，但最终两份测试都亏损。当前问题不只是“探索不够”，而是独立生成与验证选优没有稳定找到泛化方向。尚无证据证明所有损失来自重复搜索。</p>'
        +'<p><strong>中心化：</strong>两次测试Sharpe为−0.61和1.10，均值0.25掩盖了很大的分歧。管理者有组织修订方向的轨迹，但一份选到反转方向，另一份选到含高Beta与状态切换的动量方向；不能把均值看成稳定的中心化收益。</p>'
        +'<p><strong>去中心化：</strong>一次赢家来自首轮，无同伴信息；另一次收到一条同伴有效结果，却只有极小验证改善，最终测试−1.44。当前更准确的说法是“共享没有在这两次运行中稳定创造超越首轮的候选”，而不是“大家互相抄袭导致失败”。</p>'
        +'<p><strong>混合：</strong>两次都在后续轮次产生赢家，均收到管理者分工与三条同伴有效结果，两份测试均盈利。它是本轮最值得继续验证的机制线索；不过，自身反馈、同伴反馈、管理者分工同时存在，还无法分离各自贡献。</p>'
        +'<p class="source">来源：V3逐次评估和信息流配置。Sharpe均值的差值不是收益率差值，也不是组合策略Sharpe。比较没有控制底层金融因子，不是严格随机的2×2因果试验。</p>')
    dynamics=section('search-dynamics','02','过程证据：共享后有没有超过首轮？',
        table(['运行','首个有效轮次','该轮有效数','该轮最佳验证Sharpe','后续有效数','后续最佳验证Sharpe','后续−首轮最佳'],[
            [r['label'],r['first_round'],r['first_valid'],fmt(r['first_best'],4),r['later_valid'],fmt(r['later_best'],4),fmt(r.get('later_minus_first_best'),4)] for r in rows])
        +'<p class="source">“首轮”是该组首个产生有效回测的轮次；中心化·17实际为第1轮，因为第0轮管理者格式失败。单Agent每轮1次，多研究者每轮最多4次，所以首轮候选数并不匹配。空白是没有后续有效候选，不填零。这里比较后续候选的最佳值，最终全局最佳值不会因加入较差候选而下降。</p>'
        +'<h3>混合：出现了两次可追踪的方向修订</h3>'
        +'<p>混合·7从首轮最佳−0.577提升到后续0.069；赢家由W2从快速残差反转转向月末残差动量。混合·17从−0.456提升到0.040；W2先前因仓位限制没有有效结果，随后提出月度状态切换的高低波动率多空策略。两次赢家均在第1轮，收到三条同伴的亏损回测和管理者分工。</p>'
        +'<p>这支持“系统能够依据已有失败经济证据提出不同方向”的观察，却不证明同伴消息是必要原因。混合·7同时有自己的负反馈；混合·17也有自己的仓位错误反馈。没有相同任务下的共享关闭对照，无法排除自行修订或管理者单独分工也能产生同一方案。</p>'
        +'<h3>去中心化：信息共享存在，但边际搜索改进很弱</h3>'
        +'<p>去中心化·17的后续最佳0.139没有超过首轮0.216，最后保留首轮独立提案。去中心化·7仅从−0.0155变为−0.0119，改进约0.0036。不能据此证明共享有害，但这两次没有显示稳定的增量发现能力。</p>'
        +'<h3>验证改善不是协作独有，也不是测试盈利的保证</h3>'
        +'<p>单Agent·7从首轮−0.245提高到0.432，幅度约0.678，却在测试期得到−0.837。中心化·7和独立并行·7也改善了验证最佳值，但最终测试为负。因此，仅用“后续验证改善幅度”衡量拓扑，会混淆自我反馈、更多选优机会和真正泛化能力。</p>'
        +'<details><summary>展开胜出候选的实际信息来源与各worker尝试数</summary>'
        +table(['运行','胜出worker / 轮次','可见自身有效结果','可见同伴有效结果','各worker尝试数'],[
            [r['label'],f"W{r['winner_worker']} / {r['winner_round']}" if 'winner_worker'in r else '无最终提交',r.get('winner_own_valid','—'),r.get('winner_peers','—'),', '.join(f'W{k}: {v}' for k,v in sorted(r['worker_attempts'].items()))] for r in rows])+'</details>'
        +'<h3>当前信息结构还有两项具体限制</h3><ul><li><strong>共享的是完成回测的候选，包括亏损策略；执行错误不会共享。</strong>管理者和同伴只拿各W最近一次有效结果，无法直接看见仓位违规、超时、禁用函数等失败细节。混合·17的W2首轮仓位违规，管理者下一轮只知道没有有效结果；W2自己才持有具体错误。</li><li><strong>信息结构之外还有固定的资源调度。</strong>混合·17后来管理者已提出继续改善W2方案，但最后执行轮次只调用W0一次，W2没有获得新尝试。预算、上下文预留与固定顺序共同限制后续探索，不能把没有继续改进全归于研究者能力。</li></ul>'
        +'<p class="source">来源：trajectory.jsonl中与候选对应的request、trial、manager_broadcast。研究者只持有自身最近3条结果，管理者和同伴只收到上一轮快照；这描述的是当前实现，不代表所有中心化/去中心化设计的能力上界。</p>')
    examples=[('hybrid-n4-s7','hybrid-n4-s17'),('centralized-n4-s7','centralized-n4-s17'),
              ('decentralized-n4-s7','decentralized-n4-s17'),('independent-n4-s7','independent-n4-s17'),
              ('single-n1-s7','centralized-n4-s7'),('single-n1-s7','decentralized-n4-s7'),('single-n1-s7','independent-n4-s17')]
    other_single=next((r['run'] for r in single_complete if r['run']!='single-n1-s7'),None)
    if other_single:
        examples.insert(0,('single-n1-s7',other_single))
    risk=section('topology-risk','03','收益差异有多少可能来自策略风格与风险？',
        table(['策略一','策略二','测试日净收益相关系数'],[[lookup[a]['label'],lookup[b]['label'],fmt(correlation(a,b),3)] for a,b in examples])
        +'<p><strong>不同拓扑可以产生非常相似的收益。</strong>单Agent·7与中心化·7相关0.907，两者都在测试期亏损，策略描述均以较慢反转为主。这说明“不同组织结构”不保证“不同经济押注”。</p>'
        +'<p><strong>同一拓扑也可以产生完全不同的方向。</strong>中心化两份相关−0.612，去中心化两份仅0.082。去中心化这轮的差表现不能简单归因于同质化；它具有收益方向差异，却没有稳定筛选出两份都盈利的策略。相关性不能替代持仓或因子归因。</p>'
        +'<p><strong>混合的两份赢家并非收益近乎相同的重复。</strong>相关0.378，且两份策略各自2024和2025年度切片都为正。代码核读显示，混合·7主要为残差动量，混合·17则是市场趋势控制的高低波动率多空，两者机制和风险不同。</p>'
        +table(['策略','测试Sharpe','年化波动率','平均总敞口','SPY Beta','日均交易量/NAV','单市场回归年化算术截距'],[
            [r['label'],fmt(r['test']['sharpe']),pct(r['test']['volatility']),fmt(r['mean_gross_exposure'],3),fmt(r['market_beta'],3),pct(r['mean_turnover']),pct(r['arithmetic_intercept_annualized'])] for r in valid])
        +'<div class="note"><strong>两份混合策略都赚钱，但不能把它们都视作市场无关alpha。</strong><p>混合·7相对SPY的Beta约0.101、单市场回归年化算术截距约5.96%；混合·17平均总敞口仅0.5，但Beta约0.407，回归截距约−0.52%，R²约0.442。后者虽然净美元敞口接近零，仍有较明显市场风险。不能只凭约8%的年化收益，就认为两份策略具有相同的超额收益质量。</p></div>'
        +'<p>中心化·17的22.08%年化收益同样伴随约20.0%波动率、23.86%最大回撤和约0.573的市场Beta。与混合组比较时，应同时看风险，而不是只按最高收益率给拓扑排名。</p>'
        +f'<p class="source">来源：{len(valid)}份已保存hidden_daily.csv的500个共同观测。SPY使用相同execution_date→date的复权开盘收益；OLS含截距，年化算术截距=252×(策略日均收益−Beta×市场日均收益)。仅做事后描述，没有多因子、显著性或因果确认；截距不等于复合超额收益。此项分析直接使用已保存收益，未额外调用模型或重跑策略。</p>')
    return [structural,dynamics,risk]
