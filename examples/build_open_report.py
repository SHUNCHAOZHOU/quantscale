"""Build the detailed Chinese Canvas report from audited post-hoc analysis."""
import argparse
import json
from pathlib import Path


STRATEGIES = {
    'single-n1-s7': {
        'name': '月度信号：双周期动量、近月反转、低波动',
        'formula': '0.55×Rank(6个月动量) + 0.45×Rank(12个月动量) − 0.20×Rank(近21日收益) − 0.25×Rank(126日波动率)',
        'mechanics': '用上个月末数据，动量跳过最近约21个交易日；筛选流动性，每边最多50只，等权，目标总敞口约0.98。',
        'audit': '短历史时12个月项退回6个月项；信号输入自行做前向填充。月度更新信号不等于整月不交易，回测每天仍按目标权重再平衡。测试亏损在取消交易费的静态情景下仍存在。',
    },
    'single-n1-s17': {
        'name': '月度信号：动量、近月反转、低波动、流动性',
        'formula': '0.40×Rank(动量) + 0.25×Rank(−近21日收益) + 0.20×Rank(−60日波动率) + 0.15×Rank(log美元成交额)',
        'mechanics': '以上个月末为信号时点；每边约12%的有效股票，最多60只，等权，多空各约0.5。',
        'audit': '这里的“12/6/3个月动量”是缺失时逐级回退，不是同时加权三个周期。原始假设的表述比实际代码更宽泛，应以实现为准。',
    },
    'independent-n4-s7': {
        'name': '月度信号：动量、5日反转、低波动、均线趋势',
        'formula': '0.40×Rank(12个月动量) + 0.30×Rank(−5日收益) + 0.20×Rank(−60日波动率) + 0.10×Rank(价格/100日均价−1)',
        'mechanics': '取上个月末信号，动量跳过最近约21日，多空各30只等权。',
        'audit': '流动性项使用复权收盘价乘原始成交量，是美元成交额的近似。测试年化收益只有0.59%，成本从5bp升到10bp时静态年化收益转负。',
    },
    'independent-n4-s17': {
        'name': '周度信号：波动率标准化的5日反转',
        'formula': 'Score = (5日收益 − 横截面平均5日收益) / (21日波动率×√5)，做多低分、做空高分',
        'mechanics': '信号截至最近已知的周五；排除流动性最低20%和波动率最高10%，多空各最多50只。',
        'audit': '减去横截面均值不是回归残差化，不能据此声称已经消除市场Beta。验证Sharpe为正，但测试转负；测试取消交易费后仍保留借券费，年化收益仅约0.87%。',
    },
    'decentralized-n4-s7': {
        'name': '每日信号：双周期动量为主，少量反转和低波动',
        'formula': '0.55×Rank(12个月动量) + 0.30×Rank(6个月动量) − 0.10×Rank(近21日收益) − 0.05×Rank(126日波动率)',
        'mechanics': '使用复权收盘价，动量跳过最近21日；每日重算，多空各50只等权，目标总敞口1.0。',
        'audit': '胜出候选来自第0轮、worker 1，生成时既没有自己的有效回测，也没有同伴有效结果。它是独立首提，不能作为同伴共享产生收益提升的证据。',
    },
    'decentralized-n4-s17': {
        'name': '每日信号：平滑的开盘价动量与低波动',
        'formula': '0.65×Z(12个月动量) + 0.15×Z(6个月动量) + 0.20×Z(−60日波动率)，Z截断至±3',
        'mechanics': '优先使用复权开盘价；动量跳过21日并取最近5个信号的平均，多空各50只，目标总敞口0.98。',
        'audit': '胜出候选来自第1轮、worker 3。它看到了自己前一版验证Sharpe约−0.75的反转策略，但收到的同伴有效结果仍为空；代码去掉了快速反转，属于自我修订的直接证据。',
    },
    'hybrid-n4-s7': {
        'name': '月度信号：波动率缩放动量、5日反转、趋势确认',
        'formula': '0.65×Rank(126日动量/60日波动率) + 0.25×Rank(−5日收益) + 0.10×Rank(价格/100日均价−1)',
        'mechanics': '动量跳过5日，使用上个月末信号；多空各最多50只等权，总敞口约0.998。',
        'audit': '胜出候选在第0轮产生，收到管理者方向，但没有同伴有效结果。0.02的微正Sharpe对应−0.31%的复合年化收益，因Sharpe使用算术日均收益，复利还受波动损耗影响，两者并不矛盾。',
    },
}


def build(root):
    root = Path(root)
    payload = json.loads((root / 'detailed-analysis.json').read_text())
    payload.pop('histories')
    payload['strategies'] = STRATEGIES
    return '''import {Stack,Row,H1,H2,Text,Callout,Table,BarChart,LineChart,Pill,useState} from "cursor/canvas";
const D:any = ''' + json.dumps(payload, ensure_ascii=False, allow_nan=False) + ''';
const N:Record<string,string>={single:"单 Agent",independent:"独立并行",centralized:"中心化",decentralized:"去中心化",hybrid:"混合"};
const F=(x:any,n=2)=>typeof x==="number"?x.toFixed(n):"—";
const P=(x:any)=>typeof x==="number"?(100*x).toFixed(2)+"%":"—";
const L=(run:string)=>D.runs.find((r:any)=>r.run===run)?.label||run;
const V:any[]=D.runs.filter((r:any)=>r.status==="evaluated");
const ROOT=D.root;
function Source({children}:any){return <Text size="small" tone="secondary">{children}</Text>}
function Protocol(){return <Stack gap={12}>
 <H2>研究问题与实际测量对象</H2>
 <Text>问题是：在相同的模型与资源上限下，信息共享结构是否改变自主策略搜索的有效产出和最终留出表现？本轮测量的是一个有约束的开放代码搜索系统：研究者自行提出规则和实现，评估器负责运行和反馈。它不是拥有任意数据与工具、可以训练任意模型的完全开放研究环境。</Text>
 <Table headers={["项目","本轮设置"]} rows={[
 ["行情与股票池","Yahoo真实行情；当前标普成分股中500只在目标年份有历史，研究期495只、训练期491只有观测"],
 ["时间划分","训练2016–2021；验证2022–2023；测试2024–2025。前8年用于搜索，最后2年留出"],
 ["候选选择","每组只按验证期扣费后Sharpe选一个策略，平分时按策略SHA升序；没有把多个worker策略组合成一个投资组合"],
 ["Agent结构","单Agent=1名研究者；独立/去中心化=4名研究者；中心化/混合=4名研究者+1名管理者"],
 ["信息流","独立：只看自身最近3条结果；去中心化：再看其他worker上一轮最近有效结果；中心化：管理者分工；混合：管理者+同伴共享"],
 ["调度","每个系统内worker按顺序调用；4路并发是不同实验系统同时运行。名称中的并行表示多研究者结构，不能据此推断4倍运行加速"],
 ["预算","每组最多170,000总token、12次候选尝试；管理者、通信和失败调用都扣同一预算；上限相同，实际消耗不相同"],
 ["模型与输出","deepseek-flash，high推理；单次最大输出16,384 token（含推理），总调用上限随剩余预算收缩"],
 ["代码自由度","没有固定策略家族、因子清单、窗口区间或参数网格；可自行编写特征、组合和仓位规则；仍禁止训练预测模型、外部数据和文件访问"],
 ["仓位与成本","总敞口≤1；单股绝对权重≤2%；净敞口绝对值≤1%；交易费=权重绝对变动和×5bp；空头借券年化30bp"],
 ["决策与成交","收盘t决策，下一个有效开盘成交，再下一个开盘结算收益；各划分从空仓开始，可使用此前历史预热"],
 ["执行环境","每次策略调用限时10秒，每段回测限时180秒；隔离进程；系统未实施严格RAM上限"]]}/>
 <Source>来源：pilot-protocol.json、冻结版 prompts.py/research.py/backtest.py、请求轨迹；重复标签7、17不控制供应商的服务端随机数。</Source>
 <H2>数据处理与可复核性</H2>
 <Text>按既定质量规则跳过2021-05-05、2025-11-12两个不完整横截面会话。上游执行价格不做填补，收益区间跨过跳过的会话，年化仍按252个有效观测。部分生成策略自行对信号输入前向填充，或使用pct_change的默认缺失处理；这属于具体策略实现，不应误写为所有计算都完全没有填补。</Text>
 <Table headers={["核查项","结果"]} rows={[
 ["冻结后才开启测试","全部10次研究尝试结束；7份有效提交统一登记；全部测试完成时间晚于冻结时间"],
 ["策略与实现一致性","7份冻结内容SHA、测试记录引用SHA和实现版本一致"],
 ["评估次数","SQLite登记7次complete，无重复测试和测试失败；每个策略500个开盘到开盘区间"],
 ["收益口径复算","从逐日文件重新复算全部收益、Sharpe、波动、回撤和成本；与evaluation.json一致"],
 ["预算","所有组token≤170,000，候选尝试≤12；3组失败没有追加替代实验"],
 ["本次分析的新增操作","只读取已保存轨迹和逐日收益，未新增模型调用、未重新运行策略或用测试结果调整代码"]]}/>
 <H2>结论边界：当前不能回答什么</H2>
 <Text>① 不能把去中心化的收益优势归因于协作。两个胜出候选都没有收到同伴有效结果。② 不能证明中心化在金融研究上差：两组是输出协议失败，根本没有可比较的最终收益。③ 不能判断稳定的拓扑排序：每种结构只计划两次，且只有7/10组完成。④ 不能宣称发现了新的独立alpha：6/7个最终策略含中期动量，缺少相同约束下的传统因子基线和因子暴露检验。</Text>
 <Text>股票池使用当前成分，存在幸存者和未来成分信息偏差；缺少历史退市和逐日成分重建。历史2024–2025已经在上一轮受限搜索实验中使用，且可能存在于模型训练资料中；冻结保证本轮流程顺序，不能把它变成新的盲测。多个候选共享验证期，反复选择也可能造成验证过拟合。两年测试不覆盖充分的市场状态。</Text>
 <Text>未模拟市场冲击、订单成交约束、逐股真实借券可得性和随时间变动的借券费，也未计保证金融资和抵押现金利息；净美元敞口接近零不保证Beta或行业中性。成本压力分析固定原始收益与换手，不重新模拟净值漂移。区块重采样区间只衡量固定策略在这段历史上的条件不确定性，没有覆盖LLM运行间差异、选择偏差或数据偏差。</Text>
 <H2>下一轮验证方案：先修复测量，再检验拓扑</H2>
 <Table headers={["顺序","具体工作","要排除的混淆"]} rows={[
 ["1 · 协议与生成预实验","用独立于正式收益比较的合成数据测试JSON包装、worker标识与嵌套分工解析；约束格式修复范围，明确哪些歧义必须拒绝；所有结构采用相同的失败/修复计费规则。另测输出上限与推理预算对完整代码交付率的影响，再冻结设置","让格式可靠性和正文生成不再压过搜索能力；本轮不补跑、不改历史结果"],
 ["2 · 分开两类预算实验","固定总token衡量端到端研究效率；另一套固定有效候选数、记录全部额外token，衡量给定搜索覆盖度下的策略质量","同样的预算上限不等于同样有效搜索量；不能同时假定两者完全匹配"],
 ["3 · 通信消融和足够重复","预先固定调用顺序与预算，比较同伴内容开启/关闭；保存胜出候选的实际信息来源；增加重复次数并在实验前做功效评估","候选本身好与通信使它变好是不同问题；重复数不能靠这两次结果倒推"],
 ["4 · 传统基线与行为多样性","同样的股票池、执行、成本和仓位约束下加入简单动量/低波动基线；未来记录逐日持仓，比较因子暴露、持仓重叠和收益相关","区分常见因子再组合与新的alpha；代码SHA不同不等于经济逻辑不同"],
 ["5 · 数据与前瞻检验","重建历史成分与退市处理；规则冻结后开展未参与调试的时间窗及前瞻纸面测试；预先固定交易成本与借券压力条件","当前成分偏差、反复使用历史测试期以及交易实现假设"]]}/>
 <Source>以上为根据本轮暴露的问题制定的验证方案，不是已完成的新实验，也不是投资部署建议。</Source>
 </Stack>}
export default function QuantScaleDetailedReport(){
 const [tab,setTab]=useState("结论与收益");
 const [topology,setTopology]=useState("decentralized");
 const selected=V.filter(r=>r.topology===topology);
 const a:any[]=D.aggregates.filter((r:any)=>r.tested>0);
 const cb=(name:string)=>D.calendar.find((r:any)=>r.run===name&&r.year==="2024");
 const cy=(name:string)=>D.calendar.find((r:any)=>r.run===name&&r.year==="2025");
 return <Stack gap={20} style={{padding:24,maxWidth:1220,margin:"0 auto"}}>
 <H1>QuantScale 开放策略搜索实验 · 详细分析报告</H1>
 <Text>DeepSeek Flash · 500只当前标普成分股 · 2016–2023研究 / 2024–2025测试 · 正式重跑 sp500-deepseek-open-pilot-v2</Text>
 <Callout tone="info" title="去中心化组这次收益最高；轨迹并不支持把优势归因于同伴协作">
 两个去中心化胜出候选生成时，同伴有效结果都是0条。其平均测试Sharpe为0.69，但2025年表现显著弱于2024年，且两份策略日净收益相关系数为0.952。
 </Callout>
 <Row gap={8} wrap>{["结论与收益","策略与相似性","生成与成本","方法与后续验证"].map(t=><Pill key={t} active={t===tab} onClick={()=>setTab(t)}>{t}</Pill>)}</Row>
 {tab==="结论与收益"&&<Stack gap={16}>
 <H2>1. 本轮真正得到的结论</H2>
 <Text><strong>观察到的收益优势：去中心化两组都约0.69 Sharpe，但不是对协作收益的因果证明。</strong>其中一份策略在首轮独立生成；另一份策略仅根据自己的失败回测修订。组别标签没有说明获胜候选究竟使用了什么信息，必须结合轨迹阅读。</Text>
 <Text><strong>系统的主要损耗发生在代码生成。</strong>71次候选尝试中54次输出截断，仅17份完整代码进入代码阶段，其中16份完成训练与验证。输出token的96.3%用于推理；44次截断的用量记录显示全部输出token都是推理。有效策略的覆盖量不足以支撑稳定的结构排序。</Text>
 <Text><strong>开放代码带来了更复杂的实现，但没有呈现充分的经济逻辑多样性。</strong>17份代码的规范化AST散列均不同；最终7份中6份包含中期动量。两份去中心化收益高度相关，单Agent两份也相关0.868，说明语法多样性和收益多样性应分开衡量。</Text>
 <H2>2. 按结构汇总：均值只包括完成测试的组</H2>
 <Table headers={["结构","测试完成/计划","有效候选/尝试","验证Sharpe均值","测试Sharpe均值","测试年化收益均值","测试最大回撤均值"]} rows={D.aggregates.map((r:any)=>[r.label,`${r.tested}/${r.planned}`,`${r.valid}/${r.attempted}`,F(r.validation_sharpe_mean),F(r.test_sharpe_mean),P(r.test_annualized_return_mean),P(r.test_max_drawdown_mean)])}/>
 <Source>来源：comparison.csv、trajectory.jsonl。每次运行最多12次候选、170,000 token；失败不是零收益。Sharpe与年化收益均值是逐运行指标的算术均值，不是投资组合表现。</Source>
 <Text size="small" tone="secondary">横轴：结构（完成数/计划数） · 纵轴：扣费后年化Sharpe，无量纲</Text>
 <BarChart categories={a.map(r=>`${r.label} (${r.tested}/2)`)} series={[{name:"验证2022–2023",data:a.map(r=>r.validation_sharpe_mean),tone:"neutral"},{name:"测试2024–2025",data:a.map(r=>r.test_sharpe_mean),tone:"info"}]} height={280} showValues referenceLines={[{value:0,label:"零Sharpe",tone:"neutral"}]}/>
 <Source>来源：冻结的selection.json与单次evaluation.json；中央式没有有效提交，未画收益柱。上述柱形不表示统计显著的排名。</Source>
 <H2>3. 七份最终策略：训练、验证到测试</H2>
 <Table headers={["运行","训练Sharpe","验证Sharpe","测试Sharpe","两年累计收益","年化收益","年化波动","最大回撤"]} rows={V.map(r=>[r.label,F(r.train_sharpe),F(r.validation_sharpe),F(r.test.sharpe),P(r.test.total_return),P(r.test.annualized_return),P(r.test.volatility),P(r.test.max_drawdown)])}/>
 <Text>16个有效候选里只有1个验证Sharpe为正，即独立并行·17的周度反转；其测试Sharpe却降至−0.42。去中心化两份的验证Sharpe都略负，而测试转正。这种变化说明验证排序在本轮不稳定，不应只因最后测试期赚钱就把候选视为已经确认有效。</Text>
 <Text>7份测试里4份Sharpe略大于0，但仅3份复合收益为正：混合·7的日均算术收益略正，波动损耗后复合年化为−0.31%。这不是指标计算矛盾，也不应把它计入真正盈利的策略数。</Text>
 <Source>来源：selection.json、evaluation.json；日均收益÷样本日波动×√252得到Sharpe；复合年化=(逐日净值乘积)^(252/500)−1。</Source>
 <H2>4. 盈利是否跨年份持续？</H2>
 <Table headers={["策略","2024实际累计收益","2024 Sharpe","2025实际累计收益","2025 Sharpe"]} rows={V.map(r=>[r.label,P(cb(r.run).total_return),F(cb(r.run).sharpe),P(cy(r.run).total_return),F(cy(r.run).sharpe)])}/>
 <Text>去中心化·7在2024年赚15.14%、2025年赚1.31%；去中心化·17在2024年赚20.21%、2025年亏4.21%。其优势主要来自2024年，不能称为逐年稳定有效。其他多数候选同样在2025年明显恶化，提示共同因子或市场状态敏感性；仅凭收益序列还不能确定具体因果来源。</Text>
 <Row gap={6} wrap>{a.map(r=><Pill key={r.topology} active={topology===r.topology} onClick={()=>setTopology(r.topology)}>{r.label}</Pill>)}</Row>
 <Text size="small" tone="secondary">横轴：月末日期 · 纵轴：复合净值，初始1；当前选择：{N[topology]}</Text>
 <LineChart categories={selected[0].monthly_nav.dates} series={selected.map((r,i)=>({name:r.label,data:r.monthly_nav.values,tone:(i===0?"info":"neutral") as "info"|"neutral"}))} height={290} beginAtZero={false} referenceLines={[{value:1,label:"初始净值",tone:"neutral"}]}/>
 <Source>来源：hidden_daily.csv；连续复合后按月末抽样，月间回撤不可从这张抽样图精确读取；下表使用全部逐日观测。</Source>
 <Table headers={["当前策略","最大回撤峰值日","谷值日","测试结束是否收复峰值","期末距历史峰值"]} rows={selected.map(r=>[r.label,r.drawdown_episode.peak,r.drawdown_episode.trough,r.drawdown_episode.recovery||"尚未收复",P(r.drawdown_episode.end_below_peak)])}/>
 <H2>5. 零美元净敞口不等于零市场风险</H2>
 <Table headers={["策略","相对SPY Beta","与SPY日收益相关","单市场回归年化算术截距"]} rows={V.map(r=>[r.label,F(r.spy.beta,3),F(r.spy.correlation,3),P(r.spy.arithmetic_intercept_annualized)])}/>
 <Text>两份去中心化策略的SPY Beta约0.24、0.17，存在正向市场暴露；单市场回归截距约为年化3.28%、4.01%。截距只是这段历史上的描述性拟合，没有做显著性或多因子风险调整，不能直接称为已确认alpha。同区间SPY年化约22.31%，但其风险和敞口与这些多空策略不同，不能据收益高低直接判定优劣。</Text>
 <Source>来源：已下载spy.csv与策略逐日记录；逐行使用相同execution_date→date的复权开盘收益，跨过同一个缺失会话；OLS含截距，现金基准为零。年化截距=252×(策略日均收益−Beta×SPY日均收益)，不是复合超额收益。</Source>
 <H2>6. 统计不确定性：本轮不足以锁定排名</H2>
 <Table headers={["固定策略","测试Sharpe","条件95%区块重采样区间"]} rows={V.map(r=>[r.label,F(r.test.sharpe),`[${F(r.sharpe_block_interval[0])}, ${F(r.sharpe_block_interval[1])}]`])}/>
 <Table headers={["固定策略组均值之差","观测差值","条件95%区块重采样区间"]} rows={D.bootstrap.contrasts.map((r:any)=>[`去中心化 − ${N[r.other]}`,F(r.estimate),`[${F(r.low)}, ${F(r.high)}]`])}/>
 <Text>每个单独策略的Sharpe区间都覆盖0。去中心化相对独立并行和混合的差值区间也覆盖0；相对单Agent的固定策略差值区间不覆盖0，但这仍不能检验“拓扑本身更好”，因为候选已被选择、信息来源不同、LLM重复仅两次且3组缺失。</Text>
 <Source>方法：对500天日净收益做同步循环移动区块重采样，区块20个观测、5,000次、固定随机种子20261026，所有策略使用同一抽样索引保留相关性。取2.5%/97.5%分位；这是选定策略路径的条件区间，不包含模型搜索/候选选择/数据偏差，不是拓扑因果置信区间，且为事后分析。</Source>
 </Stack>}
 {tab==="策略与相似性"&&<Stack gap={16}>
 <H2>7. 胜出策略是否真的使用了协作信息？</H2>
 <Table headers={["最终策略","候选序号(从1起)","worker/轮次(从0起)","自身有效结果条数","同伴有效结果条数","收到管理者方向"]} rows={V.map(r=>[r.label,r.winner_attempt,`${r.winner_worker} / ${r.winner_round}`,r.winner_own_valid_results,r.winner_peer_results,r.winner_received_manager_direction?"是":"否"])}/>
 <Callout tone="warning" title="两个去中心化胜出候选均没有读取同伴有效结果">
 去中心化·7来自第0轮；去中心化·17来自第1轮，当时只有它自己的上一版形成有效回测。后一份原始假设明确提出去除高换手短期反转，但可见证据来自自身亏损反馈。轨迹支持“独立提案或自我改进找到了相对较好的策略”，不支持“同伴通信使这些最终策略变好”。
 </Callout>
 <Text>这也不证明通信没有用：分享可能影响后续未获胜的候选，只是本轮没有提供它提升最终收益的直接证据。检验通信效应需要预先设计开启/关闭同伴内容的消融，并增加独立重复。</Text>
 <Source>来源：trajectory.jsonl中与selection.json匹配的role=researcher请求，逐条统计own_results、peer_results和direction；不是从拓扑名称推断。</Source>
 <H2>8. 七份最终策略的实际代码行为</H2>
 <Text>下文是对保存代码的人工核读。Rank指当期有效股票的横截面百分位排序；Z指横截面标准化。“月度/周度”表示特征数据截止到上个月末或最近周五，不是严格的月度/周度交易：股票池、排名与目标权重仍可能逐日变化，回测每天恢复目标权重并按变动收费。</Text>
 {V.map(r=><Stack gap={6} key={r.run}><Text><strong>{r.label}：{D.strategies[r.run].name}</strong></Text>
 <Text>{D.strategies[r.run].formula}</Text><Text>{D.strategies[r.run].mechanics}</Text>
 <Text>{D.strategies[r.run].audit}</Text><Source>策略SHA：{r.strategy_id} · <a href={r.source_file}>打开完整Python代码</a></Source></Stack>)}
 <H2>9. 代码去重与收益去重给出不同答案</H2>
 <Text>本轮17份进入代码阶段的模块，规范化AST散列17种，精确重复为0；但这只排除了相同语法结构，不能排除不同写法实现相同经济假设。7份最终策略中6份有中期动量成分，唯一以周度短期反转为核心的是独立并行·17。模型自由编写了组合、过滤器和信号频率，却仍主要围绕价格、成交量与常见风格构建策略。</Text>
 <Table headers={["策略一","策略二","测试日净收益相关系数","共同观测数"]} rows={D.correlations.slice(0,7).map((r:any)=>[L(r.left),L(r.right),F(r.correlation,3),r.observations])}/>
 <Text>去中心化两份相关0.952，是本轮最相似的一对；单Agent两份相关0.868。相关性说明已实现净收益同涨同跌，不等于持仓完全相同或代码直接复制。未保存逐日逐股权重，报告不虚构持仓重叠率，也不把共有动量成分当作已验证的因子归因。</Text>
 <Text>这轮只支持一个有限判断：去掉固定策略菜单后，程序实现的多样性增加，但最终收益方向仍有明显聚集。不能把聚集完全归因于模型能力，因为输入只有价量数据、组合约束相同、模型和推理设置相同，有效搜索又只有16个候选。</Text>
 <Source>来源：CodeSpec规范化AST散列、保存的Python模块、hidden_daily.csv；相关性为按日期对齐的Pearson系数，未重新回测。与上一轮参数散列的重复率不是同一度量，不能直接比较百分比。</Source>
 </Stack>}
 {tab==="生成与成本"&&<Stack gap={16}>
 <H2>10. 搜索漏斗：真正回测的策略覆盖量很小</H2>
 <Table headers={["环节","数量","解释"]} rows={[
 ["模型调用","77","71次研究者调用 + 6次管理者调用；总计1,215,706 token"],
 ["候选尝试","71 / 最多120","12次是每组上限，token先耗尽或管理者终止会提前结束"],
 ["输出截断","54 / 71 = 76.1%","未生成完整策略；其中44次输出用量全部计为推理"],
 ["完整代码","17 / 71 = 23.9%","17种规范化AST；没有精确重复"],
 ["执行失败","1 / 17","策略使用环境不可用的hasattr；不是收益表现差"],
 ["训练和验证均完成","16 / 71 = 22.5%","16个有效候选中仅1个验证Sharpe为正"],
 ["最终提交并测试","7 / 10组","中心化2组、混合1组因管理者输出协议失败，无有效最终收益"]]}/>
 <Text>输出token共1,067,144，其中推理1,027,472，占96.28%；剩余非推理输出约39,672，其中还包括JSON包装和管理者分工。这个配置下，模型大部分资源没有转化为可执行代码。它反映当前模型、提示、输出限制和解析方式的共同作用；不能由此推断所有DeepSeek配置或所有多Agent系统都会如此。</Text>
 <Text size="small" tone="secondary">横轴：结构 · 纵轴：有效候选数/100,000实际token（失败组费用也计入）</Text>
 <BarChart categories={D.aggregates.map((r:any)=>r.label)} series={[{name:"有效候选 / 10万token",data:D.aggregates.map((r:any)=>r.valid_per_100k_tokens),tone:"info"}]} height={250} showValues/>
 <Source>来源：全部轨迹的response.tokens与trial.validation；单系列各项为同一度量，非单次调用成功率。</Source>
 <Table headers={["结构","尝试/有效","实际token","每10万token有效候选","管理者token","通信相关调用token上界"]} rows={D.aggregates.map((r:any)=>[r.label,`${r.attempted}/${r.valid}`,r.tokens.toLocaleString(),F(r.valid_per_100k_tokens),r.manager_tokens.toLocaleString(),r.communication_tokens_upper_bound.toLocaleString()])}/>
 <Text>去中心化与独立并行都只产出5个有效候选，效率约1.56与1.53个/10万token，差距很小；收益差异不能简单解释为前者测试了更多想法。通信相关token是包含同伴上下文的完整调用总量，含模型自己的推理和代码，属于上界；它不是纯通信文本的增量费用。</Text>
 <H2>11. 管理者失败如何影响公平性？</H2>
 <Table headers={["运行","具体失效点","是否有可比较策略收益"]} rows={[
 ["中心化 · 17","首次管理者响应JSON解析失败，adapter_error记录缺少逗号；当前代码随后取directions触发终止","无"],
 ["中心化 · 7","首轮4个研究者候选均输出截断；下轮管理者返回直接的worker键分工，漏掉directions外层包装","无"],
 ["混合 · 17","管理者分工采用嵌套对象并含附加内容，不符合每个worker对应非空字符串的当前契约","无"]]}/>
 <Text>管理者错误会终止整组，而研究者错误只扣一次候选并继续，存在明显的协议失败惩罚不对称。这使管理型结构额外暴露于解析器可靠性风险。将它们记为系统完成失败是合理的端到端测量，但不能把无提交解释为金融策略收益为零或为负，更不能据此评价中心化的搜索质量。</Text>
 <Text>另一个调度因素是每轮固定从worker 0开始，预算提前耗尽时后面的worker尝试更少。例如独立并行两组都是3、3、2、2次，而非每人都完成3次。固定总token实验允许这种资源竞争，但应披露，未来可用轮换顺序检验它的影响。</Text>
 <Source>来源：原始manager响应、aborted事件及worker尝试计数。此前因worker标识契约问题中止的准备运行另有72,341个已确认token和3个未取得用量回执的在途请求；正式重跑预算统一降为170,000，原账本另存，不与本轮策略成绩合并。</Source>
 <H2>12. 相同Sharpe不意味着相同的成本韧性</H2>
 <Table headers={["最终策略","日均交易量/NAV","两年交易费累计/NAV","两年借券费累计/NAV","0bp静态年化收益","5bp原始年化收益","10bp静态年化收益","20bp静态年化收益"]} rows={V.map(r=>[r.label,P(r.test.average_daily_turnover),P(r.test.total_cost),P(r.test.total_borrow_cost),P(r.cost_sensitivity[0].annualized_return),P(r.test.annualized_return),P(r.cost_sensitivity[3].annualized_return),P(r.cost_sensitivity[4].annualized_return)])}/>
 <Text>去中心化·7日均交易名义量21.08%，去中心化·17仅8.60%；两年交易费支出分别相当于NAV的5.27%和2.15%。两者基础成本下Sharpe都约0.69，但成本提高到20bp时，前者静态年化约−0.21%，后者仍约3.93%。因此，后者的成本容忍度更好；这不等于它在其他市场状态下一定更好，它在2025年仍亏损。</Text>
 <Text>独立并行·17日均交易量37.02%，两年交易费累计9.25%，取消交易费、仍扣借券费的静态年化收益约0.87%；5bp时变为−3.73%，成本足以吞掉这份弱边际。两份单Agent在0bp情景下仍亏损，说明它们的问题不只是费用。</Text>
 <Text>上表累计费用是逐日成本占当日NAV比例的求和，不是“总收益直接减去几个百分点”的归因；不能把5.27%当作初始本金的精确现金损失。交易量按权重绝对变化之和定义，一笔等额换仓的买入与卖出都计入，勿与只计一侧的换手口径混用。</Text>
 <Source>静态成本公式：新日净收益=保存的gross_return−保存的borrow_cost−保存的turnover×新bp/10,000。固定仓位路径及5bp运行形成的换手，不重新模拟费用改变后的NAV漂移，因此是事后近似压力分析，不是新的策略回测或部署成本估计。</Source>
 </Stack>}
 {tab==="方法与后续验证"&&<Protocol/>}
 <Source>分析数据：<a href={ROOT+"/detailed-analysis.json"}>detailed-analysis.json</a> · 原始汇总：<a href={ROOT+"/comparison.csv"}>comparison.csv</a> · 冻结凭据：<a href={ROOT+"/all-strategies-frozen.json"}>all-strategies-frozen.json</a> · 正式测试完成时间：{D.completed.completed_at_utc}</Source>
 </Stack>;
}
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    Path(args.output).write_text(build(args.root))
