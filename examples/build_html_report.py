"""Create a self-contained, offline Chinese HTML report from audited results."""
import argparse
from html import escape
import json
import re
from pathlib import Path

from .build_open_report import STRATEGIES
from .report_topologies import setup_html, topology_html


def fmt(value, places=2):
    return f'{value:.{places}f}' if isinstance(value, (int, float)) else '—'


def pct(value):
    return f'{100 * value:.2f}%' if isinstance(value, (int, float)) else '—'


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(
        '<th scope="col">' + escape(str(x)) + '</th>' for x in headers
    ) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(
        '<td>' + escape(str(x)) + '</td>' for x in row
    ) + '</tr>' for row in rows) + '</tbody></table></div>'


def section(identifier, number, title, body):
    return f'<section id="{identifier}"><div class="section-no">{number}</div><h2>{title}</h2>{body}</section>'


def build(root):
    root = Path(root)
    data = json.loads((root / 'detailed-analysis.json').read_text())
    valid = [r for r in data['runs'] if r['status'] == 'evaluated']
    names = {r['topology']: r['label'] for r in data['aggregates']}
    labels = {r['run']: r['label'] for r in valid}
    years = {(r['run'], r['year']): r for r in data['calendar']}
    parts = []
    parts.append(section('summary', '01', '先看结论', '''
    <div class="finding"><span>收益</span><p>去中心化两次运行的测试 Sharpe 都约为 <strong>0.69</strong>，平均年化收益 <strong>7.72%</strong>，在完成测试的组里最高。但收益主要来自 2024 年，2025 年一份小赚、一份亏损。</p></div>
    <div class="finding"><span>归因</span><p>这两份胜出候选生成时，收到的<strong>同伴有效结果都是零条</strong>。轨迹支持独立提案和自我修订产生了较好候选，尚不支持“同伴协作带来收益提升”。</p></div>
    <div class="finding"><span>瓶颈</span><p>71 次候选尝试，只有 <strong>16 次</strong>完成训练与验证。54 次输出截断；输出 token 的 <strong>96.3%</strong>用于推理。当前比较受到生成完成率和解析器可靠性的显著影响。</p></div>
    <p class="takeaway">本轮值得保留的是两份可继续验证的动量组合策略，以及一套可审计的实验记录。现有证据还不足以确认稳定的拓扑排名或可部署的独立 alpha。</p>
    <p class="source">阅读提示：先看实验设定和五种拓扑图，再读收益与协作归因；策略代码、统计区间和完整约束可按左侧目录查看。</p>'''))
    parts.append(section('experiment', '02', '实验具体怎么设定？', setup_html(table)))
    parts.append(section('topologies', '03', '五种拓扑：研究者之间如何连接？', topology_html()))
    aggregate_rows = [[r['label'], f"{r['tested']}/{r['planned']}", f"{r['valid']}/{r['attempted']}",
                       fmt(r.get('validation_sharpe_mean')), fmt(r.get('test_sharpe_mean')),
                       pct(r.get('test_annualized_return_mean')), pct(r.get('test_max_drawdown_mean'))]
                      for r in data['aggregates']]
    parts.append(section('results', '02', '不同结构，实际得到了什么？',
        '<p>使用 500 只当前标普成分股的真实 Yahoo 行情。2016–2021 年训练，2022–2023 年验证选优，2024–2025 年测试。以下收益均扣除交易费和借券费。</p>'
        + table(['结构', '完成/计划', '有效候选/尝试', '验证 Sharpe', '测试 Sharpe', '年化收益', '最大回撤'], aggregate_rows)
        + '<p class="source">逐次指标的算术均值，不是组合投资表现。失败组保留在完成率分母中，缺失收益不填零。混合仅 1 组完成，中心化没有可比较的最终收益。</p>'
        + '<div class="chart-box"><h3>验证与测试 Sharpe 均值</h3><p class="chart-label">横轴：结构及测试完成数 / 计划数 · 纵轴：年化 Sharpe，无量纲</p><div id="sharpe-chart"></div><div class="legend"><span><i class="gray"></i>验证 2022–2023</span><span><i></i>测试 2024–2025</span></div><p class="source">来源：comparison.csv、topology-summary.csv；中央式缺失，不画零收益柱。</p></div>'
        + '<details><summary>展开七份最终策略的训练、验证与测试明细</summary>'
        + table(['运行', '训练 Sharpe', '验证 Sharpe', '测试 Sharpe', '两年累计收益', '年化收益', '最大回撤'],
                [[r['label'], fmt(r['train_sharpe']), fmt(r['validation_sharpe']), fmt(r['test']['sharpe']), pct(r['test']['total_return']), pct(r['test']['annualized_return']), pct(r['test']['max_drawdown'])] for r in valid])
        + '<p>16 个有效候选中仅 1 个验证 Sharpe 为正，即独立并行·17的周度反转；其测试 Sharpe 转为 −0.42。7 份测试中 4 份 Sharpe 略正，但只有 3 份复合收益为正。混合组的微正算术日均收益经过波动损耗后，复合年化仍为 −0.31%，两种指标并不矛盾。</p></details>'))
    parts.append(section('time', '03', '收益集中在哪一年？回撤有没有收复？',
        table(['策略', '2024 实际收益', '2024 Sharpe', '2025 实际收益', '2025 Sharpe'],
              [[r['label'], pct(years[r['run'], '2024']['total_return']), fmt(years[r['run'], '2024']['sharpe']), pct(years[r['run'], '2025']['total_return']), fmt(years[r['run'], '2025']['sharpe'])] for r in valid])
        + '<div class="note"><strong>去中心化的优势主要来自 2024 年。</strong><p>两份策略 2024 年分别赚 15.14%、20.21%；2025 年分别为 +1.31%、−4.21%。其他多数候选也在 2025 年恶化，提示共有因子或市场状态敏感性，但仅凭收益序列不能确定具体原因。</p></div>'
        + '<div class="chart-box"><div class="chart-heading"><h3>测试期净值路径</h3><label>选择结构 <select id="nav-select"><option value="decentralized">去中心化</option><option value="single">单 Agent</option><option value="independent">独立并行</option><option value="hybrid">混合</option></select></label></div><p class="chart-label">横轴：月末日期 · 纵轴：复合净值，初始 1</p><div id="nav-chart"></div><div id="nav-legend" class="legend"></div><p class="source">来源：hidden_daily.csv；逐日复合后月末抽样。鼠标停在节点可查看数值；月末抽样不会显示所有月内回撤。</p></div>'
        + '<details><summary>展开完整日序列计算的最大回撤区间</summary>'
        + table(['策略', '回撤峰值日', '谷值日', '测试结束是否收复', '期末距历史峰值'],
                [[r['label'], r['drawdown_episode']['peak'], r['drawdown_episode']['trough'], r['drawdown_episode']['recovery'] or '尚未收复', pct(r['drawdown_episode']['end_below_peak'])] for r in valid]) + '</details>'))
    parts.append(section('collaboration', '04', '收益最高，是否真的是协作带来的？',
        '<p>需要检查胜出候选生成时实际收到了什么，而不能只看它属于哪个结构。</p>'
        + table(['最终策略', '候选序号', 'worker / 轮次', '自身有效结果', '同伴有效结果', '管理者方向'],
                [[r['label'], r['winner_attempt'], f"{r['winner_worker']} / {r['winner_round']}", r['winner_own_valid_results'], r['winner_peer_results'], '有' if r['winner_received_manager_direction'] else '无'] for r in valid])
        + '<p class="source">候选从 1 编号，worker 和轮次从 0 编号。来源：与 selection.json 匹配的原始 researcher 请求，而非从组别名称推断。</p>'
        + '<div class="two-col"><article><h3>去中心化 · 7</h3><p>来自第 0 轮、worker 1。生成前没有自己的有效回测，也没有同伴有效结果，是首轮独立提案。</p></article><article><h3>去中心化 · 17</h3><p>来自第 1 轮、worker 3。看到了自己上一版约 −0.75 的验证 Sharpe，但同伴有效结果仍为空。新版删除了高换手快速反转，转向更慢的动量与低波动。</p></article></div>'
        + '<p class="takeaway">本轮没有提供“同伴通信提升最终收益”的直接证据。这也不能证明通信无用：分享可能影响后续未获胜候选。真正检验通信效果，需要预先设计开启 / 关闭同伴内容的消融和更多独立重复。</p>'))
    strategy_html = ''
    for r in valid:
        s = STRATEGIES[r['run']]
        source = Path(r['source_file']).read_text()
        strategy_html += f'''<details class="strategy"><summary><span>{escape(r['label'])}</span> {escape(s['name'])}<small>测试 Sharpe {fmt(r['test']['sharpe'])} · 年化 {pct(r['test']['annualized_return'])}</small></summary>
        <div class="formula">{escape(s['formula'])}</div><p>{escape(s['mechanics'])}</p><p class="audit"><strong>代码核读：</strong>{escape(s['audit'])}</p>
        <details class="code-details"><summary>查看完整 Python 代码</summary><pre><code>{escape(source)}</code></pre><p class="source">SHA：{escape(r['strategy_id'])}</p></details></details>'''
    parts.append(section('strategies', '05', '七份策略究竟怎么做？',
        '<p>下文以实际代码为准。Rank 是当期有效股票的百分位排序；Z 是横截面标准化。“月度 / 周度”表示特征截止到月末或周五，并不等于只在那天交易：股票池、排名和目标权重仍可能逐日变化，回测每天按目标权重再平衡。</p>' + strategy_html))
    parts.append(section('diversity', '06', '代码多样性与收益多样性',
        '<p><strong>17 份代码有 17 种规范化 AST 散列，精确重复为零。</strong>但散列不同只排除了相同语法结构，不能排除同一经济假设的不同写法。最终 7 份中，6 份含中期动量；唯一以周度短期反转为核心的是独立并行·17。</p>'
        + table(['策略一', '策略二', '测试日净收益相关系数', '共同观测数'], [[labels[r['left']], labels[r['right']], fmt(r['correlation'], 3), r['observations']] for r in data['correlations'][:7]])
        + '<p>去中心化两份相关 <strong>0.952</strong>，单 Agent 两份相关 <strong>0.868</strong>。开放代码增加了组合、过滤器、窗口和频率的实现变化，但最终收益方向仍聚集。相关性不证明仓位完全一样；本轮未保存逐日逐股权重，不能计算真实持仓重叠率。</p>'
        + '<p class="source">来源：保存的 Python 代码、规范化 AST 与按日期对齐的 500 个测试日净收益。没有重新回测。输入只有价量、约束相同、模型相同，且有效候选仅 16 个，不能把同质化完全归因于模型能力。上一轮参数散列与本轮代码散列不是同一度量。</p>'))
    parts.append(section('generation', '07', '搜索资源去了哪里？',
        '<div class="pipeline"><div><b>71</b><span>候选尝试</span></div><span>→</span><div><b>17</b><span>完整代码</span></div><span>→</span><div><b>16</b><span>训练与验证完成</span></div><span>→</span><div><b>7</b><span>最终测试策略</span></div></div>'
        + '<p>77 次模型调用包括 71 次研究者调用和 6 次管理者调用，正式重跑共消耗 <strong>1,215,706 token</strong>。54 次候选因输出截断失败，占尝试的 76.1%；其中 44 次用量记录显示全部输出 token 都用于推理。17 份完整代码中，1 份因调用环境不支持的 hasattr 失败。</p>'
        + '<p>输出 token 共 1,067,144，其中推理 1,027,472，占 <strong>96.28%</strong>；非推理输出约 39,672，还包括 JSON 包装和管理者分工。当前配置下，大部分资源未转化为可执行程序。这个结果反映模型、提示、推理预算、输出上限和解析方式的共同影响。</p>'
        + table(['结构', '尝试 / 有效', '实际 token', '每10万 token 有效候选', '管理者 token'], [[r['label'], f"{r['attempted']} / {r['valid']}", f"{r['tokens']:,}", fmt(r['valid_per_100k_tokens']), f"{r['manager_tokens']:,}"] for r in data['aggregates']])
        + '<p>去中心化与独立并行都只有 5 个有效候选，效率约 1.56 与 1.53 个 / 10万 token，差距很小。去中心化的测试收益优势不能简单解释为它探索了更多有效策略。</p>'
        + '<details><summary>展开管理者失败、计费和调度问题</summary>'
        + table(['运行', '失效点'], [['中心化 · 17', '首次管理者 JSON 解析失败，随后缺少 directions 导致终止。'], ['中心化 · 7', '首轮四个候选均输出截断；下轮分工漏掉 directions 外层包装。'], ['混合 · 17', '分工采用嵌套对象，不符合每个 worker 对应非空字符串的当前契约。']])
        + '<p>管理者错误会终止整组，研究者错误只扣一次尝试并继续，协议惩罚不对称。无提交可计为端到端系统失败，却不能解释为金融策略收益为零或为负。</p><p>每轮固定从 worker 0 开始，预算先耗尽时后面的人尝试更少：独立并行两组都是 3、3、2、2 次。每个系统内的 worker 实际顺序调用；四路并发是不同实验系统同时运行，因此不能从本轮推断四倍研究加速。</p><p>通信相关调用的完整 token 含推理和代码，只能作为通信费用上界，不能当纯增量通信成本。此前因 worker 标识契约问题中止的准备运行，另有 72,341 个已确认 token 和 3 个未取得用量回执的在途请求；正式重跑统一降低每组预算，没有把准备运行的策略结果混入本轮。</p></details>'))
    parts.append(section('costs', '08', '同样的 Sharpe，成本承受能力并不相同',
        table(['策略', '日均交易量/NAV', '两年交易费累计/NAV', '0bp 年化', '5bp 原始年化', '10bp 年化', '20bp 年化'],
              [[r['label'], pct(r['test']['average_daily_turnover']), pct(r['test']['total_cost']), pct(r['cost_sensitivity'][0]['annualized_return']), pct(r['test']['annualized_return']), pct(r['cost_sensitivity'][3]['annualized_return']), pct(r['cost_sensitivity'][4]['annualized_return'])] for r in valid])
        + '<div class="note"><strong>去中心化 · 17 的成本韧性更好。</strong><p>它的日均交易量只有 8.60%，而另一份为 21.08%，相差约 2.45 倍。成本升至 20bp 时，两份静态年化收益分别约 +3.93%、−0.21%。但低换手版本在 2025 年仍亏损，成本更稳健不等于跨市场状态更稳健。</p></div>'
        + '<p>独立并行·17的日均交易量为 37.02%，两年交易费累计相当于 NAV 的 9.25%；取消交易费但保留借券费时，静态年化仅约 0.87%，5bp 足以吞掉这份弱边际。两份单 Agent 在 0bp 情景下仍亏损，问题不只是费用。</p>'
        + '<p class="source">事后近似压力分析：新日净收益 = 保存的 gross_return − borrow_cost − turnover × 新 bp / 10,000。固定原始交易路径及换手，不重算费用影响下的 NAV 漂移，不是新增策略回测。累计费用是每天成本占当日 NAV 比例的求和，不能直接当作初始本金的精确现金损失；等额买卖两边都计入交易量。所有情景仍保留借券费。</p>'))
    parts.append(section('uncertainty', '09', '风险暴露和统计不确定性',
        '<h3>美元中性，不代表市场 Beta 中性</h3>'
        + table(['策略', 'SPY Beta', '与SPY日收益相关', '单市场回归年化算术截距'], [[r['label'], fmt(r['spy']['beta'], 3), fmt(r['spy']['correlation'], 3), pct(r['spy']['arithmetic_intercept_annualized'])] for r in valid])
        + '<p>去中心化两份 Beta 约 0.24、0.17，仍有正向市场暴露。单市场回归截距约为年化 3.28%、4.01%，只是这段历史的描述性拟合，没有多因子调整或显著性确认，不能直接称为独立 alpha。同区间 SPY 年化约 22.31%，但其风险和敞口不同，不能仅比较收益高低。</p><p class="source">使用已下载 SPY 复权开盘价，与每一条策略记录的 execution_date→date 对齐，跨过相同缺失会话。截距 = 252×(策略日均收益−Beta×SPY日均收益)，不是复合超额收益。</p>'
        + '<h3>区间有多宽？</h3>'
        + table(['固定策略', '测试 Sharpe', '条件95%区块重采样区间'], [[r['label'], fmt(r['test']['sharpe']), f"[{fmt(r['sharpe_block_interval'][0])}, {fmt(r['sharpe_block_interval'][1])}] "] for r in valid])
        + '<details><summary>展开固定策略组之间的差值区间与重采样方法</summary>'
        + table(['去中心化减去…', '观测差值', '条件95%区间'], [[names[r['other']], fmt(r['estimate']), f"[{fmt(r['low'])}, {fmt(r['high'])}]"] for r in data['bootstrap']['contrasts']])
        + '<p>对 500 个日净收益同步进行循环移动区块重采样：每块 20 个观测，5,000 次，种子 20261026。所有策略使用同一抽样索引以保留相关性，取 2.5% 和 97.5% 分位。区间只反映选定策略在该历史路径上的条件不确定性，不覆盖 LLM 运行间差异、候选选择偏差或数据偏差，且属于事后分析。</p></details>'
        + '<p class="takeaway">每份策略的区间都覆盖零。去中心化相对独立并行和混合的差值区间也覆盖零；相对单 Agent 的固定策略差值区间不覆盖零，但仍不是“拓扑本身更好”的因果证据。</p>'))
    parts.append(section('methods', '10', '实验边界、审计与下一步',
        table(['约束', '具体设置'], [['股票池', '当前标普成分中500只有目标年份行情；研究期495只、训练期491只有观测。存在幸存者和未来成分偏差。'], ['搜索范围', '无固定家族、因子清单、参数网格或窗口范围；允许自行写数值策略，但禁止训练预测模型、外部数据和文件访问。'], ['预算与选择', '每组最多170,000总token、12次候选；所有失败和管理者调用计费。仅按验证扣费后Sharpe选一份，不组合worker策略。'], ['仓位', '总敞口≤1；单股绝对权重≤2%；净敞口绝对值≤1%。'], ['执行与费用', '收盘决策→下个有效开盘成交→再下个开盘结算；各划分空仓开始，历史可预热；交易5bp，借券年化30bp。'], ['数据缺失', '跳过2021-05-05、2025-11-12两个不完整横截面，不填补执行价格；跨会话收益，年化仍252。个别策略自行填补信号输入。'], ['执行限制', '策略调用限时10秒，每段回测180秒；进程隔离，未实施严格RAM上限。']])
        + '<h3>哪些事情已经核实？</h3><ul><li>10 次研究尝试都结束后，才对 7 份有效提交开启测试；3 组失败未追加替代实验。</li><li>7 份冻结内容 SHA、实现版本和测试引用一致；登记 7 次完成，每份仅测试一次、500 个收益区间。</li><li>已从逐日文件重新复算收益、Sharpe、波动、回撤和成本，与评估记录一致；预算均未超限。</li><li>本报告只读取已保存数据，没有新增模型调用、重跑策略或根据测试结果调整代码。</li></ul>'
        + '<h3>哪些问题仍然没有解决？</h3><ul><li>2024–2025 已用于上一轮实验，也可能存在于模型训练资料；本轮冻结顺序正确，不等于新的盲测。</li><li>每种结构只计划两次，实际仅 7/10 组完成；共享验证期反复选优，可能发生验证过拟合。</li><li>缺少历史逐日成分、退市处理和相同约束下的传统动量/低波动基线。</li><li>未模拟冲击、订单成交约束、真实借券可得性和变动费率，也未计保证金融资与抵押现金利息。</li></ul>'
        + '<h3>下一轮按什么顺序做？</h3><ol class="steps"><li><strong>先解决协议和代码交付。</strong>用独立合成数据预试 JSON 包装、worker 标识、嵌套分工及输出预算；预先固定格式修复范围和统一计费政策。</li><li><strong>分开效率与策略质量。</strong>一套固定总 token 比较端到端产出；另一套固定有效候选数并记录额外资源，比较搜索覆盖量匹配后的表现。</li><li><strong>做真正的通信消融。</strong>预设相同调度和预算，比较同伴内容开/关；增加重复并事前做功效评估，保存获胜候选的信息来源。</li><li><strong>加入传统基线与持仓记录。</strong>匹配成本、执行和仓位约束，比较动量/低波动基线，未来记录因子暴露、持仓重叠和收益相关。</li><li><strong>使用更可信的数据与前瞻验证。</strong>重建历史成分与退市，规则冻结后在未参与调试的时间窗和纸面前瞻环境验证。</li></ol>'
        + '<p class="source">这些是后续验证方案，不是已经完成的新实验或投资部署建议。</p>'))
    nav = [('summary', '核心结论'), ('experiment', '实验设定'), ('topologies', '五种拓扑图'), ('results', '结构比较'), ('time', '年份与净值'), ('collaboration', '协作归因'), ('strategies', '策略与代码'), ('diversity', '搜索相似性'), ('generation', '生成效率'), ('costs', '成本压力'), ('uncertainty', '风险与区间'), ('methods', '方法与后续')]
    parts = [re.sub(r'<div class="section-no">\d+</div>', f'<div class="section-no">{i:02d}</div>', part, count=1)
             for i, part in enumerate(parts, 1)]
    chart_data = dict(aggregates=data['aggregates'], runs=[dict(label=r['label'], topology=r['topology'], nav=r['monthly_nav']) for r in valid])
    embedded = json.dumps(chart_data, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('&', '\\u0026')
    template = Path(__file__).with_name('open_report_template.html').read_text()
    return template.replace('@@NAV@@', ''.join(f'<a href="#{key}">{i:02d}<span>{label}</span></a>' for i, (key, label) in enumerate(nav, 1))).replace('@@BODY@@', '\n'.join(parts)).replace('@@DATA@@', embedded)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    args = parser.parse_args()
    root = Path(args.root)
    target = root / 'QuantScale-实验分析报告.html'
    target.write_text(build(root))
    print(target.resolve())
