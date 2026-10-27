"""Inline SVG diagrams of the actual pilot information flow."""


def node(x, y, label, subtitle='', manager=False, width=100):
    kind = 'manager' if manager else 'worker'
    return (f'<g class="{kind}"><rect x="{x-width/2}" y="{y-24}" width="{width}" height="48" rx="7"/>'
            f'<text x="{x}" y="{y+5}" text-anchor="middle">{label}</text></g>'
            + (f'<text class="node-sub" x="{x}" y="{y+45}" text-anchor="middle">{subtitle}</text>' if subtitle else ''))


def arrow(identifier, path, kind='management', both=False):
    start = f' marker-start="url(#{identifier}-{kind})"' if both else ''
    return f'<path class="edge {kind}" d="{path}" marker-end="url(#{identifier}-{kind})"{start}/>'


def svg(identifier, title, body, height=320):
    defs = ''.join(f'<marker id="{identifier}-{kind}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-{kind}" d="M 0 0 L 10 5 L 0 10 Z"/></marker>' for kind in ['management', 'feedback', 'peer'])
    return f'<svg class="topology-svg" viewBox="0 0 560 {height}" role="img" aria-label="{title}"><title>{title}</title><defs>{defs}</defs>{body}</svg>'


def graph(kind):
    if kind == 'single':
        body = arrow(kind, 'M 290 114 L 290 197', 'feedback')
        body += arrow(kind, 'M 183 220 C 85 220 90 90 203 90', 'feedback')
        body += node(280, 90, 'W0 · 研究者', width=145)
        body += '<rect class="context-box" x="180" y="198" width="200" height="44" rx="5"/><text class="context-label" x="280" y="225" text-anchor="middle">自身最近 3 条候选结果</text>'
        body += '<text class="edge-label" x="365" y="158">回测反馈</text><text class="edge-label" x="95" y="167">下次提案读取</text>'
        body += '<text class="node-sub" x="280" y="285" text-anchor="middle">1 名研究者，无管理者、无同伴共享</text>'
        return svg(kind, '单 Agent：一个研究者读取自己的历史反馈', body)
    if kind == 'independent':
        body = ''.join(node(x, y, f'W{i}', '仅自身最近 3 条结果') for i, (x, y) in enumerate([(135, 85), (425, 85), (135, 215), (425, 215)]))
        body += '<text class="node-sub" x="280" y="300" text-anchor="middle">W 之间无消息边，不共享同伴策略或回测结果</text>'
        return svg(kind, '独立并行：四个研究者互不共享消息', body)
    if kind == 'decentralized':
        coords = [(105, 85), (455, 85), (105, 230), (455, 230)]
        paths = ['M 162 85 L 398 85', 'M 162 230 L 398 230', 'M 105 114 L 105 201',
                 'M 455 114 L 455 201', 'M 161 110 L 399 205', 'M 399 110 L 161 205']
        body = ''.join(arrow(kind, path, 'peer', True) for path in paths)
        body += ''.join(node(x, y, f'W{i}') for i, (x, y) in enumerate(coords))
        body += '<text class="node-sub" x="280" y="291" text-anchor="middle">同伴边：上一轮快照 → 下一轮提案，不是实时群聊</text>'
        return svg(kind, '去中心化：四个研究者按上一轮快照相互共享有效结果', body)
    # Manager assignment is current-round; the return path is previous-round context.
    hybrid = kind == 'hybrid'
    centers = [70, 210, 350, 490]
    body = ''.join(arrow(kind, f'M {252+i*18} 77 L {x} 147') for i, x in enumerate(centers))
    body += arrow(kind, 'M 130 305 L 18 305 L 18 53 L 205 53', 'feedback')
    body += ''.join(arrow(kind, f'M {x} 199 L {x} 277', 'feedback') for x in centers)
    if hybrid:
        # All six peer pairs; curves sit below the worker row.
        for i, left in enumerate(centers):
            for j in range(i + 1, len(centers)):
                right = centers[j]
                depth = 208 + (j-i)*23
                body += arrow(kind, f'M {left+8} 197 Q {(left+right)/2} {depth} {right-8} 197', 'peer', True)
    body += node(280, 53, 'M · 管理者', manager=True, width=150)
    body += ''.join(node(x, 172, f'W{i}') for i, x in enumerate(centers))
    body += '<rect class="context-box" x="46" y="279" width="468" height="51" rx="5"/><text class="context-label" x="280" y="301" text-anchor="middle">上一轮：每个 W 最近一次有效结果</text><text class="node-sub" x="280" y="319" text-anchor="middle">软件整理的快照，不是额外的 Agent</text>'
    body += '<text class="edge-label" x="340" y="108">本轮分工</text><text class="edge-label" x="24" y="262">下轮读取</text>'
    return svg(kind, '混合：管理者分工，加上研究者之间的同伴结果共享' if hybrid else '中心化：管理者读取上轮结果并向四个研究者分配方向，研究者不直接互通', body, 350)


def topology_html():
    specifications = [
        ('single', '单 Agent', '1 名研究者', '每次提案读取自己最近 3 条候选结果，包括错误记录。用同一份训练与验证反馈进行下一次尝试。'),
        ('independent', '独立并行', '4 名研究者，无管理者', '各自只看自身历史。12 次候选上限按每人最多 3 次分配，共享一组 token 预算；实际可能提前耗尽。'),
        ('centralized', '中心化', '4 名研究者 + 1 名管理者', '每轮开始时，管理者读取各 W 截至上一轮的最近有效结果并给每人分工。W 读取分工和自身历史，不直接读取其他 W 的结果。'),
        ('decentralized', '去中心化', '4 名研究者，无管理者', '每名 W 可读取其他 W 在上一轮结束前最近一次有效结果，里面包含假设、代码和训练/验证指标。软件只负责整理快照。'),
        ('hybrid', '混合', '4 名研究者 + 1 名管理者', '同时保留管理者分工和同伴共享：每名 W 读取本轮分工、自身历史，以及上轮其他 W 的最近有效结果。'),
    ]
    body = '<p>五组使用同一类研究者、数据接口和回测器，区别在于<strong>下一次提案能读取哪些信息</strong>。图中的 W 是生成策略代码的研究者，M 是负责分工的管理者；评估器与数据属于公共实验设施，不计入 Agent 数。</p>'
    body += '<div class="topology-legend"><span><i class="management-key"></i>管理者 → 研究者：研究分工</span><span><i class="peer-key"></i>研究者 ↔ 研究者：同伴有效结果</span><span><i class="feedback-key"></i>虚线：历史结果 / 下一轮反馈</span></div>'
    for identifier, title, fleet, explanation in specifications:
        body += f'<article class="topology-figure"><div class="topology-drawing">{graph(identifier)}</div><div class="topology-explanation"><div class="eyebrow">{identifier.upper()}</div><h3>{title}</h3><p class="fleet-count">{fleet}</p><p>{explanation}</p></div></article>'
    body += '<div class="note"><strong>图示需要结合三条实际执行规则理解。</strong><ul><li>所有 W 都只收到自身最近 3 条候选结果；图中未在每个 W 下重复画本地历史。</li><li>同伴与管理者读取的是“截至上一轮的快照”。本轮先调用的 W 即使已产生新结果，也不会立即分享给同轮其他 W；没有有效结果时，相应上下文为空。</li><li>W 在每组内按编号顺序调用，不是四路同时执行；实验并发 4 路指不同实验组。每组最后统一按验证 Sharpe 选 1 份策略，不把多名 W 的策略自动组合成投资组合。</li></ul></div>'
    body += '<p class="source">图源：正式实验冻结的 research.py 信息流。实线和虚线表示逻辑信息关系，不代表同一时刻通信或额外模型调用。管理者和同伴内容均计入该组统一 token 预算。</p>'
    return body


def setup_html(table):
    return '''<p>这是一次<strong>固定总资源上限的开放策略代码搜索</strong>：比较相同模型和交易约束下，不同信息结构能完成多少有效研究，以及选出的策略在留出期如何表现。</p>
    <div class="experiment-timeline" aria-label="2016至2021训练、2022至2023验证、2024至2025测试"><div class="phase train"><span>训练 · 6 年</span><strong>2016–2021</strong><small>反馈研究期表现</small></div><div class="phase validation"><span>验证 · 2 年</span><strong>2022–2023</strong><small>反馈并选择优胜者</small></div><div class="phase test"><span>测试 · 2 年</span><strong>2024–2025</strong><small>冻结后仅评估一次</small></div></div>
    <p class="source">前 8 年用于挖掘；其中后 2 年用于验证选优。测试数据和测试收益不放入研究者请求。此测试年份已用于上一轮实验，属于本轮流程留出，不是新盲测。</p>''' + table(['实验要素', '正式重跑的设定'], [
        ['股票与数据', 'Yahoo 真实行情；当前标普成分中500只有目标年份数据，研究期495只、训练期491只有观测。不是历史逐日标普成分池。'],
        ['模型与输出', 'deepseek-flash，high推理；单次最多16,384输出token，包含推理；请求总token还受单次65,536及剩余预算限制。'],
        ['重复与总组数', '5种拓扑 × 2次运行（标签7、17）= 10组；标签不控制服务端随机数。计划上限120次候选、1,700,000 token。'],
        ['每组预算', '最多12次候选尝试、170,000总token；管理者、同伴上下文、格式错误和失败调用均计费。不保证实际消耗或有效候选数相同。'],
        ['搜索自由度', '研究者自己写Python特征、信号、过滤器与仓位；不限定策略家族、窗口或参数网格。仍禁止预测模型训练、外部数据和文件访问。'],
        ['仓位限制', '总敞口≤100%；单股绝对权重≤2%；净敞口绝对值≤1%。模型需返回每只当期股票的有限权重，未持有项为0。'],
        ['执行与费用', '收盘t决策→下个有效开盘成交→再下个开盘结算；交易名义量按权重绝对变化之和计算，费率5bp，空头借券年化30bp。'],
        ['选优与测试', '每组按验证扣费后Sharpe选1份，平分按策略SHA升序；10组研究尝试全部结束后统一登记，7份有效提交各测试1次，3组失败不补跑。'],
        ['执行限制', '每次策略调用限时10秒，每段回测限时180秒；隔离进程，未实施严格RAM上限。各数据划分从空仓开始，可读取之前历史预热。'],
    ]) + '''<h3>一个候选如何变成最终测试策略？</h3><ol class="experiment-flow"><li><b>提出候选</b><span>假设 + 完整Python模块</span></li><li><b>运行研究回测</b><span>代码检查 → 训练 → 验证</span></li><li><b>反馈并继续</b><span>依据拓扑提供自身 / 同伴 / 分工信息</span></li><li><b>选出并冻结</b><span>每组取验证Sharpe最高的1份</span></li><li><b>统一测试</b><span>全部组结束后，最后2年各评估1次</span></li></ol>
    <p>研究者提交错误或回测失败：扣一次候选并继续。管理者输出不满足协议：当前实现直接终止该组，这是本轮需要披露的惩罚不对称。反馈没有形成有效结果时，同伴快照中不会凭空出现可共享的策略。</p>
    <p class="source">来源：pilot-protocol.json、正式冻结的 config.py / prompts.py / research.py / backtest.py。以上是已经运行的设定，不是新增实验。</p>'''
