"""Summarize an open-code pilot, retaining missing and failed conditions."""
import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd


ORDER = ['single', 'independent', 'centralized', 'decentralized', 'hybrid']


def collect(root):
    root = Path(root)
    completed = json.loads((root / 'completed.json').read_text())
    protocol = json.loads((root / 'pilot-protocol.json').read_text())
    evaluated = {}
    if (root / 'comparison.csv').exists():
        evaluated = {r['run']: r for r in pd.read_csv(root / 'comparison.csv').to_dict('records')}
    evaluation_errors = {x['run']: x['error'] for x in completed['evaluation_failures']}
    records = []
    for item in sorted(protocol['schedule'], key=lambda x: (ORDER.index(x['topology']), x['seed'])):
        name = '%s-n%d-s%d' % (item['topology'], item['fleet_size'], item['seed'])
        folder = root / name
        journal = [json.loads(x) for x in (folder / 'trajectory.jsonl').read_text().splitlines()]
        trials = [x for x in journal if x['kind'] == 'trial']
        responses = [x for x in journal if x['kind'] == 'response']
        errors = Counter(x['error'] for x in trials if 'error' in x)
        aborted = [x['error'] for x in journal if x['kind'] == 'aborted']
        record = dict(item, run=name, trials=len(trials), valid=sum('validation' in x for x in trials),
                      tokens=sum(x['tokens'] for x in responses), calls=len(responses),
                      truncated_calls=sum('did not finish: length' in x['response'].get('adapter_error', '') for x in responses),
                      completion_tokens=sum(x['response'].get('_provenance', {}).get('usage', {}).get('completion_tokens', 0) for x in responses),
                      reasoning_tokens=sum(x['response'].get('_provenance', {}).get('usage', {}).get('completion_tokens_details', {}).get('reasoning_tokens', 0) for x in responses),
                      errors=dict(errors),
                      status='evaluated' if name in evaluated else
                             'evaluation_failed' if name in evaluation_errors else 'no_submission',
                      failure=evaluation_errors.get(name, '; '.join(aborted)))
        if name in evaluated:
            for key in ['validation_sharpe', 'hidden_sharpe', 'hidden_annualized_return',
                        'hidden_max_drawdown', 'hidden_total_return', 'hidden_observations']:
                if key in evaluated[name]:
                    record[key] = evaluated[name][key]
        if (folder / 'selection.json').exists():
            selection = json.loads((folder / 'selection.json').read_text())
            record['hypothesis'] = selection['hypothesis']
            record['strategy_id'] = selection['strategy_id']
            record['source_file'] = str((folder / 'strategies' / (selection['strategy_id'] + '.py')).resolve())
        records.append(record)
    returns = {}
    for record in records:
        if record['status'] == 'evaluated':
            daily = pd.read_csv(root / 'hidden-evaluation' / record['run'] / 'hidden_daily.csv')
            returns[record['run']] = daily.set_index('date').net_return
    correlations = []
    for i, left in enumerate(returns):
        for right in list(returns)[i + 1:]:
            paired = pd.concat([returns[left], returns[right]], axis=1).dropna()
            if len(paired) > 1 and (paired.std() > 0).all():
                correlations.append(dict(left=left, right=right,
                                         correlation=float(paired.iloc[:, 0].corr(paired.iloc[:, 1])),
                                         observations=len(paired)))
    correlations.sort(key=lambda x: x['correlation'], reverse=True)
    payload = dict(protocol=protocol, completed=completed, runs=records, correlations=correlations)
    (root / 'open-report-data.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return payload


def canvas(payload):
    return '''import {Stack,H1,H2,Text,Callout,Table,BarChart} from "cursor/canvas";
const DATA = ''' + json.dumps(payload, ensure_ascii=False, allow_nan=False) + ''';
const NAMES:Record<string,string>={single:"单 Agent",independent:"独立并行",centralized:"中心化",decentralized:"去中心化",hybrid:"混合"};
const STATUS:Record<string,string>={evaluated:"已完成测试",evaluation_failed:"测试执行失败",no_submission:"无有效提交"};
const f=(x:unknown)=>typeof x==="number"?x.toFixed(3):"—";
const pct=(x:unknown)=>typeof x==="number"?(100*x).toFixed(2)+"%":"—";
export default function OpenPilot(){
 const runs:any[]=DATA.runs;
 const valid=runs.filter(r=>r.status==="evaluated");
 const aggregate:any[]=DATA.completed.aggregate;
 const best=[...aggregate].sort((a,b)=>b.hidden_sharpe_mean-a.hidden_sharpe_mean)[0];
 return <Stack gap={18} style={{padding:24,maxWidth:1150,margin:"0 auto"}}>
 <H1>QuantScale · 开放策略代码实验</H1>
 <Text>DeepSeek Flash · 真实 Yahoo 行情 · 2016–2021 训练 / 2022–2023 验证 / 2024–2025 测试</Text>
 <Callout tone="warning" title={`${DATA.completed.runs} / ${DATA.completed.planned_runs} 组完成测试`}>
 每种结构计划两次运行。失败组保留在分母中，缺失收益不按零收益填补。样本很小，不能据此确认结构的稳定优劣；测试年份也已在上一轮实验使用过。
 </Callout>
 {best&&<Text><strong>本轮完成测试组中，{NAMES[best.topology]}的平均测试 Sharpe 最高：{f(best.hidden_sharpe_mean)}，年化收益均值 {pct(best.hidden_annualized_return_mean)}。</strong>这是成功组的描述性结果，不能确认因果关系。</Text>}
 <Text>允许自主编写数值策略、特征和仓位规则；没有预设策略家族、参数网格或窗口范围。仍受已提供行情字段、多空仓位规则、禁止预测模型训练与执行环境约束。</Text>
 <H2>全部计划运行：收益与有效产出</H2>
 <Table headers={["结构","重复标签","状态","有效 / 尝试","实际 token","验证 Sharpe","测试 Sharpe","测试年化收益","测试最大回撤"]}
 rows={runs.map(r=>[NAMES[r.topology],r.seed,STATUS[r.status],`${r.valid} / ${r.trials}`,r.tokens.toLocaleString(),f(r.validation_sharpe),f(r.hidden_sharpe),pct(r.hidden_annualized_return),pct(r.hidden_max_drawdown)])}/>
 <Text size="small" tone="secondary">来源：comparison.csv、trajectory.jsonl、completed.json。Sharpe 为扣费后日收益以 252 年化、现金基准零；年化收益为复合口径。重复标签不控制服务端随机数。</Text>
 {aggregate.length>0&&<Stack gap={8}><H2>完成测试组的 Sharpe 均值</H2>
 <Text size="small" tone="secondary">横轴：结构（括号内为完成数 / 计划数） · 纵轴：年化 Sharpe，无量纲</Text>
 <BarChart categories={aggregate.map(r=>`${NAMES[r.topology]} (${r.repetitions}/2)`)} series={[
 {name:"验证期 2022–2023",data:aggregate.map(r=>r.validation_sharpe_mean),tone:"neutral"},
 {name:"测试期 2024–2025",data:aggregate.map(r=>r.hidden_sharpe_mean),tone:"info"}]} height={300} showValues referenceLines={[{value:0,label:"零 Sharpe",tone:"neutral"}]}/>
 <Text size="small" tone="secondary">来源：topology-summary.csv；仅针对成功测试组的条件均值，不代表包含失败风险的整体效用；逐组 Sharpe 的均值不是组合 Sharpe。</Text></Stack>}
 {valid.length>0&&<Stack gap={8}><H2>最终策略：模型原始假设</H2>{valid.map(r=><Stack gap={4} key={r.run}><Text><strong>{NAMES[r.topology]} · {r.seed}</strong> · {r.strategy_id}</Text><Text>{r.hypothesis}</Text><Text size="small" tone="secondary">源代码：{r.source_file}</Text></Stack>)}</Stack>}
 {runs.some(r=>r.failure||Object.keys(r.errors).length)&&<Stack gap={8}><H2>失败记录</H2>
 <Table headers={["结构 / 重复","终止原因","候选错误及次数"]} rows={runs.filter(r=>r.failure||Object.keys(r.errors).length).map(r=>[`${NAMES[r.topology]} / ${r.seed}`,r.failure||"—",Object.entries(r.errors).map(([e,n])=>`${n} × ${e}`).join("；")||"—"])}/></Stack>}
 {DATA.correlations.length>0&&<Stack gap={8}><H2>测试收益最相关的五对策略</H2>
 <Table headers={["策略一","策略二","日净收益 Pearson 相关系数","共同交易观测数"]} rows={(DATA.correlations as any[]).slice(0,5).map(r=>[r.left,r.right,f(r.correlation),r.observations])}/>
 <Text size="small" tone="secondary">来源：已保存的 2024–2025 hidden_daily.csv，按日期内连接；未重新回测。相关性反映收益共同变化，不证明代码、仓位或经济逻辑相同。</Text></Stack>}
 <H2>预算、冻结与适用范围</H2>
 <Text>输出截断调用 {runs.reduce((s,r)=>s+r.truncated_calls,0)} 次。已报告推理 token {runs.reduce((s,r)=>s+r.reasoning_tokens,0).toLocaleString()}，占输出 token 的 {pct(runs.reduce((s,r)=>s+r.reasoning_tokens,0)/Math.max(1,runs.reduce((s,r)=>s+r.completion_tokens,0)))}。这是本次生成设置下的完成率结果，不能单独归因于拓扑。</Text>
 <Text>每组最多 12 个候选、170,000 token；管理者和通信共用该预算。实际 token 总数 {DATA.completed.actual_model_tokens.toLocaleString()}，模型调用 {DATA.completed.actual_model_calls} 次，候选尝试 {DATA.completed.actual_candidate_trials} 次。实际消耗不同，不能称为严格相等计算量。</Text>
 <Text size="small" tone="secondary">以上 token 仅统计正式重跑。此前因 worker 标识协议问题中止的准备运行另有 72,341 个已确认 token，3 个在途请求未取得用量回执；原始账本保留在 results/sp500-deepseek-open-pilot/infrastructure-abort.json。</Text>
 <Text>所有计划研究尝试结束后才统一冻结并开启测试；有效提交只尝试测试一次，失败不追加替代实验。代码单次调用限时 10 秒、每段回测限时 180 秒；操作系统未施加严格内存上限。执行失败反映策略、模型输出和运行条件的共同影响。</Text>
 <Text>股票池为当前成分股中有目标年份数据的 500 只，研究期 495 只；存在幸存者和未来成分信息偏差。缺失横截面会话按固定规则整体跳过，不填补价格。</Text>
 <Text>收盘决策、下个有效开盘交易，随后开盘到开盘收益；交易成本 5 bp，借券年化 30 bp。未模拟冲击成本、真实借券可得性或容量。历史测试不构成新盲测或可部署投资结论。</Text>
 </Stack>;
}
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--canvas', required=True)
    args = parser.parse_args()
    Path(args.canvas).write_text(canvas(collect(args.root)))
