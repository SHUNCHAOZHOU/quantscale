"""Build an embedded-data Canvas only after the full pilot has completed."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build(root):
    root = Path(root)
    completed = json.loads((root / "completed.json").read_text())
    frame = pd.read_csv(root / "comparison.csv")
    if completed["runs"] != 10 or len(frame) != 10:
        raise ValueError("The full ten-run pilot must complete before reporting")
    order = ["single", "independent", "centralized", "decentralized", "hybrid"]
    frame["order"] = frame.topology.map({name: i for i, name in enumerate(order)})
    frame = frame.sort_values(["order", "seed"])
    aggregates = pd.read_csv(root / "topology-summary.csv").set_index("topology").loc[order].reset_index()
    extra = frame.groupby("topology").agg(
        mean_communication_tokens_upper_bound=("communication_turn_tokens_upper_bound", "mean"),
        mean_unique_variants=("unique_strategy_variants", "mean"),
        mean_successful_candidates=("successful_candidates", "mean"),
        mean_sharpe_gap=("validation_test_sharpe_gap", "mean"))
    aggregates = aggregates.merge(extra, on="topology", validate="one_to_one")
    rows, navs, fingerprints = [], {}, set()
    for record in frame.to_dict(orient="records"):
        run = record["run"]
        selection = json.loads((root / run / "selection.json").read_text())
        record["strategy"] = selection["strategy"]
        record["hypothesis"] = selection["hypothesis"]
        trials = [json.loads(line) for line in (root / run / "trajectory.jsonl").read_text().splitlines()]
        record["invalid_trials"] = sum(event["kind"] == "trial" and "error" in event for event in trials)
        for event in trials:
            if event["kind"] == "response":
                fingerprints.add(str(event["response"].get("_provenance", {}).get("system_fingerprint")))
        daily = pd.read_csv(root / "hidden-evaluation" / run / "hidden_daily.csv", parse_dates=["date"])
        daily["nav"] = np.cumprod(1 + daily.net_return)
        monthly = daily.groupby(daily.date.dt.to_period("M"), sort=True).tail(1)
        navs[run] = {"dates": ["2024-01-02"] + monthly.date.dt.strftime("%Y-%m-%d").tolist(),
                     "nav": [1.0] + monthly.nav.tolist()}
        rows.append(record)
    payload = {"completed": completed, "rows": rows, "aggregates": aggregates.to_dict(orient="records"),
               "navs": navs, "fingerprints": sorted(fingerprints)}
    (root / "report-data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return '''import { Stack, Row, H1, H2, Text, Callout, Table, BarChart, LineChart, Pill, useState } from "cursor/canvas";
const DATA = ''' + data + ''';
const ORDER = ["single", "independent", "centralized", "decentralized", "hybrid"] as const;
type Topology = typeof ORDER[number];
const NAMES: Record<Topology,string> = {single:"SAS · 1",independent:"Independent · 4",centralized:"Centralized · 4+1",decentralized:"Decentralized · 4",hybrid:"Hybrid · 4+1"};
const FAMILY: Record<string,string> = {momentum:"动量",reversal:"反转",low_volatility:"低波动",volume:"成交量异常"};
const NAVS: Record<string,{dates:string[];nav:number[]}> = DATA.navs;
const f = (x:number) => Number(x).toFixed(3);
const p = (x:number) => (100*Number(x)).toFixed(2)+"%";
export default function QuantScalePilot() {
  const [topology,setTopology] = useState<Topology>("single");
  const selected = DATA.rows.filter(r => r.topology===topology);
  const best = [...DATA.aggregates].sort((a,b)=>b.hidden_sharpe_mean-a.hidden_sharpe_mean)[0];
  return <Stack gap={18} style={{padding:24,maxWidth:1100,margin:"0 auto"}}>
    <H1>QuantScale · 五种拓扑的真实行情结果</H1>
    <Text>DeepSeek Flash · 当前标普成分股 · 2016–2021 训练 / 2022–2023 验证 / 2024–2025 隐藏测试</Text>
    <Callout tone="info" title={`${NAMES[best.topology as Topology]} 在这轮的隐藏 Sharpe 均值最高：${f(best.hidden_sharpe_mean)}`}>
      每种拓扑仅两次独立运行，以下均为描述性结果，不能据此确认稳定的拓扑优劣。全部十个策略在开启任何隐藏评估前已冻结。
    </Callout>
    <Text>每组最多 12 个候选、200,000 token；中心式与混合式的“4+1”表示 4 名 worker 加 1 名管理器。管理器和通信计入同一预算。</Text>
    <H2>验证与隐藏 Sharpe：两次运行的算术均值</H2>
    <section aria-label="各拓扑验证和隐藏净收益 Sharpe 比较">
      <Text size="small" tone="secondary">横轴：拓扑 · 纵轴：扣费后年化 Sharpe（无量纲，现金基准为零）</Text>
      <BarChart categories={ORDER.map(t=>NAMES[t])} series={[
        {name:"验证 Sharpe 均值 · 2022–2023",data:DATA.aggregates.map(r=>r.validation_sharpe_mean),tone:"neutral"},
        {name:"隐藏 Sharpe 均值 · 2024–2025",data:DATA.aggregates.map(r=>r.hidden_sharpe_mean),tone:"info"}
      ]} height={310} showValues referenceLines={[{value:0,label:"零 Sharpe",tone:"neutral"}]}/>
      <Text size="small" tone="secondary">来源：本次 topology-summary.csv。均值针对逐次 Sharpe，不是组合策略 Sharpe。</Text>
    </section>
    <H2>十次运行的实际结果</H2>
    <Table headers={["拓扑","重复标签","验证 Sharpe","隐藏 Sharpe","隐藏年化收益","隐藏最大回撤","实际 token","无效候选"]}
      rows={DATA.rows.map(r=>[NAMES[r.topology as Topology],r.seed,f(r.validation_sharpe),f(r.hidden_sharpe),p(r.hidden_annualized_return),p(r.hidden_max_drawdown),r.total_tokens.toLocaleString(),r.invalid_trials])}
      columnAlign={["left","right","right","right","right","right","right","right"]}/>
    <Text size="small" tone="secondary">来源：comparison.csv 与 trajectory.jsonl。重复标签 7、17 不控制服务端 RNG。交易成本 5 bp / 交易名义量，借券年化 30 bp。</Text>
    <H2>研究成本、重复搜索与方向多样性</H2>
    <Table headers={["拓扑","实际 token 均值","通信相关调用 token 上界均值","唯一策略均值 / 12","重复率均值","策略家族熵均值"]}
      rows={DATA.aggregates.map(r=>[NAMES[r.topology as Topology],Math.round(r.mean_total_tokens).toLocaleString(),Math.round(r.mean_communication_tokens_upper_bound).toLocaleString(),r.mean_unique_variants.toFixed(1),p(r.mean_duplicate_variant_rate),f(r.mean_family_entropy)])}
      columnAlign={["left","right","right","right","right","right"]}/>
    <Text size="small" tone="secondary">来源：process.json，逐次指标算术平均。重复率按相同策略参数 SHA-256 计算；家族熵仅衡量四种规则家族，不等同于语义多样性。通信上界包括带 peer 上下文的完整调用。</Text>
    <H2>具体策略与隐藏期净值</H2>
    <Row gap={6} wrap>{ORDER.map(t=><Pill key={t} active={t===topology} onClick={()=>setTopology(t)}>{NAMES[t]}</Pill>)}</Row>
    {selected.map(r=><Text key={r.run}>重复 {r.seed}：{FAMILY[r.strategy.family]}，窗口 {r.strategy.lookback} 日，跳过 {r.strategy.skip} 日，多空各选 {p(r.strategy.fraction)}，{r.strategy.volatility_scale?"按波动率缩放仓位":"等权仓位"}。</Text>)}
    <section aria-label="所选拓扑的两次隐藏期策略净值">
      <Text size="small" tone="secondary">横轴：月末日期 · 纵轴：净值（初始为 1，月末抽样）</Text>
      <LineChart categories={NAVS[selected[0].run].dates} series={selected.map((r,i)=>({name:`重复 ${r.seed}`,data:NAVS[r.run].nav,tone:(i===0?"info":"neutral") as "info"|"neutral"}))} height={290} beginAtZero={false} referenceLines={[{value:1,label:"初始净值",tone:"neutral"}]}/>
      <Text size="small" tone="secondary">来源：hidden-evaluation/*/hidden_daily.csv；日净收益连续复合后抽取月末值。表中回撤使用完整日序列计算。</Text>
    </section>
    <H2>适用范围与数据处理</H2>
    <Text>股票池共有 500 只，其中 495 只在研究期有记录。使用当前成分股，存在幸存者与未来成分信息偏差；不是历史逐日标普股票池。</Text>
    <Text>规则空间仅含动量、反转、低波动与成交量异常的参数变体；没有自由写 Python 或训练预测模型。每组有相同的预算上限，实际 token 消耗不同。</Text>
    <Text>按预先固定的质量规则排除 2021-05-05 和 2025-11-12 两个不完整横截面会话，不填补价格；相邻开盘区间跨过被排除日期，年化仍用 252 个有效观测。</Text>
    <Text>隐藏期是历史数据留出，不保证未出现在 LLM 预训练资料中。结果包含交易和借券成本，但未模拟市场冲击、真实借券可得性和容量。不是可部署投资结论。</Text>
    <Text size="small" tone="secondary">完整轨迹、冻结凭据与单次评估均保存在 results/sp500-deepseek-topology-pilot/。本轮总候选 {DATA.completed.actual_candidate_trials}，实际模型 token {DATA.completed.actual_model_tokens.toLocaleString()}。</Text>
  </Stack>;
}
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    # The managed canvas directory is provisioned by the host; do not create it.
    Path(args.output).write_text(build(args.root))
