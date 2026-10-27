"""Bounded five-topology DeepSeek pilot; reveal holdout only after all freezes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .config import Config
from .data import prepare_partitions
from .deepseek_adapter import DeepSeekBackend
from .evaluation import evaluate, implementation_hash
from .research import TOPOLOGIES, run_research


def run_pilot(source, config_path, output, seeds=(7, 17), workers=2):
    config = Config.load(config_path)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    code_hash = implementation_hash()
    plan = [{"topology": topology, "fleet_size": 1 if topology == "single" else 4, "seed": seed}
            for seed in seeds for topology in TOPOLOGIES]
    # Save the order before any model call. No outcome-dependent scheduling.
    np.random.default_rng(20261026).shuffle(plan)
    protocol = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": str(Path(source).resolve()), "config": config.to_dict(),
                "implementation_sha256": code_hash, "schedule": plan,
                "concurrent_systems": workers, "total_trial_cap": len(plan) * config.backtests,
                "total_token_cap": len(plan) * config.tokens,
                "replication_note": "seed is a request/repetition label; provider RNG seed is not controlled",
                "missing_data_policy_note": "Skip incomplete cross-sectional sessions after first observed quote; "
                                            "no price imputation; adjacent open-to-open intervals span skipped dates",
                "scope": ("open Python policies" if config.strategy_mode == "python" else "restricted rule space")
                         + "; current-constituent survivorship bias; historical holdout"}
    (root / "pilot-protocol.json").write_text(json.dumps(protocol, indent=2))
    research_path, evaluator_path = prepare_partitions(source, root / "operator-data", config)
    env_file = Path.cwd() / ".env"

    def research(item):
        if implementation_hash() != code_hash:
            raise RuntimeError("Source code changed during the preregistered pilot")
        name = "%s-n%d-s%d" % (item["topology"], item["fleet_size"], item["seed"])
        start = time.monotonic()
        result = run_research(research_path, replace(config, seed=item["seed"]), root / name,
                              item["topology"], item["fleet_size"], DeepSeekBackend(env_file))
        return name, item, result, time.monotonic() - start

    completed, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(research, item): item for item in plan}
        for future in as_completed(futures):
            item = futures[future]
            try:
                name, item, result, elapsed = future.result()
                completed.append({"run": name, **item, "research_seconds": elapsed,
                                  "validation_sharpe": result["winner"]["validation"]["sharpe"],
                                  "strategy_id": result["winner"]["strategy_id"], **result["process"]})
                print("FROZEN " + name, flush=True)
            except Exception as exc:
                failures.append({**item, "error": str(exc)})
                print("FAILED " + str(item) + ": " + str(exc), flush=True)
            pd.DataFrame(completed).to_csv(root / "research-summary.csv", index=False)
            (root / "research-failures.json").write_text(json.dumps(failures, indent=2))
    if failures or len(completed) != len(plan):
        raise RuntimeError("Some research conditions failed; no hidden tests were opened. See research-failures.json")
    if implementation_hash() != code_hash:
        raise RuntimeError("Implementation changed; hidden tests remain closed")
    roster = {item["run"]: json.loads((root / item["run"] / "frozen.json").read_text())["artifact_sha256"]
              for item in completed}
    (root / "all-strategies-frozen.json").write_text(json.dumps({
        "frozen_before_any_evaluation_utc": datetime.now(timezone.utc).isoformat(),
        "submissions": roster}, indent=2))
    rows = []
    for item in sorted(completed, key=lambda row: (row["topology"], row["seed"])):
        name = item["run"]
        selection = json.loads((root / name / "selection.json").read_text())
        result = evaluate(root / name / "frozen.json", evaluator_path, root / "hidden-evaluation" / name)
        row = {**item, "validation_test_sharpe_gap": result["validation_test_sharpe_gap"],
               **{"hidden_" + key: value for key, value in result["hidden"].items()},
               "strategy": json.dumps(selection["strategy"], sort_keys=True),
               "excluded_data_sessions": json.dumps(result["excluded_data_sessions"])}
        rows.append(row)
        pd.DataFrame(rows).to_csv(root / "comparison.csv", index=False)
        print("EVALUATED %s hidden_sharpe=%.4f" % (name, result["hidden"]["sharpe"]), flush=True)
    comparison = pd.DataFrame(rows)
    aggregate = comparison.groupby("topology").agg(
        repetitions=("seed", "count"),
        validation_sharpe_mean=("validation_sharpe", "mean"),
        hidden_sharpe_mean=("hidden_sharpe", "mean"),
        hidden_sharpe_min=("hidden_sharpe", "min"),
        hidden_sharpe_max=("hidden_sharpe", "max"),
        hidden_annualized_return_mean=("hidden_annualized_return", "mean"),
        hidden_max_drawdown_mean=("hidden_max_drawdown", "mean"),
        mean_total_tokens=("total_tokens", "mean"),
        mean_duplicate_variant_rate=("duplicate_variant_rate", "mean"),
    ).reset_index()
    if "family_entropy" in comparison:
        entropy = comparison.groupby("topology").family_entropy.mean()
        aggregate["mean_family_entropy"] = aggregate.topology.map(entropy)
    aggregate.to_csv(root / "topology-summary.csv", index=False)
    (root / "completed.json").write_text(json.dumps({
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": len(rows), "actual_model_tokens": int(comparison.total_tokens.sum()),
        "actual_candidate_trials": int(comparison.total_backtests.sum()),
        "aggregate": aggregate.to_dict(orient="records")}, indent=2))
    return aggregate.to_dict(orient="records")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/sp500-yahoo-2016-2025/prices.csv")
    parser.add_argument("--config", default="configs/sp500-pilot.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    print(json.dumps(run_pilot(args.source, args.config, args.output, workers=args.workers), indent=2))


if __name__ == "__main__":
    main()
