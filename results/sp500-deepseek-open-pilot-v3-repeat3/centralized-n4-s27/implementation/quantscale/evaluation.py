"""Frozen submissions and an evaluator-side, durable once-only registry.

This is a protocol boundary, not a sandbox against a malicious local process.
Use a separate account/container to keep evaluator inputs and state private.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .config import Config
from .data import MarketData, fingerprint, load_data
from .strategies import RuleStrategy, StrategySpec


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def implementation_hash():
    root = Path(__file__).parent
    hasher = hashlib.sha256()
    for name in sorted(path.name for path in root.glob("*.py")):
        hasher.update(name.encode())
        hasher.update((root / name).read_bytes())
    return hasher.hexdigest()


def runtime_versions():
    return {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}


def freeze(root, protocol, selection):
    payload = {"protocol": protocol, "strategy": selection["strategy"],
               "validation": selection["validation"], "implementation_sha256": implementation_hash(),
               "runtime_versions": runtime_versions()}
    artifact = {"payload": payload, "artifact_sha256": digest(payload)}
    snapshot = Path(root) / "implementation" / "quantscale"
    snapshot.mkdir(parents=True, exist_ok=False)
    for source in Path(__file__).parent.glob("*.py"):
        (snapshot / source.name).write_bytes(source.read_bytes())
    (snapshot.parent / "runtime.json").write_text(json.dumps(runtime_versions(), indent=2))
    with (Path(root) / "frozen.json").open("x") as handle:
        json.dump(artifact, handle, indent=2, allow_nan=False)
    return artifact


def evaluate(frozen_path, evaluator_path, output):
    artifact = json.loads(Path(frozen_path).read_text())
    payload = artifact["payload"]
    if digest(payload) != artifact["artifact_sha256"]:
        raise ValueError("Frozen artifact has been modified")
    if payload["implementation_sha256"] != implementation_hash():
        raise ValueError("Implementation changed since freeze; use the original version")
    if payload["runtime_versions"] != runtime_versions():
        raise ValueError("Python/NumPy/Pandas versions differ from the frozen research runtime")
    config = Config.from_dict(payload["protocol"]["config"])
    python_code = payload["strategy"].get("kind") == "python"
    if python_code:
        from .code_strategy import CodeSpec, SandboxedStrategy
        spec = CodeSpec(source=payload["strategy"]["source"])
    else:
        spec = StrategySpec(**payload["strategy"])
    df = load_data(evaluator_path, config)
    research = df[df.date <= pd.Timestamp(config.validation.end)].reset_index(drop=True)
    if fingerprint(research) != payload["protocol"]["research_data_hash"]:
        raise ValueError("Evaluator warm-up data differs from the frozen research dataset")
    if df[(df.date >= pd.Timestamp(config.test.start)) & (df.date <= pd.Timestamp(config.test.end))].date.nunique() < 3:
        raise ValueError("Insufficient hidden data")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    # State lives with evaluator data, independent of the chosen output folder.
    registry = Path(evaluator_path).resolve().parent / ".quantscale-evaluations.sqlite3"
    key = artifact["artifact_sha256"]
    connection = sqlite3.connect(str(registry), timeout=30)
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS evaluations (artifact TEXT PRIMARY KEY, status TEXT NOT NULL, result TEXT)")
        try:
            with connection:
                connection.execute("INSERT INTO evaluations VALUES (?, 'running', NULL)", (key,))
        except sqlite3.IntegrityError:
            raise ValueError("This frozen submission has already been evaluated or attempted") from None
        try:
            market = MarketData(df[df.date <= pd.Timestamp(config.test.end)], config)
            policy = SandboxedStrategy(spec, config) if python_code else RuleStrategy(spec, config)
            metrics, daily = run_backtest(policy, market, config.test)
            annual = {}
            for year in sorted(daily.date.dt.year.unique()):
                # Descriptive calendar slices of the same realized return path.
                from .backtest import performance
                annual[str(year)] = performance(daily[daily.date.dt.year == year], config.annualization)
            result = {"artifact_sha256": key, "evaluated_at": datetime.now(timezone.utc).isoformat(),
                      "evaluator_data_sha256": fingerprint(df), "hidden": metrics,
                      "validation": payload["validation"],
                      "validation_test_sharpe_gap": payload["validation"]["sharpe"] - metrics["sharpe"],
                      "positive_hidden_sharpe": metrics["sharpe"] > 0,
                      "by_calendar_year": annual,
                      "excluded_data_sessions": [str(date.date()) for date in market.excluded_dates],
                      "backend": payload["protocol"]["backend"]}
            serialized = json.dumps(result, indent=2, allow_nan=False)
            daily.to_csv(root / "hidden_daily.csv", index=False)
            (root / "evaluation.json").write_text(serialized)
            with connection:
                connection.execute("UPDATE evaluations SET status='complete', result=? WHERE artifact=?", (serialized, key))
            return result
        except Exception as exc:
            with connection:
                connection.execute("UPDATE evaluations SET status='failed', result=? WHERE artifact=?", (str(exc), key))
            raise
    finally:
        connection.close()
