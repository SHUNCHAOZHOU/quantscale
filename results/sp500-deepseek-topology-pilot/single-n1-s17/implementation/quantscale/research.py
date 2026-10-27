"""Budgeted research with deterministic scheduling and explicit information flow.

An external JSON command can supply LLM decisions. The offline demo backend is
a seeded rule search, not an LLM and not scientific evidence about AI scaling.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .data import MarketData, fingerprint, load_data
from .strategies import RuleStrategy, StrategySpec

TOPOLOGIES = ("single", "independent", "centralized", "decentralized", "hybrid")
FAMILIES = ("momentum", "reversal", "low_volatility", "volume")


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class Budget:
    token_limit: int
    backtest_limit: int
    tokens: int = 0
    backtests: int = 0
    communication_tokens: int = 0

    def charge_tokens(self, amount, communication=False):
        if type(amount) is not int or amount < 0:
            raise ValueError("Backend token usage must be a nonnegative integer")
        self.tokens += amount
        if communication:
            self.communication_tokens += amount
        if self.tokens > self.token_limit:
            raise RuntimeError("Backend breached the authorized token limit; run aborted")

    def charge_backtest(self):
        if self.backtests >= self.backtest_limit:
            raise BudgetExhausted("Backtest budget exhausted")
        self.backtests += 1


class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.sequence = 0

    def write(self, kind, **payload):
        self.sequence += 1
        with self.path.open("a") as handle:
            handle.write(json.dumps({"sequence": self.sequence, "kind": kind, **payload},
                                    sort_keys=True, allow_nan=False) + "\n")


class DemoBackend:
    """Seeded, offline proposal generator; token accounting is an estimate."""
    name = "scripted-demo"
    token_accounting = "estimated UTF-8 bytes / 4; no model inference"

    def __init__(self, seed):
        self.seed = seed
        self.rngs = {}

    def invoke(self, request):
        if request["role"] == "manager":
            response = {"directions": {str(i): FAMILIES[(request["round"] + i) % len(FAMILIES)]
                                       for i in range(request["fleet_size"])}}
        else:
            agent = request["agent"]
            rng = self.rngs.setdefault(agent, np.random.default_rng(np.random.SeedSequence([self.seed, agent])))
            family = request.get("direction")
            if family not in FAMILIES:
                family = str(rng.choice(FAMILIES))
            seen = request.get("peer_results", [])
            if seen and rng.random() < 0.5:
                family = max(seen, key=lambda r: r["validation"]["sharpe"])["strategy"]["family"]
            spec = StrategySpec(family=family, lookback=int(rng.choice([5, 10, 20, 60, 120])),
                                skip=int(rng.choice([0, 1, 5])),
                                fraction=float(rng.choice([0.2, 0.3, 0.5])),
                                volatility_scale=bool(rng.integers(0, 2)))
            response = {"hypothesis": family + " with a %d-day horizon" % spec.lookback,
                        "strategy": spec.to_dict()}
        usage = (len(json.dumps(request).encode()) + len(json.dumps(response).encode()) + 3) // 4
        if usage > request["max_total_tokens"]:
            raise BudgetExhausted("Not enough remaining tokens for another turn")
        return response, usage


class CommandBackend:
    """Trusted provider adapter. JSON stdin -> JSON stdout; argv, never a shell.

    The adapter must enforce max_total_tokens at its provider, including input,
    output and reasoning, then return actual usage.total_tokens. See README.
    """
    name = "external-command"
    token_accounting = "adapter-reported actual usage; provider enforcement required"

    def __init__(self, command):
        if not command:
            raise ValueError("Empty backend command")
        self.command = command

    def invoke(self, request):
        completed = subprocess.run(self.command, input=json.dumps(request), text=True,
                                   capture_output=True, timeout=180, check=True)
        if len(completed.stdout) > 1000000:
            raise ValueError("Adapter response exceeds 1 MB")
        result = json.loads(completed.stdout)
        usage = result["usage"]["total_tokens"]
        if result.get("budget_exhausted"):
            if usage != 0:
                raise ValueError("Adapter budget refusal must not incur inference tokens")
            raise BudgetExhausted("Input plus minimum output would exceed remaining token allowance")
        response = result["response"]
        if "provenance" in result and isinstance(response, dict):
            response = {**response, "_provenance": result["provenance"]}
        return response, usage


def process_metrics(trials, budget, turns, communications):
    successful = [t for t in trials if "validation" in t]
    hashes = [t["strategy_id"] for t in trials]
    families = [t["strategy"]["family"] for t in trials]
    counts = pd.Series(families).value_counts(normalize=True) if families else pd.Series(dtype=float)
    entropy = float(-(counts * np.log(counts)).sum()) if len(counts) else 0.0
    agents = {t["agent"] for t in trials}
    return {"total_tokens": budget.tokens, "communication_turn_tokens_upper_bound": budget.communication_tokens,
            "total_backtests": budget.backtests, "agent_turns": turns,
            "communication_calls": communications, "hypotheses": len(trials),
            "successful_candidates": len(successful), "unique_strategy_variants": len(set(hashes)),
            "duplicate_variant_rate": 1 - len(set(hashes)) / len(hashes) if hashes else 0.0,
            "unique_families": len(set(families)), "family_entropy": entropy,
            "mean_trials_per_active_researcher": len(trials) / len(agents) if agents else 0.0}


def run_research(data_path, config, output, topology="single", fleet_size=1, backend=None,
                 budget_mode="fixed-total"):
    if topology not in TOPOLOGIES or type(fleet_size) is not int or fleet_size < 1:
        raise ValueError("Invalid topology or fleet size")
    if topology == "single" and fleet_size != 1:
        raise ValueError("single requires fleet_size=1")
    df = load_data(data_path, config)
    if (df.date > pd.Timestamp(config.validation.end)).any():
        raise ValueError("Research input contains post-validation data; use prepare first")
    for period in (config.train, config.validation):
        if df[(df.date >= pd.Timestamp(period.start)) & (df.date <= pd.Timestamp(period.end))].date.nunique() < 3:
            raise ValueError("Insufficient train or validation data")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    journal = Journal(root / "trajectory.jsonl")
    backend = backend or DemoBackend(config.seed)
    budget = Budget(config.tokens, config.backtests)
    market = MarketData(df, config)
    protocol = {"schema_version": 1, "config": config.to_dict(), "topology": topology,
                "fleet_size": fleet_size, "manager_count": int(topology in ("centralized", "hybrid")),
                "research_data_hash": fingerprint(df), "backend": backend.name,
                "token_accounting": backend.token_accounting,
                "selection": "highest net validation Sharpe, SHA256 ascending tie-break",
                "execution": "close t -> next open -> following open; each split starts flat",
                "budget_mode": budget_mode, "seed": config.seed}
    protocol["excluded_data_sessions"] = [str(date.date()) for date in market.excluded_dates]
    if hasattr(backend, "configuration"):
        protocol["backend_configuration"] = backend.configuration
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2))
    journal.write("protocol", **protocol)
    train = df[(df.date >= pd.Timestamp(config.train.start)) & (df.date <= pd.Timestamp(config.train.end))]
    inspection = {"train_rows": len(train), "train_assets": train.ticker.nunique(),
                  "train_start": str(train.date.min().date()), "train_end": str(train.date.max().date()),
                  "fields": list(df.columns),
                  "median_daily_volume": float(train.volume.median())}
    histories = [[] for _ in range(fleet_size)]
    trials = []
    turns = communications = round_index = 0
    # Fixed-total independent allocation differs by at most one call.
    quotas = [config.backtests // fleet_size + int(i < config.backtests % fleet_size) for i in range(fleet_size)]
    attempts = [0] * fleet_size
    metric_cache = {}

    def invoke(role, payload, communication=False):
        nonlocal turns, communications
        remaining = budget.token_limit - budget.tokens
        if remaining <= 0:
            raise BudgetExhausted("Token budget exhausted")
        request = {"role": role, "round": round_index, "fleet_size": fleet_size,
                   "seed": config.seed,
                   "max_total_tokens": min(config.max_turn_tokens, remaining),
                   "instructions": "Propose only rule-based StrategySpec JSON; no training. "
                                   "Optimize cost-adjusted validation Sharpe. Use only supplied research information.",
                   "strategy_schema": {"family": list(FAMILIES), "lookback": "integer 2..252",
                                       "skip": "integer 0..20", "fraction": "number (0,.5]",
                                       "volatility_scale": "boolean"}, **payload}
        journal.write("request", request=request)
        response, usage = backend.invoke(request)
        budget.charge_tokens(usage, communication)
        turns += 1
        communications += int(communication)
        journal.write("response", role=role, response=response, tokens=usage,
                      communication=communication)
        if usage > request["max_total_tokens"]:
            raise RuntimeError("Adapter exceeded per-turn token cap")
        return response

    try:
        while budget.backtests < budget.backtest_limit:
            # Peer exchange is synchronous: everyone sees the previous round snapshot.
            snapshot = []
            for history in histories:
                latest_valid = next((trial for trial in reversed(history) if "validation" in trial), None)
                if latest_valid is not None:
                    snapshot.append(latest_valid)
            directions = {}
            if topology in ("centralized", "hybrid"):
                directions = invoke("manager", {"worker_results": snapshot}, True)["directions"]
                if (not isinstance(directions, dict)
                        or set(directions) != {str(i) for i in range(fleet_size)}
                        or any(value not in FAMILIES for value in directions.values())):
                    raise ValueError("Manager must return one valid family direction per worker")
                journal.write("manager_broadcast", directions=directions)
            progressed = False
            for agent in range(fleet_size):
                if budget.backtests >= budget.backtest_limit:
                    break
                if topology in ("single", "independent") and attempts[agent] >= quotas[agent]:
                    continue
                peers = [t for t in snapshot if t["agent"] != agent] if topology in ("decentralized", "hybrid") else []
                proposal = invoke("researcher", {"agent": agent, "inspection": inspection,
                                                "own_results": histories[agent][-3:],
                                                "peer_results": peers,
                                                "direction": directions.get(str(agent))}, bool(peers))
                # Count every requested trial, including malformed/failed strategies.
                budget.charge_backtest()
                attempts[agent] += 1
                progressed = True
                trial = {"agent": agent, "round": round_index,
                         "hypothesis": str(proposal.get("hypothesis", ""))[:4000]}
                try:
                    spec = StrategySpec(**proposal["strategy"])
                    if spec.lookback + spec.skip + 1 > config.lookback_days:
                        raise ValueError("Strategy requires more history than lookback_days permits")
                    trial.update(strategy=spec.to_dict(), strategy_id=spec.digest)
                    trial["cached_backtest"] = spec.digest in metric_cache
                    if spec.digest not in metric_cache:
                        policy = RuleStrategy(spec, config)
                        train_metrics, _ = run_backtest(policy, market, config.train)
                        validation_metrics, _ = run_backtest(policy, market, config.validation)
                        metric_cache[spec.digest] = (train_metrics, validation_metrics)
                    trial["train"], trial["validation"] = metric_cache[spec.digest]
                except (ValueError, KeyError, TypeError, OverflowError) as exc:
                    trial["error"] = str(exc)
                histories[agent].append(trial)
                trials.append(trial)
                journal.write("trial", **trial)
                score = trial.get("validation", {}).get("sharpe")
                print("%s seed=%d trial=%d/%d worker=%d validation=%s" %
                      (topology, config.seed, budget.backtests, config.backtests, agent,
                       "%.4f" % score if score is not None else "FAILED"), file=sys.stderr, flush=True)
            round_index += 1
            if not progressed:
                break
    except BudgetExhausted as exc:
        journal.write("budget_exhausted", reason=str(exc))
    except Exception as exc:
        journal.write("aborted", error=str(exc), tokens=budget.tokens, backtests=budget.backtests)
        raise

    # Invalid specs have no strategy metadata; exclude them from diversity counters.
    metrics = process_metrics([t for t in trials if "strategy_id" in t], budget, turns, communications)
    metrics["hypotheses"] = len(trials)
    (root / "process.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    valid = [t for t in trials if "validation" in t]
    if not valid:
        raise RuntimeError("No valid strategy completed; inspect trajectory.jsonl or raise token budget")
    winner = sorted(valid, key=lambda t: (-t["validation"]["sharpe"], t["strategy_id"]))[0]
    (root / "selection.json").write_text(json.dumps(winner, indent=2, allow_nan=False))
    # Import here to keep the research loop unaware of evaluator inputs.
    from .evaluation import freeze
    artifact = freeze(root, protocol, winner)
    journal.write("frozen", artifact_sha256=artifact["artifact_sha256"])
    return {"output": str(root.resolve()), "winner": winner, "process": metrics,
            "artifact_sha256": artifact["artifact_sha256"]}
