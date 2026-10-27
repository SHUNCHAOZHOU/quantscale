import argparse
from dataclasses import replace
import json
from pathlib import Path
import shlex
import sys

from .config import Config, Period
from .data import prepare_partitions, synthetic_data
from .evaluation import evaluate
from .research import CommandBackend, DemoBackend, TOPOLOGIES, run_research


def make_backend(args, seed):
    if args.backend_command:
        return CommandBackend(shlex.split(args.backend_command))
    if args.backend == "deepseek":
        from .deepseek_adapter import DeepSeekBackend
        return DeepSeekBackend()
    return DemoBackend(seed)


def main(argv=None):
    parser = argparse.ArgumentParser(description="QuantScale research and separate hidden evaluation")
    sub = parser.add_subparsers(dest="action", required=True)
    demo = sub.add_parser("demo", help="Offline synthetic-data end-to-end smoke experiment")
    demo.add_argument("--output", default="results/demo")
    demo.add_argument("--backtests", type=int, default=4)
    demo.add_argument("--topology", choices=TOPOLOGIES, default="single")
    demo.add_argument("--fleet-size", type=int, default=1)
    demo.add_argument("--seed", type=int, default=7)
    prepare = sub.add_parser("prepare", help="Operator: separate research and evaluator files")
    prepare.add_argument("--data", required=True)
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--output", required=True)
    research = sub.add_parser("research", help="Research only; refuses any hidden-period rows")
    research.add_argument("--data", required=True)
    research.add_argument("--config", required=True)
    research.add_argument("--output", required=True)
    research.add_argument("--topology", choices=TOPOLOGIES, default="single")
    research.add_argument("--fleet-size", type=int, default=1)
    research_backend = research.add_mutually_exclusive_group()
    research_backend.add_argument("--backend", choices=("demo", "deepseek"), default="deepseek")
    research_backend.add_argument("--backend-command", help="Trusted JSON adapter argv; no shell expansion")
    hidden = sub.add_parser("evaluate", help="Evaluator: run a frozen submission exactly once")
    hidden.add_argument("--frozen", required=True)
    hidden.add_argument("--data", required=True)
    hidden.add_argument("--output", required=True)
    scaling = sub.add_parser("scaling", help="Research-only matrix; each run gets a fresh backend")
    scaling.add_argument("--data", required=True)
    scaling.add_argument("--config", required=True)
    scaling.add_argument("--matrix", required=True)
    scaling.add_argument("--output", required=True)
    scaling_backend = scaling.add_mutually_exclusive_group()
    scaling_backend.add_argument("--backend", choices=("demo", "deepseek"), default="deepseek")
    scaling_backend.add_argument("--backend-command")
    args = parser.parse_args(argv)
    try:
        if args.action == "demo":
            root = Path(args.output)
            root.mkdir(parents=True, exist_ok=False)
            config = Config(train=Period("2018-01-01", "2018-06-30"),
                            validation=Period("2018-07-01", "2018-12-31"),
                            test=Period("2019-01-01", "2019-06-30"),
                            universe_size=48, backtests=args.backtests,
                            tokens=100000, max_turn_tokens=16000, seed=args.seed)
            (root / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
            source = root / "synthetic.csv"
            synthetic_data("2017-01-02", "2019-06-30", assets=48, seed=args.seed).to_csv(source, index=False)
            research_path, evaluator_path = prepare_partitions(source, root / "operator-data", config)
            run = run_research(research_path, config, root / "research", args.topology, args.fleet_size)
            result = evaluate(root / "research/frozen.json", evaluator_path, root / "evaluation")
            result["notice"] = "SYNTHETIC DATA + SCRIPTED SEARCH. Pipeline smoke test; no claim about LLM research or investability."
            result["process"] = run["process"]
            print(json.dumps(result, indent=2, allow_nan=False))
        elif args.action == "prepare":
            paths = prepare_partitions(args.data, args.output, Config.load(args.config))
            print(json.dumps({"research": str(paths[0]), "evaluator_only": str(paths[1])}, indent=2))
        elif args.action == "research":
            config = Config.load(args.config)
            backend = make_backend(args, config.seed)
            result = run_research(args.data, config, args.output, args.topology, args.fleet_size, backend)
            print(json.dumps(result, indent=2, allow_nan=False))
        elif args.action == "evaluate":
            print(json.dumps(evaluate(args.frozen, args.data, args.output), indent=2, allow_nan=False))
        else:
            config = Config.load(args.config)
            matrix = json.loads(Path(args.matrix).read_text())
            mode = matrix.get("budget_mode", "fixed-total")
            if mode not in ("fixed-total", "fixed-per-agent"):
                raise ValueError("Unknown budget mode")
            root = Path(args.output)
            root.mkdir(parents=True, exist_ok=False)
            rows = []
            for topology in matrix["topologies"]:
                for size in ([1] if topology == "single" else matrix["fleet_sizes"]):
                    for budget in matrix["backtest_budgets"]:
                        for seed in matrix["seeds"]:
                            settings = replace(config, backtests=budget * (size if mode == "fixed-per-agent" else 1),
                                               tokens=config.tokens * (size if mode == "fixed-per-agent" else 1), seed=seed)
                            name = "%s-n%d-b%d-s%d" % (topology, size, budget, seed)
                            print("Running " + name, file=sys.stderr, flush=True)
                            backend = make_backend(args, seed)
                            result = run_research(args.data, settings, root / name, topology, size, backend,
                                                  budget_mode=mode)
                            rows.append({"run": name, "topology": topology, "fleet_size": size,
                                         "backtest_budget": settings.backtests, "seed": seed,
                                         "validation_sharpe": result["winner"]["validation"]["sharpe"],
                                         **result["process"]})
                            import pandas as pd
                            pd.DataFrame(rows).to_csv(root / "research_summary.csv", index=False)
            print(json.dumps({"runs": len(rows), "output": str(root.resolve()),
                              "next": "Evaluate each frozen.json in a separate evaluator environment."}, indent=2))
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.exit(2, "quantscale: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
