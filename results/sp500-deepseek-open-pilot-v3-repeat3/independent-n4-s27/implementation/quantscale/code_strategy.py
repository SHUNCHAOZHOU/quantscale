"""Open numerical strategy code executed in an OS-sandboxed child, never in the host."""
import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import selectors
import shutil
import struct
import subprocess
import sys
import tempfile
import time

import pandas as pd

ALLOWED_IMPORTS = frozenset({"numpy", "pandas", "math", "statistics", "functools", "itertools", "collections"})
BLOCKED_NAMES = frozenset({"open", "eval", "exec", "compile", "getattr", "setattr", "delattr",
                           "globals", "locals", "vars", "dir", "input", "breakpoint", "help", "__import__"})


def validate_source(source):
    if not isinstance(source, str) or not source.strip():
        raise ValueError("strategy_code must be a nonempty Python module")
    if len(source.encode()) > 128000:
        raise ValueError("Strategy source exceeds 128 KB")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError("Strategy syntax error on line %s: %s" % (exc.lineno, exc.msg)) from None
    entries = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "generate_weights"]
    if len(entries) != 1:
        raise ValueError("Define exactly one top-level generate_weights(history, universe) function")
    args = entries[0].args
    if len(args.posonlyargs + args.args) != 2 or args.vararg or args.kwarg or args.kwonlyargs or args.defaults:
        raise ValueError("generate_weights must accept exactly history and universe")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if isinstance(node, ast.ImportFrom) and (node.level or any(a.name == "*" for a in node.names)):
                raise ValueError("Relative imports and star imports are unavailable")
            if any(module.split(".")[0] not in ALLOWED_IMPORTS for module in modules):
                raise ValueError("Use only the numerical/statistical libraries in runtime_contract")
        if isinstance(node, ast.Name) and (node.id in BLOCKED_NAMES or node.id.startswith("__")):
            raise ValueError("Filesystem, dynamic execution and introspection builtins are unavailable")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Private/introspection attributes are unavailable")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {
                "fit", "fit_transform", "partial_fit", "train", "train_on_batch"}:
            raise ValueError("Training predictive models is outside the experiment protocol")
    return tree


@dataclass(frozen=True)
class CodeSpec:
    source: str

    def __post_init__(self):
        validate_source(self.source)

    @property
    def digest(self):
        canonical = ast.dump(validate_source(self.source), include_attributes=False)
        return hashlib.sha256(("python_ast_v1\n" + canonical).encode()).hexdigest()

    def to_dict(self):
        return {"kind": "python", "source": self.source}


def sandbox_profile(executable, worker, scratch, library_paths):
    def literal(path):
        return json.dumps(str(Path(path).resolve()))
    read_rules = ['(literal "/")', '(subpath "/System")', '(subpath "/usr/lib")', '(subpath "/usr/share")',
                  '(subpath "/private/preboot")',
                  '(subpath "/Library/Apple")', '(subpath "/private/var/db/timezone")',
                  '(literal "/dev/null")', '(literal "/dev/urandom")', '(literal "/dev/random")',
                  '(literal "/private/etc/localtime")', '(literal "/etc/localtime")',
                  '(literal %s)' % literal(executable), '(literal %s)' % literal(worker),
                  '(subpath %s)' % literal(scratch)]
    read_rules += ['(subpath %s)' % literal(path) for path in library_paths]
    return '\n'.join([
        '(version 1)', '(deny default)', '(deny network*)',
        '(allow process-exec (literal %s))' % literal(executable),
        '(allow sysctl-read)', '(allow mach-lookup)',
        '(allow file-read* %s)' % ' '.join(read_rules),
        '(allow file-write* (subpath %s) (literal "/dev/null"))' % literal(scratch),
    ])


class SandboxedStrategy:
    """Incremental point-in-time RPC: the child receives no future-price file or array."""
    def __init__(self, spec, config):
        self.spec, self.config = spec, config
        self.process = None
        self.scratch = None
        self.runtime_limits = {}

    def prepare(self, market):
        self.close()
        launcher = shutil.which("sandbox-exec") if sys.platform == "darwin" else None
        if not launcher:
            raise RuntimeError("Open code requires the macOS OS sandbox; no unsandboxed fallback is permitted")
        self.market = market
        self.last_index = -1
        self.started = time.monotonic()
        self.scratch = tempfile.TemporaryDirectory(prefix="quantscale-policy-")
        executable = Path(sys.executable).resolve()
        # Apple's framework launcher posix_spawns Python.app. Launch that binary
        # directly so the sandbox need not permit general child execution.
        framework_python = Path(sys.base_prefix) / "Resources/Python.app/Contents/MacOS/Python"
        if sys.platform == "darwin" and framework_python.is_file():
            executable = framework_python.resolve()
        worker = Path(__file__).with_name("strategy_worker.py").resolve()
        sites = sorted({str(Path(importlib.util.find_spec(name).origin).resolve().parent.parent)
                        for name in ("numpy", "pandas")})
        libraries = sites + [str(Path(os.__file__).resolve().parent), str(Path(sys.base_prefix).resolve())]
        profile = sandbox_profile(executable, worker, self.scratch.name, libraries)
        self.stderr = tempfile.TemporaryFile()
        child_env = {"HOME": self.scratch.name, "TMPDIR": self.scratch.name,
                     "PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
                     "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
        self.process = subprocess.Popen([launcher, "-p", profile, str(executable), "-I", "-B", str(worker),
                                         json.dumps(sites)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr, cwd=self.scratch.name, env=child_env)
        os.set_blocking(self.process.stdout.fileno(), False)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = b""
        try:
            ready = self._rpc({"kind": "init", "source": self.spec.source,
                               "history_days": self.config.code_history_days,
                               "cpu_seconds": self.config.code_backtest_timeout_seconds,
                               "memory_mb": self.config.code_memory_mb})
            self.runtime_limits = ready["limits"]
        except Exception:
            self.close()
            raise

    def _rpc(self, request):
        remaining = self.config.code_backtest_timeout_seconds - (time.monotonic() - self.started)
        allowance = min(self.config.code_call_timeout_seconds, remaining)
        if allowance <= 0:
            raise ValueError("Strategy backtest wall-time budget exhausted")
        payload = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            self.process.stdin.write(struct.pack("!Q", len(payload)) + payload)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            raise ValueError("Strategy sandbox worker terminated during request") from None
        deadline = time.monotonic() + allowance
        while b"\n" not in self.buffer:
            if time.monotonic() >= deadline:
                raise ValueError("Strategy decision exceeded its execution timeout")
            if not self.selector.select(max(0, deadline - time.monotonic())):
                raise ValueError("Strategy decision exceeded its execution timeout")
            block = os.read(self.process.stdout.fileno(), 65536)
            if not block:
                self.stderr.seek(0)
                detail = self.stderr.read(1000).decode(errors="replace")
                raise ValueError("Strategy sandbox worker terminated: " + detail)
            self.buffer += block
            if len(self.buffer) > 2000000:
                raise ValueError("Strategy worker response exceeds 2 MB")
        line, self.buffer = self.buffer.split(b"\n", 1)
        try:
            result = json.loads(line)
        except ValueError:
            raise ValueError("Strategy worker returned invalid JSON") from None
        if "error" in result:
            raise ValueError("Strategy execution failed: " + str(result["error"])[:2000])
        return result

    def weights_at(self, index, universe):
        start = self.last_index + 1
        if start == 0 and self.config.code_history_days is not None:
            start = max(0, index + 1 - self.config.code_history_days)
        block = {name: field.iloc[start:index + 1].to_numpy(copy=True)
                 for name, field in self.market.fields.items()}
        result = self._rpc({"kind": "decision", "fields": block,
                            "dates": self.market.dates[start:index + 1].to_numpy(),
                            "tickers": list(self.market.fields["open"].columns),
                            "universe": list(universe)})
        self.last_index = index
        pairs = result.get("weights")
        if not isinstance(pairs, list) or any(not isinstance(pair, list) or len(pair) != 2 for pair in pairs):
            raise ValueError("Worker weights must be ticker/weight pairs")
        return pd.Series([pair[1] for pair in pairs], index=[pair[0] for pair in pairs], dtype=float)

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=10)
            self.process.stdin.close()
            self.process.stdout.close()
            self.selector.close()
            self.stderr.close()
            self.process = None
        if self.scratch is not None:
            self.scratch.cleanup()
            self.scratch = None
