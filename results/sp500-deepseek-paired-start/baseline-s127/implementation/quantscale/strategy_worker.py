"""Trusted sandbox child. Never import this module in the research host."""
import builtins
from contextlib import redirect_stdout
import io
import json
import os
import pickle
import resource
import struct
import sys


def main():
    for site in json.loads(sys.argv[1]):
        sys.path.append(site)
    import numpy as np
    import pandas as pd

    allowed_imports = {"numpy", "pandas", "math", "statistics", "functools", "itertools", "collections"}
    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name.split(".")[0] not in allowed_imports:
            raise ImportError("Import outside the numerical runtime contract")
        return builtins.__import__(name, globals, locals, fromlist, level)

    safe_names = ["abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int",
                  "isinstance", "issubclass", "len", "list", "map", "max", "min", "next", "object",
                  "pow", "print", "range", "reversed", "round", "set", "slice", "sorted", "str",
                  "sum", "tuple", "type", "zip", "Exception", "ValueError", "TypeError", "KeyError",
                  "RuntimeError", "ZeroDivisionError", "StopIteration", "property", "staticmethod", "classmethod",
                  "__build_class__"]
    safe_builtins = {name: getattr(builtins, name) for name in safe_names}
    safe_builtins["__import__"] = guarded_import
    inbound = sys.stdin.buffer

    def receive():
        header = inbound.read(8)
        if not header:
            return None
        size = struct.unpack("!Q", header)[0]
        if size > 256 * 1024 * 1024:
            raise ValueError("Parent history message exceeds 256 MB")
        payload = bytearray()
        while len(payload) < size:
            block = inbound.read(size - len(payload))
            if not block:
                raise EOFError("Truncated parent history")
            payload.extend(block)
        return pickle.loads(payload)

    def respond(value):
        sys.stdout.write(json.dumps(value, allow_nan=False) + "\n")
        sys.stdout.flush()

    init = receive()
    resource.setrlimit(resource.RLIMIT_CPU, (int(init["cpu_seconds"]), int(init["cpu_seconds"]) + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    # Best-effort heap cap; OS/NumPy mmap allocations need not obey RLIMIT_DATA.
    heap = int(init["memory_mb"]) * 1024 * 1024
    soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
    inherited = [limit for limit in (soft, hard) if limit != resource.RLIM_INFINITY]
    heap = min([heap] + inherited)
    heap_applied = True
    try:
        resource.setrlimit(resource.RLIMIT_DATA, (heap, hard))
    except (ValueError, OSError):
        # Some macOS Python/kernel combinations reject RLIMIT_DATA entirely.
        # Keep inherited limits and disclose this; wall/CPU limits still apply.
        heap_applied = False
    namespace = {"__builtins__": safe_builtins, "__name__": "strategy", "np": np, "pd": pd}
    try:
        with redirect_stdout(io.StringIO()):
            exec(compile(init["source"], "<submitted_strategy>", "exec"), namespace)
        policy = namespace["generate_weights"]
        respond({"ready": True, "limits": {"heap_limit_applied": heap_applied,
                                             "requested_heap_mb": init["memory_mb"],
                                             "cpu_seconds": init["cpu_seconds"]}})
    except Exception as exc:
        respond({"error": type(exc).__name__ + ": " + str(exc)[:2000]})
        return
    buffers, dates, length, capacity = {}, [], 0, 0
    while True:
        request = receive()
        if request is None:
            return
        try:
            new_rows = len(request["dates"])
            needed = length + new_rows
            if needed > capacity:
                capacity = max(1024, needed, capacity * 2)
                for name, values in request["fields"].items():
                    expanded = np.empty((capacity, values.shape[1]), dtype=values.dtype)
                    if name in buffers:
                        expanded[:length] = buffers[name][:length]
                    buffers[name] = expanded
            for name, values in request["fields"].items():
                buffers[name][length:needed] = values
            dates.extend(request["dates"])
            length = needed
            start = max(0, length - init["history_days"]) if init["history_days"] is not None else 0
            history = {name: pd.DataFrame(values[start:length].copy(),
                                          index=pd.DatetimeIndex(dates[start:]), columns=request["tickers"])
                       for name, values in buffers.items()}
            with redirect_stdout(io.StringIO()):
                weights = policy(history, pd.Index(request["universe"]))
            if not isinstance(weights, pd.Series):
                raise TypeError("generate_weights must return pandas.Series")
            respond({"weights": [[str(ticker), float(weight)] for ticker, weight in weights.items()]})
        except Exception as exc:
            respond({"error": type(exc).__name__ + ": " + str(exc)[:2000]})


if __name__ == "__main__":
    main()
