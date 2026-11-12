"""Loop a batched CPU factorization and capture native backtraces if one call hangs.

The batched dense LU of a stack of complex matrices goes wrong on some oneMKL builds once
``torch.set_num_threads`` has been called with more than one thread (see
``pgml.solver._dense_lapack``), and on a CI runner the same call has been seen never to
return. This probe repeats the call in that thread state for a fixed time. When one call
takes longer than ``--hang-after`` seconds, a watchdog attaches ``gdb`` to the process,
prints every native thread's backtrace and exits with status 3.

``--target torch`` makes the raw ``torch.linalg.solve`` call, the one the guard avoids in
that state; ``--target pgml`` goes through ``pgml.solver.solve_harmonic`` and so checks
that the guarded path returns.

Usage::

    pixi run -e cpu python tools/lapack_hang_probe.py --target torch --threads 4 \\
        --seconds 600

Attaching needs ``gdb`` and ptrace permission for a child process (the script asks for
it with ``prctl``; a container may also need ``kernel.yama.ptrace_scope=0``).
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import threading
import time

import torch

_PR_SET_PTRACER = 0x59616D61
_PR_SET_PTRACER_ANY = ctypes.c_ulong(-1)


def _allow_child_ptrace() -> None:
    """Let the watchdog's gdb child attach to this process under Yama scope 1."""
    try:
        ctypes.CDLL(None).prctl(_PR_SET_PTRACER, _PR_SET_PTRACER_ANY, 0, 0, 0)
    except (AttributeError, OSError, TypeError):
        pass


def _watch(state: dict, hang_after: float) -> None:
    while True:
        time.sleep(1.0)
        started = state["started"]
        if started is not None and time.perf_counter() - started > hang_after:
            print(
                f"\nHANG: call {state['calls'] + 1} has run {hang_after:.0f} s; "
                "native backtraces follow",
                flush=True,
            )
            traces = subprocess.run(
                [
                    "gdb",
                    "-p",
                    str(os.getpid()),
                    "-batch",
                    "-ex",
                    "thread apply all bt 30",
                ],
                capture_output=True,
                text=True,
            )
            print(traces.stdout, traces.stderr, sep="\n", flush=True)
            os._exit(3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target", choices=("torch", "pgml"), default="torch")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rows", type=int, default=600)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--hang-after", type=float, default=120.0)
    args = parser.parse_args()

    _allow_child_ptrace()
    torch.set_num_threads(args.threads)
    print(torch.__config__.parallel_info(), flush=True)

    generator = torch.Generator().manual_seed(0)
    n = args.rows
    y = torch.randn(
        args.batch, n, n, dtype=torch.complex128, generator=generator
    ) / n + torch.eye(n, dtype=torch.complex128)
    i = torch.randn(args.batch, n, dtype=torch.complex128, generator=generator)

    if args.target == "pgml":
        from pgml.solver import solve_harmonic

        def call():
            return solve_harmonic(y, i)
    else:

        def call():
            return torch.linalg.solve(y, i)

    state = {"started": None, "calls": 0}
    threading.Thread(target=_watch, args=(state, args.hang_after), daemon=True).start()

    raised = wrong = 0
    slowest = 0.0
    end = time.perf_counter() + args.seconds
    while time.perf_counter() < end:
        state["started"] = time.perf_counter()
        try:
            v = call()
        except RuntimeError:
            raised += 1
        else:
            residual = (torch.einsum("bij,bj->bi", y, v) - i).abs().amax()
            wrong += int(float(residual) > 1e-8)
        slowest = max(slowest, time.perf_counter() - state["started"])
        state["started"] = None
        state["calls"] += 1

    print(
        f"\ntarget={args.target} threads={torch.get_num_threads()} rows={n} "
        f"batch={args.batch} calls={state['calls']} raised={raised} wrong={wrong} "
        f"slowest={slowest:.2f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
