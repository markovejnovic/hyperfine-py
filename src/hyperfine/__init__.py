"""Run the `hyperfine <https://github.com/sharkdp/hyperfine>`_ benchmarking tool from Python.

Quick start::

    import hyperfine

    report = hyperfine.run("sleep 0.01", "sleep 0.02", runs=5)
    print(report)  # table + "X ran N times faster than Y" summary
    print(report.fastest.command, report["sleep 0.02"].mean)

Reusable configuration::

    bench = hyperfine.Benchmark("make -j {n}", parameter_scan=hyperfine.ParameterScan("n", 1, 8))
    print(bench.to_args())
    report = bench.run()
"""

from __future__ import annotations

from hyperfine._benchmark import Benchmark, arun, run
from hyperfine._errors import (
    AmbiguousCommandError,
    BenchmarkFailedError,
    BenchmarkTimeoutError,
    HyperfineError,
    HyperfineNotFoundError,
    ReportParseError,
    ValidationError,
)
from hyperfine._options import (
    BenchmarkOptions,
    OptionName,
    Output,
    OutputTarget,
    ParameterScan,
    ParameterValue,
    SortLike,
    SortOrder,
    StrPath,
    Style,
    StyleLike,
    TimeUnit,
    TimeUnitLike,
)
from hyperfine._report import BenchmarkReport, BenchmarkResult, JSONValue, Speedup, format_duration
from hyperfine._runner import ENV_VAR, HyperfineVersion, find_binary, version

__all__ = [
    "ENV_VAR",
    "AmbiguousCommandError",
    "Benchmark",
    "BenchmarkFailedError",
    "BenchmarkOptions",
    "BenchmarkReport",
    "BenchmarkResult",
    "BenchmarkTimeoutError",
    "HyperfineError",
    "HyperfineNotFoundError",
    "HyperfineVersion",
    "JSONValue",
    "OptionName",
    "Output",
    "OutputTarget",
    "ParameterScan",
    "ParameterValue",
    "ReportParseError",
    "SortLike",
    "SortOrder",
    "Speedup",
    "StrPath",
    "Style",
    "StyleLike",
    "TimeUnit",
    "TimeUnitLike",
    "ValidationError",
    "arun",
    "find_binary",
    "format_duration",
    "run",
    "version",
]
