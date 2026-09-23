# hyperfine-py

Typed, ergonomic Python interface for [hyperfine](https://github.com/sharkdp/hyperfine),
the command-line benchmarking tool.

- Zero runtime dependencies, Python 3.12+, ships `py.typed`
- Every hyperfine option available as a typed keyword argument, validated before anything runs
- Results parsed from hyperfine's `--export-json` into frozen dataclasses (no stdout scraping)
- Sync and `asyncio` APIs, a reusable immutable `Benchmark` config, and a clear exception hierarchy

## Installation

```sh
pip install hyperfine-py      # or: uv add hyperfine-py
```

You also need the `hyperfine` binary (`brew install hyperfine`, `cargo install hyperfine`,
`apt install hyperfine`, ...). It is looked up from the `binary=` argument, then the
`HYPERFINE_BIN` environment variable, then `PATH`.

## Quick start

```python
import hyperfine

report = hyperfine.run("sleep 0.01", "sleep 0.02", warmup=1, runs=10)

print(report)
# Command      Mean     StdDev  Min      Max      Relative
# -----------  -------  ------  -------  -------  --------
# sleep 0.01   13.4 ms  0.4 ms  12.9 ms  14.1 ms      1.00
# sleep 0.02   23.6 ms  0.5 ms  22.8 ms  24.3 ms      1.76
#
# 'sleep 0.01' ran
#   1.76 ± 0.07 times faster than 'sleep 0.02'

report.fastest.command  # 'sleep 0.01'
report["sleep 0.02"].mean  # seconds, as a float
report["sleep 0.02"].mean_ms  # milliseconds
report[0].times  # every individual run (tuple[float, ...] | None)
report[0].stddev  # None when there was only a single run
```

Looking up a command name that matches several results (the same command benchmarked
twice, or a `reference=` that is also one of the commands) raises
`AmbiguousCommandError`, a `KeyError`. Use `report.find_all(name)` or an index instead.

## Reusable configuration

`Benchmark` is an immutable, hashable, validated configuration. Build it once and run it as
often as you like:

```python
from hyperfine import Benchmark, ParameterScan

bench = Benchmark(
    "make -j {threads}",
    parameter_scan=ParameterScan("threads", 1, 8),
    prepare="make clean",
    warmup=1,
    min_runs=5,
)

bench.to_args()  # ('--warmup', '1', '--min-runs', '5', '--prepare', 'make clean', ...)
bench.argv()  # same, prefixed with the resolved hyperfine executable
report = bench.run()

faster = bench.with_options(runs=3)  # derive a modified copy (runs replaces min_runs/max_runs)
lean = bench.without("prepare", "warmup")  # reset options to their defaults
other = bench.with_commands("ninja -j {threads}")
```

## Async

```python
import asyncio
import hyperfine


async def main() -> None:
    report = await hyperfine.arun("gzip -k -f big.txt", "zstd -k -f big.txt", runs=5)
    print(report.summary())


asyncio.run(main())
```

Cancelling the awaiting task kills the hyperfine process.

## Options

All options are keyword-only and shared by `run()`, `arun()` and `Benchmark(...)`
(see `hyperfine.BenchmarkOptions` for the full typed list):

| Python                                   | hyperfine                                   |
| ---------------------------------------- | ------------------------------------------- |
| `warmup`, `min_runs`, `max_runs`, `runs` | `--warmup`, `--min-runs`, `--max-runs`, `--runs` |
| `setup`, `cleanup`                       | `--setup`, `--cleanup`                      |
| `prepare`, `conclude` (str or one per command) | `--prepare`, `--conclude`             |
| `parameter_scan=ParameterScan(name, start, stop, step=None)` | `--parameter-scan`, `--parameter-step-size` |
| `parameter_lists={"compiler": ["gcc", "clang"]}` | `--parameter-list` (repeated)       |
| `no_shell=True`                          | `-N` (run commands without a shell)         |
| `shell_command="bash --norc"` / `"default"` | `--shell`                                |
| `ignore_failure=True` / `ignore_failure=[1, 2]` | `--ignore-failure` / `--ignore-failure=1,2` |
| `command_names`, `reference`, `reference_name` | `--command-name`, `--reference`, `--reference-name` |
| `style`, `sort`, `time_unit` (enums or their string values) | `--style`, `--sort`, `--time-unit` |
| `stdin=Path(...)`                        | `--input`                                   |
| `output="pipe"` / `Output.PIPE` / `Path("out.log")` / one per command | `--output` (files must be `pathlib.Path`; plain strings are the keywords `null`, `pipe`, `inherit`) |
| `export_json`, `export_csv`, `export_markdown`, `export_asciidoc`, `export_orgmode` | `--export-*` |
| `extra_args=[...]`                       | raw arguments, inserted before the commands |
| `cwd`, `env` (merged over `os.environ`), `timeout`, `binary` | process settings        |
| `stream=True`                            | show hyperfine's live progress instead of capturing it |

By default hyperfine runs with `--style none` and its output is captured; pass `stream=True`
to watch the progress bars in your terminal. When streaming, hyperfine's error output goes to
the terminal too, so `BenchmarkFailedError.stderr` is empty.

`timeout=` (and cancelling an `arun()` task, or Ctrl-C) kills hyperfine together with the
commands it is benchmarking, so nothing keeps running in the background.

## Reports

```python
from hyperfine import BenchmarkReport

report = BenchmarkReport.from_json("results.json")  # load an existing --export-json file

for result in report:  # BenchmarkResult (frozen dataclass)
    print(result.command, result.mean, result.parameters, result.exit_codes)

report.sorted()  # fastest first
report[1:]  # slices give a tuple[BenchmarkResult, ...]
report.relative_to()  # tuple[Speedup, ...] from the reference (or fastest) to every other result
report.relative_to("baseline")  # ... or from a named result
print(report.summary())  # hyperfine's "X ran N ± M times faster than Y"
report.to_dict()  # hyperfine's JSON structure
report.to_json()
report.argv  # the command line that produced it
```

When a benchmark is run with `reference=...`, `report.reference` names that result and
`summary()`, `relative_to()` and the table's `Relative` column compare against it, just like
hyperfine's own output. JSON exports do not record the reference, so pass it when loading:
`BenchmarkReport.from_json("results.json", reference="baseline")`.

## Errors

```
HyperfineError
├── ValidationError          (also ValueError)        bad/conflicting options, raised before spawning
├── ReportParseError         (also ValueError)        malformed JSON export
├── HyperfineNotFoundError   (also FileNotFoundError) executable missing, with install hints
├── BenchmarkFailedError                              non-zero exit; .returncode, .stderr, .argv, .reason
└── BenchmarkTimeoutError    (also TimeoutError)      the `timeout` elapsed
```

```python
try:
    hyperfine.run("false")
except hyperfine.BenchmarkFailedError as exc:
    print(exc.returncode, exc.reason)
```

`hyperfine.version()` returns the installed version as a comparable `HyperfineVersion`
named tuple, e.g. `hyperfine.version() >= (1, 20, 0)`.

## Development

```sh
uv sync
uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pyright && uv run pytest -q
```

## License

MIT — see [LICENSE](LICENSE).
