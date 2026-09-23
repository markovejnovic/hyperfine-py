from __future__ import annotations

import copy
import dataclasses
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, get_args

import pytest

import hyperfine
from hyperfine import (
    Benchmark,
    BenchmarkOptions,
    Output,
    ParameterScan,
    SortOrder,
    Style,
    TimeUnit,
    ValidationError,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from dataclasses import Field


def test_minimal_args() -> None:
    assert Benchmark("true").to_args() == ("--style", "none", "--", "true")


def test_every_option_maps_to_argv(tmp_path: Path) -> None:
    bench = Benchmark(
        "cmd {n}",
        "other {n}",
        warmup=2,
        min_runs=3,
        max_runs=9,
        setup="make",
        prepare="sync",
        conclude=["c1", "c2"],
        cleanup="rm -rf out",
        parameter_scan=ParameterScan("n", 0.5, 1.5, step=0.5),
        shell_command="bash --norc",
        ignore_failure=[2, 1, 2],
        command_names=["a {n}", "b {n}"],
        reference="baseline",
        reference_name="base",
        style="basic",
        sort=SortOrder.MEAN_TIME,
        time_unit="millisecond",
        stdin=tmp_path / "in.txt",
        output=[Output.PIPE, "null", tmp_path / "out.log"],
        export_json=tmp_path / "r.json",
        export_csv=tmp_path / "r.csv",
        export_markdown=tmp_path / "r.md",
        export_asciidoc=tmp_path / "r.adoc",
        export_orgmode=tmp_path / "r.org",
        extra_args=["--show-output"],
    )
    assert bench.to_args() == (
        "--warmup", "2",
        "--min-runs", "3",
        "--max-runs", "9",
        "--setup", "make",
        "--prepare", "sync",
        "--conclude", "c1",
        "--conclude", "c2",
        "--cleanup", "rm -rf out",
        "--parameter-scan", "n", "0.5", "1.5",
        "--parameter-step-size", "0.5",
        "--shell", "bash --norc",
        "--ignore-failure=1,2",
        "--command-name", "a {n}",
        "--command-name", "b {n}",
        "--reference", "baseline",
        "--reference-name", "base",
        "--style", "basic",
        "--sort", "mean-time",
        "--time-unit", "millisecond",
        "--input", str(tmp_path / "in.txt"),
        "--output", "pipe",
        "--output", "null",
        "--output", str(tmp_path / "out.log"),
        "--export-json", str(tmp_path / "r.json"),
        "--export-csv", str(tmp_path / "r.csv"),
        "--export-markdown", str(tmp_path / "r.md"),
        "--export-asciidoc", str(tmp_path / "r.adoc"),
        "--export-orgmode", str(tmp_path / "r.org"),
        "--show-output",
        "--",
        "cmd {n}",
        "other {n}",
    )  # fmt: skip


def test_runs_and_parameter_lists() -> None:
    bench = Benchmark(
        "{compiler} -O{opt} main.c",
        runs=4,
        parameter_lists={"compiler": ["gcc", "clang"], "opt": [0, 2]},
        ignore_failure=True,
        no_shell=True,
    )
    assert bench.to_args() == (
        "--runs", "4",
        "--parameter-list", "compiler", "gcc,clang",
        "--parameter-list", "opt", "0,2",
        "-N",
        "--ignore-failure",
        "--style", "none",
        "--", "{compiler} -O{opt} main.c",
    )  # fmt: skip


def test_integer_scan_without_step() -> None:
    assert ParameterScan("threads", 1, 8).to_args() == ("--parameter-scan", "threads", "1", "8")


def test_stream_leaves_style_to_hyperfine() -> None:
    assert Benchmark("true", stream=True).to_args() == ("--", "true")
    assert Benchmark("true", stream=True, style=Style.FULL).to_args() == (
        "--style",
        "full",
        "--",
        "true",
    )


def test_export_json_override(tmp_path: Path) -> None:
    args = Benchmark("true", export_json="a.json").to_args(export_json=tmp_path / "b.json")
    assert args[-4:] == ("--export-json", str(tmp_path / "b.json"), "--", "true")


def test_enums_accept_plain_strings() -> None:
    bench = Benchmark("true", style="none", sort="command", time_unit="second")
    assert bench.style is Style.NONE
    assert bench.sort is SortOrder.COMMAND
    assert bench.time_unit is TimeUnit.SECOND


def test_single_string_options_are_normalised() -> None:
    bench = Benchmark("a", "b", prepare="p", command_names="first", output=Output.INHERIT)
    assert bench.prepare == ("p",)
    assert bench.command_names == ("first",)
    assert bench.output == (Output.INHERIT,)


def test_benchmark_is_immutable_and_hashable() -> None:
    bench = Benchmark("true", runs=2)
    with pytest.raises(AttributeError):
        bench.runs = 3  # type: ignore[misc]
    assert hash(bench) == hash(Benchmark("true", runs=2))
    assert bench == Benchmark("true", runs=2)


def test_options_round_trip() -> None:
    bench = Benchmark(
        "a {x}",
        warmup=1,
        parameter_lists={"x": ["1", "2"]},
        env={"FOO": "bar"},
        ignore_failure=[3],
        stream=True,
        cwd=".",
    )
    assert bench.options() == {
        "cwd": Path(),
        "env": {"FOO": "bar"},
        "ignore_failure": (3,),
        "parameter_lists": {"x": ("1", "2")},
        "stream": True,
        "warmup": 1,
    }
    assert Benchmark(*bench.commands, **bench.options()) == bench


def test_with_options_and_with_commands() -> None:
    bench = Benchmark("a", runs=3)
    changed = bench.with_options(runs=5, warmup=1)
    assert (changed.runs, changed.warmup, changed.commands) == (5, 1, ("a",))
    assert bench.runs == 3
    assert bench.with_commands("b", "c").commands == ("b", "c")
    assert bench.with_options(min_runs=2) == Benchmark("a", min_runs=2)
    with pytest.raises(ValidationError):
        _ = bench.with_options(min_runs=4, max_runs=2)


def test_with_options_switches_run_count_mode() -> None:
    # The README example: a bounded benchmark turned into an exact one, and back.
    bench = Benchmark("a", min_runs=5, max_runs=9, warmup=1)
    exact = bench.with_options(runs=3)
    assert (exact.runs, exact.min_runs, exact.max_runs, exact.warmup) == (3, None, None, 1)
    bounded = exact.with_options(max_runs=4)
    assert (bounded.runs, bounded.min_runs, bounded.max_runs) == (None, None, 4)


def test_with_options_switches_shell_mode() -> None:
    bench = Benchmark("a", no_shell=True)
    assert bench.with_options(shell_command="zsh").to_args()[:2] == ("--shell", "zsh")
    assert "-N" in bench.with_options(shell_command="zsh").with_options(no_shell=True).to_args()


def test_without_resets_options() -> None:
    bench = Benchmark("a", runs=3, warmup=1, env={"A": "b"})
    assert bench.without("warmup", "env") == Benchmark("a", runs=3)
    assert bench.without("setup") == bench
    with pytest.raises(ValidationError, match="unknown option"):
        _ = bench.without("bogus")  # type: ignore[arg-type]


def test_option_name_literal_matches_options() -> None:
    alias: object = hyperfine.OptionName.__value__
    names: tuple[object, ...] = get_args(alias)
    assert set(names) == set(BenchmarkOptions.__optional_keys__)


class _Replace(Protocol):
    def __call__(self, obj: Benchmark, /, **changes: int) -> Benchmark: ...


@pytest.mark.skipif(sys.version_info < (3, 13), reason="copy.replace is new in 3.13")
def test_copy_replace_uses_with_options() -> None:
    bench = Benchmark("a", min_runs=2)
    replace: _Replace = getattr(copy, "replace")  # noqa: B009 - copy.replace is 3.13+
    assert replace(bench, runs=4) == Benchmark("a", runs=4)
    assert bench.__replace__(warmup=1) == Benchmark("a", min_runs=2, warmup=1)


def test_dataclasses_replace_is_not_supported() -> None:
    with pytest.raises(ValidationError):
        _ = dataclasses.replace(Benchmark("a"), runs=2)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.1, "0.1"), (1e-05, "0.00001"), (1e16, "10000000000000000"), (2.5e-7, "0.00000025")],
)
def test_parameter_scan_never_uses_exponent_notation(value: float, expected: str) -> None:
    args = ParameterScan("x", 0, value, step=value).to_args()
    assert args == ("--parameter-scan", "x", "0", expected, "--parameter-step-size", expected)


@pytest.mark.parametrize("name", ["null", "pipe", "inherit"])
def test_relative_paths_named_like_keywords_stay_files(name: str) -> None:
    args = Benchmark("cat", stdin=Path(name), output=Path(name)).to_args()
    assert args[args.index("--input") + 1] == f"./{name}"
    assert args[args.index("--output") + 1] == f"./{name}"
    assert Benchmark("cat", output=Output(name)).to_args()[2:4] == ("--output", name)


def test_other_paths_are_passed_unchanged(tmp_path: Path) -> None:
    absolute = tmp_path / "pipe"
    args = Benchmark("cat", stdin=Path("in.txt"), output=absolute).to_args()
    assert args[2:6] == ("--input", "in.txt", "--output", str(absolute))


def test_repr_is_constructor_like() -> None:
    assert repr(Benchmark("a", "b", runs=3)) == "Benchmark('a', 'b', runs=3)"


def test_benchmark_fields_match_options() -> None:
    # Guards against the dataclass fields and the BenchmarkOptions TypedDict drifting apart.
    names = {field.name for field in cast("tuple[Field[object], ...]", fields(Benchmark))}
    assert names - {"commands"} == set(BenchmarkOptions.__optional_keys__)


def test_argv_includes_resolved_binary() -> None:
    argv = Benchmark("true", binary="sh").argv()
    assert Path(argv[0]).name == "sh"
    assert argv[1:] == ("--style", "none", "--", "true")


@pytest.mark.parametrize(
    ("commands", "options", "message"),
    [
        ((), {}, "at least one command"),
        (("",), {}, "must not be empty"),
        (("true",), {"runs": 0}, "runs: must be >= 1"),
        (("true",), {"warmup": -1}, "warmup: must be >= 0"),
        (("true",), {"runs": True}, "expected an integer"),
        (("true",), {"runs": 2, "min_runs": 3}, "runs cannot be combined"),
        (("true",), {"runs": 2, "max_runs": 3}, "runs cannot be combined"),
        (("true",), {"min_runs": 5, "max_runs": 3}, "must not exceed"),
        (("a", "b"), {"prepare": ["p1", "p2", "p3"]}, "prepare must be given once"),
        (("a", "b"), {"conclude": ["c1", "c2"], "reference": "r"}, "3 including"),
        (("a",), {"output": [Output.NULL, Output.PIPE]}, "output must be given once"),
        (("a",), {"command_names": ["x", "y"]}, "command_names"),
        (("a",), {"prepare": []}, "empty sequence"),
        (("a",), {"output": []}, "empty sequence"),
        (("a",), {"output": "file.txt"}, "pathlib.Path"),
        (("a",), {"extra_args": "--foo"}, "not a single string"),
        (("a",), {"reference_name": "r"}, "reference_name requires reference"),
        (("a",), {"timeout": 0}, "timeout"),
        (("a",), {"style": "fancy"}, "expected one of"),
        (("a",), {"ignore_failure": []}, "non-empty collection"),
        (("a",), {"ignore_failure": ["1"]}, "must be integers"),
        (("a",), {"env": {"A": 1}}, "env"),
        (("a",), {"shell_command": " "}, "shell_command: must not be empty"),
        (("a",), {"shell_command": "bash", "no_shell": True}, "no_shell cannot be combined"),
        (("a",), {"setup": 3}, "expected a string"),
        (("a {x}",), {"parameter_lists": {"x": []}}, "at least one value"),
        (("a {x}",), {"parameter_lists": {"x": "abc"}}, "not a string"),
        (("a {x}",), {"parameter_lists": {"x": ["a,b"]}}, "must not contain ','"),
        (("a {x}",), {"parameter_lists": {"{x}": ["a"]}}, "invalid parameter name"),
        (
            ("a {x}",),
            {"parameter_lists": {"x": ["1"]}, "parameter_scan": ParameterScan("y", 1, 2)},
            "cannot be combined with parameter_lists",
        ),
    ],
)
def test_validation_errors(
    commands: tuple[str, ...], options: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message) as excinfo:
        _ = Benchmark(*commands, **options)  # type: ignore[arg-type]
    assert isinstance(excinfo.value, ValueError)
    assert isinstance(excinfo.value, hyperfine.HyperfineError)


def test_parameters_skip_count_checks() -> None:
    bench = Benchmark("sleep {n}", parameter_scan=ParameterScan("n", 1, 3), prepare=["a", "b", "c"])
    assert bench.prepare == ("a", "b", "c")


@pytest.mark.parametrize(
    ("scan", "message"),
    [
        (lambda: ParameterScan("n", 3, 1), "must be >= start"),
        (lambda: ParameterScan("n", 0.1, 0.3), "step is required"),
        (lambda: ParameterScan("n", 1, 3, step=0), "must be positive"),
        (lambda: ParameterScan("", 1, 3), "invalid parameter name"),
        (lambda: ParameterScan("n", 0, math.inf, step=1), "must be finite"),
        (lambda: ParameterScan("n", 0, 1, step=math.nan), "must be finite"),
    ],
)
def test_parameter_scan_validation(scan: Callable[[], ParameterScan], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _ = scan()
