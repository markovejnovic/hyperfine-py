from __future__ import annotations

import json
import math
import pickle
from typing import TYPE_CHECKING

import pytest

from hyperfine import (
    AmbiguousCommandError,
    BenchmarkReport,
    BenchmarkResult,
    ReportParseError,
    Speedup,
    format_duration,
)
from tests.conftest import DATA

if TYPE_CHECKING:
    from pathlib import Path


def make(command: str, mean: float, stddev: float | None = 0.001) -> BenchmarkResult:
    return BenchmarkResult(
        command=command,
        mean=mean,
        stddev=stddev,
        median=mean,
        user=0.0,
        system=0.0,
        min=mean,
        max=mean,
        times=(mean,),
        exit_codes=(0,),
    )


def test_parse_real_parameter_scan_export() -> None:
    report = BenchmarkReport.from_json(DATA / "parameter_scan.json")
    assert report.commands == ("a 1", "a 2")
    first = report[0]
    assert first.command == "a 1"
    assert first.mean == pytest.approx(0.004260625)
    assert first.stddev == pytest.approx(0.0006125366863617233)
    assert first.times is not None
    assert len(first.times) == first.runs == 3
    assert first.exit_codes == (0, 0, 0)
    assert first.memory_usage_byte == (1081344, 1081344, 1081344)
    assert first.parameters == {"n": "1"}
    assert first.succeeded
    assert report["a 2"].parameters == {"n": "2"}
    assert report.argv is None


def test_single_run_has_null_stddev() -> None:
    result = BenchmarkReport.from_json(DATA / "single_run.json")[0]
    assert result.stddev is None
    assert result.stddev_ms is None
    assert result.parameters == {}
    assert result.runs == 1


def test_ignored_failure_export() -> None:
    result = BenchmarkReport.from_json(DATA / "ignored_failure.json")[0]
    assert result.exit_codes == (137, 137)
    assert not result.succeeded


def test_minimal_document_with_absent_optional_fields() -> None:
    doc = {
        "results": [
            {
                "command": "x",
                "mean": 1,
                "median": 1.0,
                "user": 0.5,
                "system": 0.25,
                "min": 1.0,
                "max": 1.0,
                "exit_codes": [None],
            }
        ]
    }
    result = BenchmarkReport.from_json_string(json.dumps(doc))[0]
    assert result.mean == 1.0
    assert isinstance(result.mean, float)
    assert result.stddev is None
    assert result.times is None
    assert result.runs == 0
    assert result.memory_usage_byte is None
    assert result.exit_codes == (None,)
    assert "times" not in result.to_dict()
    assert "memory_usage_byte" not in result.to_dict()
    del doc["results"][0]["exit_codes"]
    assert BenchmarkReport.from_json_string(json.dumps(doc))[0].exit_codes == ()


def test_round_trip_to_json() -> None:
    text = (DATA / "parameter_scan.json").read_text()
    report = BenchmarkReport.from_json_string(text, argv=["hyperfine", "x"])
    assert report.argv == ("hyperfine", "x")
    expected: object = json.loads(text)
    assert report.to_dict() == expected
    assert BenchmarkReport.from_json_string(report.to_json()) == report
    single = BenchmarkReport.from_json(DATA / "single_run.json")
    expected = json.loads((DATA / "single_run.json").read_text())
    assert single.to_dict() == expected


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not json", "invalid JSON"),
        ("[]", "document: expected an object, got list"),
        ("{}", "missing required field 'results'"),
        ('{"results": {}}', "results: expected an array"),
        ('{"results": [1]}', r"results\[0\]: expected an object"),
        ('{"results": [{}]}', "missing required field 'command'"),
        ('{"results": [{"command": 1}]}', "command: expected a string"),
    ],
)
def test_parse_errors(text: str, message: str) -> None:
    with pytest.raises(ReportParseError, match=message):
        _ = BenchmarkReport.from_json_string(text)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mean", "1", "mean: expected a number"),
        ("mean", True, "mean: expected a number"),
        ("stddev", "x", "stddev: expected a number"),
        ("times", [1, "x"], r"times\[1\]: expected a number"),
        ("exit_codes", [1.5], r"exit_codes\[0\]: expected an integer"),
        ("memory_usage_byte", "x", "memory_usage_byte: expected an array"),
        ("parameters", {"n": 1}, "parameters.n: expected a string"),
    ],
)
def test_field_type_errors(field: str, value: object, message: str) -> None:
    raw: dict[str, object] = {**make("x", 1.0).to_dict(), field: value}
    with pytest.raises(ReportParseError, match=message):
        _ = BenchmarkResult.from_dict(raw)


def test_lookup_and_iteration() -> None:
    report = BenchmarkReport((make("a", 0.2), make("b", 0.1), make("c", 0.3)))
    assert len(report) == 3
    assert [result.command for result in report] == ["a", "b", "c"]
    assert report[-1].command == "c"
    assert "b" in report
    assert "z" not in report
    with pytest.raises(KeyError, match="known commands: 'a', 'b', 'c'"):
        _ = report["z"]
    assert report.fastest.command == "b"
    assert report.slowest.command == "c"
    assert [result.command for result in report.sorted()] == ["b", "a", "c"]
    assert [result.command for result in report.sorted(reverse=True)] == ["c", "a", "b"]


def test_empty_report() -> None:
    report = BenchmarkReport(())
    assert str(report) == "BenchmarkReport (no results)"
    with pytest.raises(ValueError, match="no results"):
        _ = report.fastest


def test_relative_to_and_summary() -> None:
    report = BenchmarkReport((make("slow", 0.3), make("fast", 0.1), make("mid", 0.2)))
    speedups = report.relative_to()
    assert [(s.result.command, s.baseline.command) for s in speedups] == [
        ("fast", "mid"),
        ("fast", "slow"),
    ]
    assert speedups[0].factor == pytest.approx(2.0)
    assert speedups[1].factor == pytest.approx(3.0)
    assert all(s.is_faster for s in speedups)
    assert report.summary() == (
        "'fast' ran\n  2.00 ± 0.02 times faster than 'mid'\n  3.00 ± 0.03 times faster than 'slow'"
    )
    by_name = report.relative_to("mid")
    assert [s.baseline.command for s in by_name] == ["fast", "slow"]
    assert str(by_name[0]) == "'mid' ran 2.00 ± 0.02 times slower than 'fast'"
    assert report.relative_to(0) == report.relative_to(report["slow"])


def test_speedup_edge_cases() -> None:
    zero = make("zero", 0.0, stddev=0.0)
    other = make("other", 0.1)
    assert zero.compare(other).factor == math.inf
    assert zero.compare(make("zero2", 0.0)).factor == 1.0
    assert other.compare(zero).factor == 0.0
    no_stddev = make("single", 0.2, stddev=None)
    speedup = Speedup.between(other, no_stddev)
    assert speedup.stddev is None
    assert str(speedup) == "'other' ran 2.00 times faster than 'single'"


def test_str_and_repr() -> None:
    report = BenchmarkReport((make("a", 0.0021), make("b", 1.5, stddev=None)))
    assert repr(report) == "BenchmarkReport('a': 2.1 ms, 'b': 1.500 s)"
    text = str(report)
    assert text.splitlines()[0].split() == ["Command", "Mean", "StdDev", "Min", "Max", "Relative"]
    assert "—" in text
    assert "'a' ran" in text
    assert str(BenchmarkReport((make("a", 0.0021),))).count("\n") == 2
    assert str(make("a", 0.0021)) == "a: 2.1 ms ± 1.0 ms [2.1 ms … 2.1 ms], 1 runs"


def test_millisecond_helpers() -> None:
    result = make("a", 0.25, stddev=0.5)
    assert result.mean_ms == 250.0
    assert result.median_ms == 250.0
    assert result.stddev_ms == 500.0


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(2.5, "2.500 s"), (0.0123, "12.3 ms"), (0.0000123, "12.3 µs"), (0.0, "0.0 µs")],
)
def test_format_duration(seconds: float, expected: str) -> None:
    assert format_duration(seconds) == expected


def test_from_json_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _ = BenchmarkReport.from_json(tmp_path / "missing.json")


def test_slicing_and_bad_keys() -> None:
    report = BenchmarkReport((make("a", 0.2), make("b", 0.1), make("c", 0.3)))
    assert [result.command for result in report[1:]] == ["b", "c"]
    assert report[::-1] == tuple(reversed(report.results))
    with pytest.raises(TypeError):
        _ = report[1.5]  # type: ignore[call-overload, misc]


def test_reference_anchors_comparisons() -> None:
    results = (make("ref", 0.3), make("a", 0.2), make("b", 0.1))
    report = BenchmarkReport(results, reference="ref")
    assert report == BenchmarkReport(results)
    assert report.summary().splitlines() == [
        "'ref' ran",
        "  3.00 ± 0.03 times slower than 'b'",
        "  1.50 ± 0.01 times slower than 'a'",
    ]
    assert [s.result.command for s in report.relative_to()] == ["ref", "ref"]
    relative = [line.split()[-1] for line in report.table().splitlines()[2:]]
    assert relative == ["1.00", "0.67", "0.33"]
    assert BenchmarkReport(results).summary().startswith("'b' ran")


def test_reference_must_be_a_result() -> None:
    with pytest.raises(ReportParseError, match="reference 'nope'"):
        _ = BenchmarkReport((make("a", 0.1),), reference="nope")
    report = BenchmarkReport.from_json(DATA / "parameter_scan.json", reference="a 2")
    assert report.reference == "a 2"
    assert report.summary().startswith("'a 2' ran")


def test_duplicate_command_lookup_is_ambiguous() -> None:
    report = BenchmarkReport((make("a", 0.2), make("b", 0.1), make("a", 0.3)))
    expected = r"2 results share the command 'a' \(at indices 0, 2\)"
    with pytest.raises(AmbiguousCommandError, match=expected) as excinfo:
        _ = report["a"]
    assert isinstance(excinfo.value, KeyError)
    assert excinfo.value.indices == (0, 2)
    assert excinfo.value.command == "a"
    assert [result.mean for result in report.find_all("a")] == [0.2, 0.3]
    assert report.find_all("z") == ()
    assert report["b"].mean == 0.1


def test_ambiguous_error_survives_pickling() -> None:
    error = AmbiguousCommandError(command="a", indices=[0, 2])
    clone: object = pickle.loads(pickle.dumps(error))  # noqa: S301 - round-tripping our own data
    assert isinstance(clone, AmbiguousCommandError)
    assert (clone.command, clone.indices, str(clone)) == ("a", (0, 2), str(error))


def test_reference_also_benchmarked_resolves_to_first() -> None:
    # hyperfine exports the reference first, then the commands (one of which repeats it).
    report = BenchmarkReport((make("a", 0.3), make("b", 0.1), make("a", 0.2)), reference="a")
    assert report.summary().startswith("'a' ran")
    speedups = report.relative_to()
    assert all(speedup.result is report.results[0] for speedup in speedups)
    assert {id(speedup.baseline) for speedup in speedups} == {id(r) for r in report.results[1:]}
