from __future__ import annotations

import asyncio
import errno
import os
import pickle
import stat
import time
from typing import TYPE_CHECKING, TypedDict

import pytest

import hyperfine
from hyperfine import (
    Benchmark,
    BenchmarkFailedError,
    BenchmarkTimeoutError,
    HyperfineError,
    HyperfineNotFoundError,
    HyperfineVersion,
    ParameterScan,
)
from tests.conftest import requires_hyperfine

if TYPE_CHECKING:
    from pathlib import Path


class _Fast(TypedDict):
    runs: int
    no_shell: bool


FAST: _Fast = {"runs": 2, "no_shell": True}
"""Cheap settings for real hyperfine runs: two timing runs and no intermediate shell."""


def _nonexistent(_name: str) -> str:
    return "/nonexistent/hyperfine"


def fake_binary(tmp_path: Path, script: str) -> Path:
    """Write an executable shell script standing in for hyperfine."""
    path = tmp_path / "fake-hyperfine"
    _ = path.write_text(f"#!/bin/sh\n{script}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


# --------------------------------------------------------------------------- binary lookup


def test_not_found_binary_has_helpful_message() -> None:
    with pytest.raises(HyperfineNotFoundError, match="HYPERFINE_BIN") as excinfo:
        _ = hyperfine.run("true", binary="definitely-not-hyperfine")
    assert excinfo.value.binary == "definitely-not-hyperfine"
    assert isinstance(excinfo.value, FileNotFoundError)


def test_env_var_selects_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = fake_binary(tmp_path, 'echo "hyperfine 9.8.7"')
    monkeypatch.setenv(hyperfine.ENV_VAR, str(fake))
    assert hyperfine.find_binary() == str(fake)
    assert hyperfine.version() == HyperfineVersion(9, 8, 7)
    assert hyperfine.find_binary("sh") != str(fake)


def test_version_parse_failure(tmp_path: Path) -> None:
    fake = fake_binary(tmp_path, "echo garbage")
    with pytest.raises(HyperfineError, match="could not determine hyperfine version"):
        _ = hyperfine.version(fake)


def test_spawn_failure_maps_to_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", _nonexistent)
    with pytest.raises(HyperfineNotFoundError, match="/nonexistent/hyperfine"):
        _ = hyperfine.version()
    with pytest.raises(HyperfineNotFoundError):
        _ = hyperfine.run("true")
    with pytest.raises(HyperfineNotFoundError):
        _ = asyncio.run(hyperfine.arun("true"))


# --------------------------------------------------------------------------- fake binary behaviour


def test_failure_carries_details(tmp_path: Path) -> None:
    fake = fake_binary(tmp_path, 'echo "  Warning: noise" >&2; echo "Error: boom" >&2; exit 3')
    with pytest.raises(BenchmarkFailedError, match="status 3: Error: boom") as excinfo:
        _ = hyperfine.run("true", binary=fake)
    error = excinfo.value
    assert error.returncode == 3
    assert "Warning: noise" in error.stderr
    assert error.argv[0] == str(fake)
    assert error.argv[-2:] == ("--", "true")


def test_failure_without_stderr(tmp_path: Path) -> None:
    fake = fake_binary(tmp_path, "exit 4")
    with pytest.raises(BenchmarkFailedError) as excinfo:
        _ = hyperfine.run("true", binary=fake)
    assert str(excinfo.value) == "hyperfine exited with status 4"
    assert excinfo.value.reason == ""


def test_failure_reason_falls_back_to_last_line(tmp_path: Path) -> None:
    fake = fake_binary(tmp_path, 'echo "error: clap says no" >&2; echo "Usage: x" >&2; exit 2')
    with pytest.raises(BenchmarkFailedError) as excinfo:
        _ = hyperfine.run("true", binary=fake)
    assert excinfo.value.reason == "error: clap says no"
    fake = fake_binary(tmp_path, 'echo "something odd" >&2; exit 2')
    with pytest.raises(BenchmarkFailedError, match="something odd"):
        _ = hyperfine.run("true", binary=fake)


def test_missing_export_is_reported(tmp_path: Path) -> None:
    fake = fake_binary(tmp_path, "exit 0")
    with pytest.raises(HyperfineError, match="did not write its JSON export"):
        _ = hyperfine.run("true", binary=fake)


def test_sync_timeout(tmp_path: Path) -> None:
    fake = fake_binary(tmp_path, "exec sleep 10")
    with pytest.raises(BenchmarkTimeoutError, match=r"0\.2 seconds") as excinfo:
        _ = hyperfine.run("true", binary=fake, timeout=0.2)
    assert excinfo.value.timeout == 0.2
    assert isinstance(excinfo.value, TimeoutError)


async def _slow_async(fake: Path, limit: float) -> None:
    _ = await hyperfine.arun("true", binary=fake, timeout=limit)


def test_async_timeout(tmp_path: Path) -> None:
    fake = fake_binary(tmp_path, "exec sleep 10")
    with pytest.raises(BenchmarkTimeoutError) as excinfo:
        asyncio.run(_slow_async(fake, 0.2))
    assert excinfo.value.argv[0] == str(fake)


@pytest.mark.asyncio
async def test_async_cancellation_kills_process(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    fake = fake_binary(tmp_path, f'echo $$ > "{pid_file}"; exec sleep 10')
    task = asyncio.create_task(hyperfine.arun("true", binary=fake))
    for _ in range(200):
        if pid_file.exists() and pid_file.read_text().strip():
            break
        await asyncio.sleep(0.01)
    _ = task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def _wait_for_pid(pid_file: Path) -> int:
    for _ in range(500):
        if pid_file.exists() and pid_file.read_text().strip():
            return int(pid_file.read_text())
        time.sleep(0.01)
    msg = f"{pid_file} was never written"
    raise AssertionError(msg)


def _assert_gone(pid: int) -> None:
    # The orphaned child is reaped by init shortly after being killed.
    for _ in range(500):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.01)
    msg = f"process {pid} is still running"
    raise AssertionError(msg)


def _spawning_fake(tmp_path: Path) -> tuple[Path, Path]:
    pid_file = tmp_path / "child.pid"
    fake = fake_binary(tmp_path, f'sleep 30 & echo $! > "{pid_file}"; wait')
    return fake, pid_file


def test_sync_timeout_kills_benchmarked_commands(tmp_path: Path) -> None:
    fake, pid_file = _spawning_fake(tmp_path)
    with pytest.raises(BenchmarkTimeoutError):
        _ = hyperfine.run("true", binary=fake, timeout=0.5)
    _assert_gone(_wait_for_pid(pid_file))


def test_async_timeout_kills_benchmarked_commands(tmp_path: Path) -> None:
    fake, pid_file = _spawning_fake(tmp_path)
    with pytest.raises(BenchmarkTimeoutError):
        asyncio.run(_slow_async(fake, 0.5))
    _assert_gone(_wait_for_pid(pid_file))


def test_interrupt_kills_benchmarked_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, pid_file = _spawning_fake(tmp_path)

    def interrupted(_self: object, timeout: float | None = None) -> tuple[bytes, bytes]:
        _ = _wait_for_pid(pid_file), timeout
        raise KeyboardInterrupt

    monkeypatch.setattr("subprocess.Popen.communicate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _ = hyperfine.run("true", binary=fake)
    _assert_gone(_wait_for_pid(pid_file))


def test_streamed_failure_mentions_terminal(tmp_path: Path) -> None:
    fake = fake_binary(tmp_path, "exit 5")
    with pytest.raises(BenchmarkFailedError, match="streamed to the terminal") as excinfo:
        _ = hyperfine.run("true", binary=fake, stream=True)
    assert excinfo.value.streamed
    assert excinfo.value.stderr == ""


def test_errors_survive_pickling() -> None:
    failed = BenchmarkFailedError(returncode=2, stderr="Error: x", argv=["h", "a"], streamed=True)
    clone: object = pickle.loads(pickle.dumps(failed))  # noqa: S301 - round-tripping our own data
    assert isinstance(clone, BenchmarkFailedError)
    assert (clone.returncode, clone.stderr, clone.argv, clone.streamed) == (
        2,
        "Error: x",
        ("h", "a"),
        True,
    )
    assert str(clone) == str(failed)
    timeout = BenchmarkTimeoutError(timeout=1.5, argv=["h"])
    timeout_clone: object = pickle.loads(pickle.dumps(timeout))  # noqa: S301 - round-tripping our own data
    assert isinstance(timeout_clone, BenchmarkTimeoutError)
    assert (timeout_clone.timeout, timeout_clone.argv, str(timeout_clone)) == (
        1.5,
        ("h",),
        str(timeout),
    )
    missing = HyperfineNotFoundError("nope")
    missing_clone: object = pickle.loads(pickle.dumps(missing))  # noqa: S301 - round-tripping our own data
    assert isinstance(missing_clone, HyperfineNotFoundError)
    assert (missing_clone.binary, str(missing_clone)) == ("nope", str(missing))


def test_not_found_error_sets_errno_and_filename() -> None:
    error = HyperfineNotFoundError("nope")
    assert error.errno == errno.ENOENT
    filename: object = error.filename
    assert filename == "nope"
    assert str(error).startswith("hyperfine executable not found: 'nope'.")


# --------------------------------------------------------------------------- real hyperfine


@requires_hyperfine
def test_version_of_real_binary() -> None:
    found = hyperfine.version()
    assert found >= (1, 0, 0)
    assert str(found).count(".") == 2


@requires_hyperfine
def test_run_two_commands() -> None:
    report = hyperfine.run("sleep 0.001", "sleep 0.02", **FAST)
    assert report.commands == ("sleep 0.001", "sleep 0.02")
    assert report.fastest.command == "sleep 0.001"
    assert report.slowest.command == "sleep 0.02"
    for result in report:
        assert result.runs == 2
        assert result.times is not None
        assert result.stddev is not None
        assert result.succeeded
    assert report.argv is not None
    assert "--export-json" in report.argv
    assert report.relative_to()[0].is_faster


@requires_hyperfine
def test_single_run_and_benchmark_reuse() -> None:
    bench = Benchmark("true", runs=1, no_shell=True, warmup=1)
    first, second = bench.run(), bench.run()
    assert first[0].stddev is None
    assert second[0].runs == 1


@requires_hyperfine
@pytest.mark.asyncio
async def test_async_parameter_scan() -> None:
    report = await hyperfine.arun(
        "sleep 0.00{n}",
        parameter_scan=ParameterScan("n", 1, 3, step=2),
        command_names="nap {n}",
        **FAST,
    )
    assert report.commands == ("nap 1", "nap 3")
    assert [result.parameters for result in report] == [{"n": "1"}, {"n": "3"}]


@requires_hyperfine
@pytest.mark.asyncio
async def test_async_runs_concurrently() -> None:
    reports = await asyncio.gather(
        hyperfine.arun("true", **FAST), hyperfine.arun("sleep 0.001", **FAST)
    )
    assert [report.commands for report in reports] == [("true",), ("sleep 0.001",)]


@requires_hyperfine
def test_parameter_lists_product() -> None:
    report = hyperfine.run("echo {x}{y}", parameter_lists={"x": ["a", "b"], "y": [1, 2]}, **FAST)
    assert [dict(result.parameters) for result in report] == [
        {"x": "a", "y": "1"},
        {"x": "b", "y": "1"},
        {"x": "a", "y": "2"},
        {"x": "b", "y": "2"},
    ]


@requires_hyperfine
def test_real_failure() -> None:
    with pytest.raises(BenchmarkFailedError, match="non-zero exit code") as excinfo:
        _ = hyperfine.run("false", **FAST)
    assert excinfo.value.returncode != 0


@requires_hyperfine
def test_ignore_failure_modes() -> None:
    report = hyperfine.run("false", ignore_failure=True, **FAST)
    assert report[0].exit_codes == (1, 1)
    assert not report[0].succeeded
    report = hyperfine.run("false", ignore_failure=[1], **FAST)
    assert report[0].exit_codes == (1, 1)


@requires_hyperfine
def test_reference_with_name() -> None:
    report = hyperfine.run("sleep 0.001", reference="true", reference_name="base", **FAST)
    assert report.commands == ("base", "sleep 0.001")
    assert report.reference == "base"
    assert report.summary().startswith("'base' ran")


@requires_hyperfine
def test_reference_is_the_comparison_baseline() -> None:
    report = hyperfine.run("sleep 0.001", reference="sleep 0.02", **FAST)
    assert report.reference == "sleep 0.02"
    assert report.summary().splitlines() == [
        "'sleep 0.02' ran",
        report.summary().splitlines()[1],
    ]
    assert "times slower than 'sleep 0.001'" in report.summary()
    assert report.relative_to()[0].result.command == "sleep 0.02"


@requires_hyperfine
def test_tiny_float_parameter_scan() -> None:
    report = hyperfine.run(
        "echo {x}", parameter_scan=ParameterScan("x", 1e-05, 0.10001, step=0.1), **FAST
    )
    assert [result.parameters["x"] for result in report] == ["0.00001", "0.10001"]


@requires_hyperfine
def test_real_timeout_leaves_no_benchmarked_process(tmp_path: Path) -> None:
    pid_file = tmp_path / "cmd.pid"
    with pytest.raises(BenchmarkTimeoutError):
        _ = hyperfine.run(f'echo $$ > "{pid_file}"; exec sleep 30', runs=1, timeout=1)
    _assert_gone(_wait_for_pid(pid_file))


@requires_hyperfine
def test_shell_commands_with_cwd_env_and_exports(tmp_path: Path) -> None:
    _ = (tmp_path / "marker").write_text("x")
    report = hyperfine.run(
        'test "$GREETING" = hello && test -f marker',
        runs=2,
        cwd=tmp_path,
        env={"GREETING": "hello"},
        setup="true",
        prepare="true",
        conclude="true",
        cleanup="true",
        export_json="kept.json",
        export_markdown=tmp_path / "out.md",
        time_unit="millisecond",
        sort="command",
    )
    assert report[0].succeeded
    assert hyperfine.BenchmarkReport.from_json(tmp_path / "kept.json") == report
    assert "Command" in (tmp_path / "out.md").read_text()


@requires_hyperfine
def test_stream_lets_output_through(capfd: pytest.CaptureFixture[str]) -> None:
    report = hyperfine.run("true", stream=True, **FAST)
    assert report.commands == ("true",)
    captured = capfd.readouterr()
    assert "Benchmark 1: true" in captured.out


@requires_hyperfine
def test_stdin_and_output_file(tmp_path: Path) -> None:
    source = tmp_path / "in.txt"
    _ = source.write_text("payload\n")
    sink = tmp_path / "out.txt"
    report = hyperfine.run("cat", stdin=source, output=sink, **FAST)
    assert report[0].succeeded
    assert "payload" in sink.read_text()
