"""Locating and executing the hyperfine binary."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import signal
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, override

from hyperfine._errors import (
    BenchmarkFailedError,
    BenchmarkTimeoutError,
    HyperfineError,
    HyperfineNotFoundError,
)
from hyperfine._report import BenchmarkReport

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from hyperfine._options import StrPath

__all__ = ["ENV_VAR", "HyperfineVersion", "Invocation", "find_binary", "version"]

ENV_VAR = "HYPERFINE_BIN"
"""Environment variable consulted for the hyperfine executable."""

_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


class HyperfineVersion(NamedTuple):
    """A parsed ``hyperfine --version``, comparable as a tuple.

    Attributes:
        major: Major version.
        minor: Minor version.
        patch: Patch version.
    """

    major: int
    minor: int
    patch: int

    @override
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def find_binary(binary: StrPath | None = None) -> str:
    """Resolve the hyperfine executable.

    The lookup order is the ``binary`` argument, the ``HYPERFINE_BIN`` environment
    variable, and finally ``hyperfine`` on ``PATH``.

    Args:
        binary: Explicit executable name or path.

    Returns:
        The absolute path of the executable.

    Raises:
        HyperfineNotFoundError: If the executable cannot be found.
    """
    candidate = os.fspath(binary) if binary is not None else os.environ.get(ENV_VAR) or "hyperfine"
    resolved = shutil.which(candidate)
    if resolved is None:
        raise HyperfineNotFoundError(candidate)
    return resolved


def version(binary: StrPath | None = None) -> HyperfineVersion:
    """Return the version of the hyperfine executable.

    Args:
        binary: Explicit executable name or path; see :func:`find_binary`.

    Returns:
        The parsed version.

    Raises:
        HyperfineNotFoundError: If the executable cannot be found.
        HyperfineError: If the version output cannot be parsed.
    """
    exe = find_binary(binary)
    try:
        completed = subprocess.run(  # noqa: S603 - argv is a resolved executable, no shell
            [exe, "--version"], capture_output=True, text=True, check=False
        )
    except OSError as exc:
        raise HyperfineNotFoundError(exe) from exc
    match = _VERSION_RE.search(completed.stdout)
    if completed.returncode != 0 or match is None:
        msg = f"could not determine hyperfine version from {completed.stdout!r}"
        raise HyperfineError(msg)
    major, minor, patch = (int(part) for part in match.groups())
    return HyperfineVersion(major, minor, patch)


@dataclass(frozen=True, slots=True, kw_only=True)
class Invocation:
    """A fully prepared hyperfine process launch."""

    argv: tuple[str, ...]
    json_path: Path
    cwd: StrPath | None
    env: Mapping[str, str] | None
    timeout: float | None
    stream: bool
    reference: str | None = None

    def _environment(self) -> dict[str, str] | None:
        if self.env is None:
            return None
        return {**os.environ, **self.env}

    def _finish(self, returncode: int, stderr: str) -> BenchmarkReport:
        if returncode != 0:
            raise BenchmarkFailedError(
                returncode=returncode, stderr=stderr, argv=self.argv, streamed=self.stream
            )
        try:
            text = self.json_path.read_bytes()
        except OSError as exc:
            msg = f"hyperfine did not write its JSON export to {self.json_path}"
            raise HyperfineError(msg) from exc
        return BenchmarkReport.from_json_string(text, argv=self.argv, reference=self.reference)

    def run(self) -> BenchmarkReport:
        """Run hyperfine synchronously.

        Returns:
            The parsed report.

        Raises:
            HyperfineNotFoundError: If the executable cannot be started.
            BenchmarkTimeoutError: If ``timeout`` elapsed.
            BenchmarkFailedError: If hyperfine exited with a non-zero status.
        """
        pipe = None if self.stream else subprocess.PIPE
        try:
            process = subprocess.Popen(  # noqa: S603 - argv is built from typed options, no shell
                self.argv,
                stdin=subprocess.DEVNULL,
                stdout=pipe,
                stderr=pipe,
                cwd=self.cwd,
                env=self._environment(),
                process_group=_PROCESS_GROUP,
            )
        except (FileNotFoundError, PermissionError) as exc:
            raise HyperfineNotFoundError(self.argv[0]) from exc
        with process:
            try:
                stderr: bytes | None
                _, stderr = process.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired as exc:
                _kill_tree(process.pid, process.kill)
                _ = process.wait()
                raise BenchmarkTimeoutError(timeout=exc.timeout, argv=self.argv) from exc
            except BaseException:
                # e.g. KeyboardInterrupt: the child's process group no longer receives the
                # terminal's SIGINT, so take the benchmarked commands down explicitly.
                _kill_tree(process.pid, process.kill)
                _ = process.wait()
                raise
        return self._finish(process.returncode, _decode(stderr))

    async def arun(self) -> BenchmarkReport:
        """Run hyperfine as an asyncio subprocess.

        The hyperfine process is killed if the calling task is cancelled.

        Returns:
            The parsed report.

        Raises:
            HyperfineNotFoundError: If the executable cannot be started.
            BenchmarkTimeoutError: If ``timeout`` elapsed.
            BenchmarkFailedError: If hyperfine exited with a non-zero status.
        """
        pipe = None if self.stream else asyncio.subprocess.PIPE
        try:
            process = await asyncio.create_subprocess_exec(
                *self.argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=pipe,
                stderr=pipe,
                cwd=self.cwd,
                env=self._environment(),
                process_group=_PROCESS_GROUP,
            )
        except (FileNotFoundError, PermissionError) as exc:
            raise HyperfineNotFoundError(self.argv[0]) from exc
        try:
            async with asyncio.timeout(self.timeout):
                _, stderr = await process.communicate()
        except TimeoutError as exc:
            await _terminate(process)
            assert self.timeout is not None  # noqa: S101 - only asyncio.timeout can raise here
            raise BenchmarkTimeoutError(timeout=self.timeout, argv=self.argv) from exc
        except BaseException:
            await _terminate(process)
            raise
        returncode = process.returncode
        assert returncode is not None  # noqa: S101 - communicate() waits for exit
        return self._finish(returncode, _decode(stderr))


_PROCESS_GROUP = 0 if os.name == "posix" else None
"""``process_group`` for Popen: start hyperfine in its own process group (POSIX only).

Hyperfine does not forward termination to the commands it benchmarks, so killing just
hyperfine would leave them running. Killing its process group takes them down too.
"""


def _kill_tree(pid: int, kill: Callable[[], None]) -> None:
    """Kill hyperfine and the benchmarked commands in its process group."""
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGKILL)
    else:  # pragma: no cover - Windows has no process groups to kill
        with contextlib.suppress(ProcessLookupError):
            kill()


async def _terminate(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        _kill_tree(process.pid, process.kill)
        _ = await process.wait()


def _decode(data: bytes | None) -> str:
    return "" if data is None else data.decode(errors="replace")
