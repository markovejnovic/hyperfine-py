"""Typed models for hyperfine's JSON export and the comparison helpers built on top of it."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast, overload, override

from hyperfine._errors import AmbiguousCommandError, ReportParseError

if TYPE_CHECKING:
    import os
    from collections.abc import Iterator, Mapping, Sequence

__all__ = ["BenchmarkReport", "BenchmarkResult", "JSONValue", "Speedup", "format_duration"]

type JSONValue = str | int | float | bool | list[JSONValue] | dict[str, JSONValue] | None
"""A JSON document as produced by :func:`json.loads`."""

type _Slice = slice[int | None, int | None, int | None]

_MS_PER_S = 1_000.0
_US_PER_S = 1_000_000.0


def format_duration(seconds: float) -> str:
    """Render a duration in seconds with an automatically chosen unit.

    Args:
        seconds: The duration in seconds.

    Returns:
        A human friendly string such as ``"12.3 ms"``.

    Example:
        >>> format_duration(0.0123)
        '12.3 ms'
    """
    magnitude = abs(seconds)
    if magnitude >= 1.0:
        return f"{seconds:.3f} s"
    if magnitude >= 1.0 / _MS_PER_S:
        return f"{seconds * _MS_PER_S:.1f} ms"
    return f"{seconds * _US_PER_S:.1f} µs"


# --------------------------------------------------------------------------- parsing helpers


def _fail(path: str, expected: str, value: object) -> ReportParseError:
    return ReportParseError(f"{path}: expected {expected}, got {type(value).__name__}")


def _as_object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _fail(path, "an object", value)
    return cast("dict[str, object]", value)


def _as_list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise _fail(path, "an array", value)
    return cast("list[object]", value)


def _as_str(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise _fail(path, "a string", value)
    return value


def _as_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _fail(path, "a number", value)
    return float(value)


def _as_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(path, "an integer", value)
    return value


def _optional_float(value: object, path: str) -> float | None:
    return None if value is None else _as_float(value, path)


def _floats(value: object, path: str) -> tuple[float, ...]:
    return tuple(_as_float(item, f"{path}[{i}]") for i, item in enumerate(_as_list(value, path)))


def _optional_floats(value: object, path: str) -> tuple[float, ...] | None:
    return None if value is None else _floats(value, path)


def _optional_ints(value: object, path: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    items = _as_list(value, path)
    return tuple(_as_int(item, f"{path}[{i}]") for i, item in enumerate(items))


def _exit_codes(value: object, path: str) -> tuple[int | None, ...]:
    if value is None:
        return ()
    items = _as_list(value, path)
    return tuple(
        None if item is None else _as_int(item, f"{path}[{i}]") for i, item in enumerate(items)
    )


def _parameters(value: object, path: str) -> dict[str, str]:
    if value is None:
        return {}
    return {key: _as_str(item, f"{path}.{key}") for key, item in _as_object(value, path).items()}


def _mean(result: BenchmarkResult) -> float:
    return result.mean


def _factor(speedup: Speedup) -> float:
    return speedup.factor


# --------------------------------------------------------------------------- models


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkResult:
    """Timing statistics for a single benchmarked command.

    All durations are in seconds, exactly as exported by hyperfine.

    Attributes:
        command: The command (or its ``--command-name``) that was benchmarked.
        mean: Arithmetic mean of the wall clock times.
        stddev: Standard deviation of the wall clock times; ``None`` for a single run.
        median: Median wall clock time.
        user: Mean user-mode CPU time.
        system: Mean kernel-mode CPU time.
        min: Fastest wall clock time.
        max: Slowest wall clock time.
        times: Wall clock time of every run, or ``None`` if not exported.
        exit_codes: Exit code of every run; ``None`` entries mean the process was
            terminated by a signal.
        memory_usage_byte: Peak memory usage of every run, if reported by hyperfine.
        parameters: Values of the ``--parameter-scan``/``--parameter-list`` variables.
    """

    command: str
    mean: float
    stddev: float | None
    median: float
    user: float
    system: float
    min: float
    max: float
    times: tuple[float, ...] | None = None
    exit_codes: tuple[int | None, ...] = ()
    memory_usage_byte: tuple[int, ...] | None = None
    parameters: Mapping[str, str] = field(default_factory=dict[str, str], hash=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, object], *, path: str = "result") -> BenchmarkResult:
        """Build a result from one entry of hyperfine's ``results`` array.

        Args:
            data: The decoded JSON object.
            path: Location of ``data`` in the document, used in error messages.

        Returns:
            The parsed result.

        Raises:
            ReportParseError: If a field is missing or has the wrong type.
        """

        def req(key: str) -> object:
            if key not in data:
                msg = f"{path}: missing required field {key!r}"
                raise ReportParseError(msg)
            return data[key]

        return cls(
            command=_as_str(req("command"), f"{path}.command"),
            mean=_as_float(req("mean"), f"{path}.mean"),
            stddev=_optional_float(data.get("stddev"), f"{path}.stddev"),
            median=_as_float(req("median"), f"{path}.median"),
            user=_as_float(req("user"), f"{path}.user"),
            system=_as_float(req("system"), f"{path}.system"),
            min=_as_float(req("min"), f"{path}.min"),
            max=_as_float(req("max"), f"{path}.max"),
            times=_optional_floats(data.get("times"), f"{path}.times"),
            exit_codes=_exit_codes(data.get("exit_codes"), f"{path}.exit_codes"),
            memory_usage_byte=_optional_ints(
                data.get("memory_usage_byte"), f"{path}.memory_usage_byte"
            ),
            parameters=_parameters(data.get("parameters"), f"{path}.parameters"),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        """Convert back to hyperfine's JSON representation.

        Returns:
            A JSON-serialisable dictionary; optional fields that are absent are omitted.
        """
        out: dict[str, JSONValue] = {
            "command": self.command,
            "mean": self.mean,
            "stddev": self.stddev,
            "median": self.median,
            "user": self.user,
            "system": self.system,
            "min": self.min,
            "max": self.max,
        }
        if self.times is not None:
            out["times"] = list(self.times)
        if self.memory_usage_byte is not None:
            out["memory_usage_byte"] = list(self.memory_usage_byte)
        out["exit_codes"] = list(self.exit_codes)
        if self.parameters:
            out["parameters"] = dict(self.parameters)
        return out

    @property
    def runs(self) -> int:
        """Number of timing runs (``0`` if individual times were not exported)."""
        return len(self.times) if self.times is not None else 0

    @property
    def mean_ms(self) -> float:
        """Mean wall clock time in milliseconds."""
        return self.mean * _MS_PER_S

    @property
    def median_ms(self) -> float:
        """Median wall clock time in milliseconds."""
        return self.median * _MS_PER_S

    @property
    def stddev_ms(self) -> float | None:
        """Standard deviation in milliseconds, or ``None`` for a single run."""
        return None if self.stddev is None else self.stddev * _MS_PER_S

    @property
    def succeeded(self) -> bool:
        """Whether every run exited with status ``0``."""
        return all(code == 0 for code in self.exit_codes)

    def compare(self, baseline: BenchmarkResult) -> Speedup:
        """Compare this result against ``baseline``.

        Args:
            baseline: The result to compare against.

        Returns:
            A :class:`Speedup` whose ``factor`` is ``baseline.mean / self.mean``
            (greater than one when this result is faster).
        """
        return Speedup.between(self, baseline)

    @override
    def __str__(self) -> str:
        spread = f" ± {format_duration(self.stddev)}" if self.stddev is not None else ""
        return (
            f"{self.command}: {format_duration(self.mean)}{spread} "
            f"[{format_duration(self.min)} … {format_duration(self.max)}], {self.runs} runs"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Speedup:
    """How much faster ``result`` ran compared to ``baseline``.

    Attributes:
        result: The result being described.
        baseline: The result it is compared against.
        factor: ``baseline.mean / result.mean``; above ``1`` means ``result`` is faster.
        stddev: Propagated standard deviation of ``factor``, when both inputs have one.
    """

    result: BenchmarkResult
    baseline: BenchmarkResult
    factor: float
    stddev: float | None

    @classmethod
    def between(cls, result: BenchmarkResult, baseline: BenchmarkResult) -> Speedup:
        """Compute the speedup of ``result`` relative to ``baseline``.

        Args:
            result: The result being described.
            baseline: The result it is compared against.

        Returns:
            The computed speedup. Zero mean times (which hyperfine can report for
            extremely fast commands) yield ``inf`` or ``1.0`` rather than an error.
        """
        if result.mean == 0.0:
            factor = 1.0 if baseline.mean == 0.0 else math.inf
            return cls(result=result, baseline=baseline, factor=factor, stddev=None)
        factor = baseline.mean / result.mean
        stddev: float | None = None
        if result.stddev is not None and baseline.stddev is not None and baseline.mean != 0.0:
            stddev = factor * math.hypot(
                result.stddev / result.mean, baseline.stddev / baseline.mean
            )
        return cls(result=result, baseline=baseline, factor=factor, stddev=stddev)

    @property
    def is_faster(self) -> bool:
        """Whether ``result`` ran faster than ``baseline``."""
        return self.factor > 1.0

    @override
    def __str__(self) -> str:
        if self.factor >= 1.0:
            ratio, stddev, word = self.factor, self.stddev, "faster"
        else:
            ratio = 1.0 / self.factor
            stddev = None if self.stddev is None else self.stddev / self.factor**2
            word = "slower"
        spread = f" ± {stddev:.2f}" if stddev is not None else ""
        return (
            f"{self.result.command!r} ran {ratio:.2f}{spread} times {word} "
            f"than {self.baseline.command!r}"
        )


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """The outcome of a hyperfine invocation: a sequence of :class:`BenchmarkResult`.

    Results can be accessed by position (``report[0]``, ``report[1:]``), by command
    name (``report["sleep 0.1"]``) or by iteration.

    Attributes:
        results: The results in the order hyperfine exported them.
        argv: The command line that produced the report, if it was run from Python.
        reference: Command name of the ``--reference`` result, if one was benchmarked.
            Comparisons (:meth:`summary`, :meth:`relative_to`, :meth:`table`) are then
            anchored on it, like hyperfine's own output; otherwise on :attr:`fastest`.
    """

    results: tuple[BenchmarkResult, ...]
    argv: tuple[str, ...] | None = field(default=None, kw_only=True, compare=False)
    reference: str | None = field(default=None, kw_only=True, compare=False)

    def __post_init__(self) -> None:
        """Check that :attr:`reference` names one of the results.

        Raises:
            ReportParseError: If :attr:`reference` is not the name of any result.
        """
        if self.reference is not None and self.reference not in self:
            msg = f"reference {self.reference!r} is not one of the results"
            raise ReportParseError(msg)

    # ---------------------------------------------------------------- constructors

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, object],
        *,
        argv: Sequence[str] | None = None,
        reference: str | None = None,
    ) -> BenchmarkReport:
        """Build a report from a decoded hyperfine JSON document.

        Args:
            data: The decoded JSON document (``{"results": [...]}``).
            argv: Optional command line that produced the document.
            reference: Optional command name of the ``--reference`` result.

        Returns:
            The parsed report.

        Raises:
            ReportParseError: If the document does not match hyperfine's schema.
        """
        if "results" not in data:
            msg = "document: missing required field 'results'"
            raise ReportParseError(msg)
        entries = _as_list(data["results"], "results")
        results = tuple(
            BenchmarkResult.from_dict(_as_object(entry, f"results[{i}]"), path=f"results[{i}]")
            for i, entry in enumerate(entries)
        )
        return cls(results, argv=None if argv is None else tuple(argv), reference=reference)

    @classmethod
    def from_json_string(
        cls,
        text: str | bytes,
        *,
        argv: Sequence[str] | None = None,
        reference: str | None = None,
    ) -> BenchmarkReport:
        """Parse a report from the text of a hyperfine ``--export-json`` file.

        Args:
            text: The JSON text.
            argv: Optional command line that produced the document.
            reference: Optional command name of the ``--reference`` result.

        Returns:
            The parsed report.

        Raises:
            ReportParseError: If the text is not valid JSON or does not match the schema.
        """
        try:
            document: object = json.loads(text)
        except json.JSONDecodeError as exc:
            msg = f"invalid JSON: {exc}"
            raise ReportParseError(msg) from exc
        return cls.from_dict(_as_object(document, "document"), argv=argv, reference=reference)

    @classmethod
    def from_json(
        cls,
        path: str | os.PathLike[str],
        *,
        argv: Sequence[str] | None = None,
        reference: str | None = None,
    ) -> BenchmarkReport:
        """Load a report from an existing hyperfine ``--export-json`` file.

        The export does not record which command was the ``--reference``; pass
        ``reference`` to anchor comparisons on it (hyperfine exports it first).

        Args:
            path: Path of the JSON file.
            argv: Optional command line that produced the file.
            reference: Optional command name of the ``--reference`` result.

        Returns:
            The parsed report.

        Raises:
            ReportParseError: If the file is not a valid hyperfine export.
            OSError: If the file cannot be read (e.g. :class:`FileNotFoundError`).
        """
        return cls.from_json_string(Path(path).read_bytes(), argv=argv, reference=reference)

    # ---------------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, JSONValue]:
        """Convert to hyperfine's JSON document structure.

        Returns:
            A new JSON-serialisable dictionary of the form ``{"results": [...]}``.
        """
        results: list[JSONValue] = [result.to_dict() for result in self.results]
        return {"results": results}

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise to JSON text compatible with hyperfine's ``--export-json``.

        Args:
            indent: Indentation passed to :func:`json.dumps`.

        Returns:
            The JSON text.
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    # ---------------------------------------------------------------- sequence protocol

    def __len__(self) -> int:
        """Return the number of results."""
        return len(self.results)

    def __iter__(self) -> Iterator[BenchmarkResult]:
        """Iterate over the results in export order."""
        return iter(self.results)

    def __contains__(self, command: object) -> bool:
        """Return whether a result with the given command name exists."""
        return any(result.command == command for result in self.results)

    @overload
    def __getitem__(self, key: int) -> BenchmarkResult: ...
    @overload
    def __getitem__(self, key: str) -> BenchmarkResult: ...
    @overload
    def __getitem__(self, key: _Slice) -> tuple[BenchmarkResult, ...]: ...
    def __getitem__(self, key: int | str | _Slice) -> BenchmarkResult | tuple[BenchmarkResult, ...]:
        """Look up a result by position or by command name, or slice the results.

        Args:
            key: An index into :attr:`results`, a slice of it, or a command name.

        Returns:
            The matching result, or a tuple of results for a slice.

        Raises:
            KeyError: If no result has the given command name.
            AmbiguousCommandError: If several results share the command name.
            TypeError: If ``key`` is not an ``int``, ``slice`` or ``str``.
        """
        if not isinstance(key, str):
            return self.results[key]
        indices = self._indices(key)
        if len(indices) > 1:
            raise AmbiguousCommandError(command=key, indices=indices)
        if indices:
            return self.results[indices[0]]
        known = ", ".join(repr(result.command) for result in self.results)
        msg = f"no result for command {key!r}; known commands: {known}"
        raise KeyError(msg)

    def find_all(self, command: str) -> tuple[BenchmarkResult, ...]:
        """Return every result with the given command name, in export order.

        Unlike ``report[command]`` this never raises; use it when a command may have
        been benchmarked more than once.

        Args:
            command: The command name to look for.

        Returns:
            The matching results (empty if there are none).
        """
        return tuple(self.results[i] for i in self._indices(command))

    def _indices(self, command: str) -> tuple[int, ...]:
        return tuple(i for i, result in enumerate(self.results) if result.command == command)

    # ---------------------------------------------------------------- analysis

    @property
    def commands(self) -> tuple[str, ...]:
        """The command names in export order."""
        return tuple(result.command for result in self.results)

    @property
    def fastest(self) -> BenchmarkResult:
        """The result with the lowest mean time.

        Raises:
            ValueError: If the report is empty.
        """
        return min(self._non_empty(), key=_mean)

    @property
    def slowest(self) -> BenchmarkResult:
        """The result with the highest mean time.

        Raises:
            ValueError: If the report is empty.
        """
        return max(self._non_empty(), key=_mean)

    def sorted(self, *, reverse: bool = False) -> tuple[BenchmarkResult, ...]:
        """Return the results ordered by mean time (fastest first).

        Args:
            reverse: Order slowest first instead.

        Returns:
            The sorted results.
        """
        return tuple(sorted(self.results, key=_mean, reverse=reverse))

    def relative_to(
        self, reference: str | int | BenchmarkResult | None = None
    ) -> tuple[Speedup, ...]:
        """Compare the reference against every other result, like hyperfine's summary.

        Args:
            reference: The result to compare from, given as a result, a command name
                or an index. Defaults to the report's :attr:`reference` if set, else
                :attr:`fastest`.

        Returns:
            One :class:`Speedup` per other result, each with ``result`` set to the
            reference and ``baseline`` set to the other result, sorted by factor.
        """
        ref = self._resolve(reference)
        others = (result for result in self.results if result is not ref)
        return tuple(sorted((ref.compare(other) for other in others), key=_factor))

    def summary(self, reference: str | int | BenchmarkResult | None = None) -> str:
        """Render hyperfine's ``Summary`` section.

        Args:
            reference: See :meth:`relative_to`.

        Returns:
            A multi-line string: a ``'<reference>' ran`` line followed by one
            ``N ± stddev times faster than '<other>'`` line per other result.
        """
        ref = self._resolve(reference)
        lines = [f"{ref.command!r} ran"]
        for speedup in self.relative_to(ref):
            text = str(speedup).removeprefix(f"{ref.command!r} ran ")
            lines.append(f"  {text}")
        return "\n".join(lines)

    def table(self) -> str:
        """Render the results as an aligned plain-text table.

        The ``Relative`` column is each mean divided by the baseline's mean, where the
        baseline is :attr:`reference` if set, else :attr:`fastest`.

        Returns:
            The table, one row per result.
        """
        baseline_mean = self._resolve(None).mean if self.results else 0.0
        header = ("Command", "Mean", "StdDev", "Min", "Max", "Relative")
        rows: list[tuple[str, ...]] = [header]
        for result in self.results:
            relative = result.mean / baseline_mean if baseline_mean else 1.0
            rows.append(
                (
                    result.command,
                    format_duration(result.mean),
                    "—" if result.stddev is None else format_duration(result.stddev),
                    format_duration(result.min),
                    format_duration(result.max),
                    f"{relative:.2f}",
                )
            )
        widths = [max(len(row[col]) for row in rows) for col in range(len(header))]
        rendered = [
            "  ".join(
                cell.ljust(width) if col == 0 else cell.rjust(width)
                for col, (cell, width) in enumerate(zip(row, widths, strict=True))
            ).rstrip()
            for row in rows
        ]
        rendered.insert(1, "  ".join("-" * width for width in widths))
        return "\n".join(rendered)

    @override
    def __str__(self) -> str:
        if not self.results:
            return "BenchmarkReport (no results)"
        if len(self.results) == 1:
            return self.table()
        return f"{self.table()}\n\n{self.summary()}"

    @override
    def __repr__(self) -> str:
        parts = ", ".join(
            f"{result.command!r}: {format_duration(result.mean)}" for result in self.results
        )
        return f"BenchmarkReport({parts})"

    # ---------------------------------------------------------------- helpers

    def _non_empty(self) -> tuple[BenchmarkResult, ...]:
        if not self.results:
            msg = "the report contains no results"
            raise ValueError(msg)
        return self.results

    def _resolve(self, reference: str | int | BenchmarkResult | None) -> BenchmarkResult:
        if reference is None:
            if self.reference is None:
                return self.fastest
            # hyperfine exports the reference first, so the first match is it even
            # when the same command was also benchmarked on its own.
            return self.find_all(self.reference)[0]
        if isinstance(reference, BenchmarkResult):
            return reference
        return self[reference]
