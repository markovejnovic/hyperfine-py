"""The :class:`Benchmark` configuration object and the :func:`run` / :func:`arun` shortcuts."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Unpack, cast, override

from hyperfine._errors import ValidationError
from hyperfine._options import (
    BenchmarkOptions,
    Output,
    SortOrder,
    Style,
    TimeUnit,
    check_placeholder_name,
)
from hyperfine._runner import Invocation, find_binary

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator, Mapping, Sequence

    from hyperfine._options import OptionName, OutputTarget, ParameterScan, StrPath
    from hyperfine._report import BenchmarkReport

__all__ = ["Benchmark", "arun", "run"]


@dataclass(frozen=True, slots=True, init=False)
class Benchmark:
    """An immutable, validated hyperfine configuration.

    Create it with the commands to benchmark and any :class:`BenchmarkOptions`, then
    call :meth:`run` (or :meth:`arun`) as often as needed. Invalid or conflicting
    options raise :class:`~hyperfine.ValidationError` immediately.

    Example:
        >>> bench = Benchmark("sleep 0.01", "sleep 0.02", runs=5, warmup=1)
        >>> bench.to_args()
        ('--warmup', '1', '--runs', '5', '--style', 'none', '--', 'sleep 0.01', 'sleep 0.02')

    Attributes:
        commands: The commands to benchmark.
    """

    commands: tuple[str, ...] = ()
    warmup: int | None = None
    min_runs: int | None = None
    max_runs: int | None = None
    runs: int | None = None
    setup: str | None = None
    prepare: tuple[str, ...] = ()
    conclude: tuple[str, ...] = ()
    cleanup: str | None = None
    parameter_scan: ParameterScan | None = None
    parameter_lists: tuple[tuple[str, tuple[str, ...]], ...] = ()
    shell_command: str | None = None
    no_shell: bool = False
    ignore_failure: bool | tuple[int, ...] = False
    command_names: tuple[str, ...] = ()
    reference: str | None = None
    reference_name: str | None = None
    style: Style | None = None
    sort: SortOrder | None = None
    time_unit: TimeUnit | None = None
    stdin: Path | None = None
    output: tuple[Output | Path, ...] = ()
    export_json: Path | None = None
    export_csv: Path | None = None
    export_markdown: Path | None = None
    export_asciidoc: Path | None = None
    export_orgmode: Path | None = None
    extra_args: tuple[str, ...] = ()
    cwd: Path | None = None
    env: tuple[tuple[str, str], ...] | None = field(default=None, repr=False)
    timeout: float | None = None
    binary: str | None = None
    stream: bool = False

    def __init__(self, *commands: str, **options: Unpack[BenchmarkOptions]) -> None:
        """Validate and store a benchmark configuration.

        Args:
            *commands: The commands to benchmark (at least one).
            **options: See :class:`BenchmarkOptions` for every supported option.

        Raises:
            ValidationError: If the options are invalid or conflict with each other.
        """
        values: dict[str, object] = {
            "commands": _commands(commands),
            "warmup": _count(options.get("warmup"), "warmup", minimum=0),
            "min_runs": _count(options.get("min_runs"), "min_runs", minimum=1),
            "max_runs": _count(options.get("max_runs"), "max_runs", minimum=1),
            "runs": _count(options.get("runs"), "runs", minimum=1),
            "setup": _command(options.get("setup"), "setup"),
            "prepare": _per_command(options.get("prepare"), "prepare"),
            "conclude": _per_command(options.get("conclude"), "conclude"),
            "cleanup": _command(options.get("cleanup"), "cleanup"),
            "parameter_scan": options.get("parameter_scan"),
            "parameter_lists": _parameter_lists(options.get("parameter_lists")),
            "shell_command": _command(options.get("shell_command"), "shell_command"),
            "no_shell": options.get("no_shell", False),
            "ignore_failure": _ignore_failure(options.get("ignore_failure", False)),
            "command_names": _per_command(options.get("command_names"), "command_names"),
            "reference": _command(options.get("reference"), "reference"),
            "reference_name": _command(options.get("reference_name"), "reference_name"),
            "style": _enum(Style, options.get("style"), "style"),
            "sort": _enum(SortOrder, options.get("sort"), "sort"),
            "time_unit": _enum(TimeUnit, options.get("time_unit"), "time_unit"),
            "stdin": _path(options.get("stdin")),
            "output": _outputs(options.get("output")),
            "export_json": _path(options.get("export_json")),
            "export_csv": _path(options.get("export_csv")),
            "export_markdown": _path(options.get("export_markdown")),
            "export_asciidoc": _path(options.get("export_asciidoc")),
            "export_orgmode": _path(options.get("export_orgmode")),
            "extra_args": _strings(options.get("extra_args", ()), "extra_args"),
            "cwd": _path(options.get("cwd")),
            "env": _env(options.get("env")),
            "timeout": _timeout(options.get("timeout")),
            "binary": _binary(options.get("binary")),
            "stream": options.get("stream", False),
        }
        object.__init__(self)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self._validate()

    # ---------------------------------------------------------------- validation

    def _validate(self) -> None:
        if self.runs is not None and (self.min_runs is not None or self.max_runs is not None):
            msg = "runs cannot be combined with min_runs or max_runs"
            raise ValidationError(msg)
        if (
            self.min_runs is not None
            and self.max_runs is not None
            and self.min_runs > self.max_runs
        ):
            msg = f"min_runs ({self.min_runs}) must not exceed max_runs ({self.max_runs})"
            raise ValidationError(msg)
        if self.no_shell and self.shell_command is not None:
            msg = "no_shell cannot be combined with shell_command"
            raise ValidationError(msg)
        if self.parameter_scan is not None and self.parameter_lists:
            msg = "parameter_scan cannot be combined with parameter_lists"
            raise ValidationError(msg)
        if self.reference_name is not None and self.reference is None:
            msg = "reference_name requires reference"
            raise ValidationError(msg)
        if self._has_parameters:
            # With parameters, hyperfine counts the expanded commands, which we cannot
            # know without replicating its expansion rules, so leave the check to it.
            return
        total = len(self.commands) + (self.reference is not None)
        for name, items in (
            ("prepare", self.prepare),
            ("conclude", self.conclude),
            ("output", self.output),
        ):
            if len(items) > 1 and len(items) != total:
                msg = (
                    f"{name} must be given once or once per command ({total} including the "
                    f"reference), got {len(items)}"
                )
                raise ValidationError(msg)
        if len(self.command_names) > len(self.commands):
            msg = f"got {len(self.command_names)} command_names for {len(self.commands)} command(s)"
            raise ValidationError(msg)

    @property
    def _has_parameters(self) -> bool:
        return self.parameter_scan is not None or bool(self.parameter_lists)

    # ---------------------------------------------------------------- argv

    def to_args(self, *, export_json: StrPath | None = None) -> tuple[str, ...]:
        """Return the arguments passed to hyperfine (excluding the executable).

        Useful for debugging or for running the benchmark by hand.

        Args:
            export_json: Override the ``--export-json`` destination.

        Returns:
            The argument vector, ending with ``--`` and the commands.
        """
        return tuple(self._iter_args(export_json))

    def _iter_args(self, export_json: StrPath | None) -> Iterator[str]:  # noqa: C901 - flat option mapping
        def opt(flag: str, value: object) -> Iterator[str]:
            if value is not None:
                yield flag
                yield os.fspath(value) if isinstance(value, Path) else str(value)

        yield from opt("--warmup", self.warmup)
        yield from opt("--min-runs", self.min_runs)
        yield from opt("--max-runs", self.max_runs)
        yield from opt("--runs", self.runs)
        yield from opt("--setup", self.setup)
        for command in self.prepare:
            yield from opt("--prepare", command)
        for command in self.conclude:
            yield from opt("--conclude", command)
        yield from opt("--cleanup", self.cleanup)
        if self.parameter_scan is not None:
            yield from self.parameter_scan.to_args()
        for name, values in self.parameter_lists:
            yield from ("--parameter-list", name, ",".join(values))
        yield from opt("--shell", self.shell_command)
        if self.no_shell:
            yield "-N"
        if self.ignore_failure is True:
            yield "--ignore-failure"
        elif self.ignore_failure:
            yield "--ignore-failure=" + ",".join(map(str, self.ignore_failure))
        for name in self.command_names:
            yield from opt("--command-name", name)
        yield from opt("--reference", self.reference)
        yield from opt("--reference-name", self.reference_name)
        style = self.style
        if style is None and not self.stream:
            style = Style.NONE
        yield from opt("--style", style)
        yield from opt("--sort", self.sort)
        yield from opt("--time-unit", self.time_unit)
        yield from opt("--input", _file_arg(self.stdin))
        for target in self.output:
            yield from opt("--output", _file_arg(target))
        json_path = self.export_json if export_json is None else export_json
        if json_path is not None:
            yield from opt("--export-json", Path(json_path))
        yield from opt("--export-csv", self.export_csv)
        yield from opt("--export-markdown", self.export_markdown)
        yield from opt("--export-asciidoc", self.export_asciidoc)
        yield from opt("--export-orgmode", self.export_orgmode)
        yield from self.extra_args
        yield "--"
        yield from self.commands

    def argv(self) -> tuple[str, ...]:
        """Return the complete command line, including the resolved executable.

        Returns:
            The argument vector that :meth:`run` would execute (minus the temporary
            JSON export used internally).

        Raises:
            HyperfineNotFoundError: If the executable cannot be found.
        """
        return (find_binary(self.binary), *self.to_args())

    # ---------------------------------------------------------------- execution

    def run(self) -> BenchmarkReport:
        """Run the benchmark and wait for it to finish.

        Returns:
            The parsed report.

        Raises:
            HyperfineNotFoundError: If the executable cannot be found.
            BenchmarkFailedError: If hyperfine exits with a non-zero status.
            BenchmarkTimeoutError: If ``timeout`` elapses.
        """
        with tempfile.TemporaryDirectory(prefix="hyperfine-py-") as tmp:
            return self._invocation(Path(tmp)).run()

    async def arun(self) -> BenchmarkReport:
        """Run the benchmark as an asyncio subprocess.

        Cancelling the awaiting task kills the hyperfine process.

        Returns:
            The parsed report.

        Raises:
            HyperfineNotFoundError: If the executable cannot be found.
            BenchmarkFailedError: If hyperfine exits with a non-zero status.
            BenchmarkTimeoutError: If ``timeout`` elapses.
        """
        with tempfile.TemporaryDirectory(prefix="hyperfine-py-") as tmp:
            return await self._invocation(Path(tmp)).arun()

    def _invocation(self, tmp: Path) -> Invocation:
        json_path = self.export_json or tmp / "results.json"
        if self.cwd is not None and not json_path.is_absolute():
            json_path = self.cwd / json_path
        argv = (find_binary(self.binary), *self.to_args(export_json=json_path.absolute()))
        return Invocation(
            argv=argv,
            json_path=json_path,
            cwd=self.cwd,
            env=None if self.env is None else dict(self.env),
            timeout=self.timeout,
            stream=self.stream,
            reference=self.reference_name or self.reference,
        )

    # ---------------------------------------------------------------- derivation

    def options(self) -> BenchmarkOptions:
        """Return the options of this benchmark, suitable for ``Benchmark(*cmds, **opts)``.

        Returns:
            Only the options that differ from their defaults (``None``, ``()`` or ``False``).
        """
        opts: dict[str, object] = {}
        for name in sorted(BenchmarkOptions.__optional_keys__):
            value: object = getattr(self, name)
            if value is None or value is False or value == ():
                continue
            if name in {"parameter_lists", "env"} and isinstance(value, tuple):
                value = dict(cast("tuple[tuple[str, object], ...]", value))
            opts[name] = value
        return cast("BenchmarkOptions", opts)

    def with_options(self, **changes: Unpack[BenchmarkOptions]) -> Benchmark:
        """Return a copy with some options replaced.

        Setting ``runs`` drops ``min_runs``/``max_runs`` and vice versa, so a copy can
        switch between an exact and a bounded number of runs. Use :meth:`without` to
        remove other options.

        Args:
            **changes: Options to override; see :class:`BenchmarkOptions`.

        Returns:
            A new, validated :class:`Benchmark`.

        Example:
            >>> Benchmark("true", min_runs=5).with_options(runs=3).to_args()
            ('--runs', '3', '--style', 'none', '--', 'true')
        """
        merged: dict[str, object] = dict(self.options())
        if "runs" in changes:
            _ = merged.pop("min_runs", None)
            _ = merged.pop("max_runs", None)
        if "min_runs" in changes or "max_runs" in changes:
            _ = merged.pop("runs", None)
        if "shell_command" in changes:
            _ = merged.pop("no_shell", None)
        if "no_shell" in changes:
            _ = merged.pop("shell_command", None)
        merged.update(changes)
        return Benchmark(*self.commands, **cast("BenchmarkOptions", merged))

    def without(self, *names: OptionName) -> Benchmark:
        """Return a copy with the given options reset to their defaults.

        Args:
            *names: Names of options to remove, e.g. ``"warmup"``.

        Returns:
            A new, validated :class:`Benchmark`.

        Raises:
            ValidationError: If a name is not a known option.

        Example:
            >>> Benchmark("true", runs=3, warmup=1).without("warmup")
            Benchmark('true', runs=3)
        """
        unknown = sorted(set(names) - BenchmarkOptions.__optional_keys__)
        if unknown:
            msg = f"unknown option(s): {', '.join(map(repr, unknown))}"
            raise ValidationError(msg)
        merged: dict[str, object] = dict(self.options())
        for name in names:
            _ = merged.pop(name, None)
        return Benchmark(*self.commands, **cast("BenchmarkOptions", merged))

    def __replace__(self, **changes: Unpack[BenchmarkOptions]) -> Benchmark:
        """Support :func:`copy.replace` (Python 3.13+); equivalent to :meth:`with_options`.

        Note that :func:`dataclasses.replace` is not supported; use this or
        :meth:`with_options` instead.

        Args:
            **changes: Options to override; see :class:`BenchmarkOptions`.

        Returns:
            A new, validated :class:`Benchmark`.
        """
        return self.with_options(**changes)

    def with_commands(self, *commands: str) -> Benchmark:
        """Return a copy that benchmarks different commands with the same options.

        Args:
            *commands: The new commands.

        Returns:
            A new, validated :class:`Benchmark`.
        """
        return Benchmark(*commands, **self.options())

    @override
    def __repr__(self) -> str:
        parts = [repr(command) for command in self.commands]
        parts.extend(f"{key}={value!r}" for key, value in self.options().items())
        return f"Benchmark({', '.join(parts)})"


def run(*commands: str, **options: Unpack[BenchmarkOptions]) -> BenchmarkReport:
    """Benchmark one or more commands with hyperfine.

    Args:
        *commands: The commands to benchmark.
        **options: See :class:`BenchmarkOptions`.

    Returns:
        The parsed report.

    Raises:
        ValidationError: If the options are invalid.
        HyperfineNotFoundError: If the executable cannot be found.
        BenchmarkFailedError: If hyperfine exits with a non-zero status.
        BenchmarkTimeoutError: If ``timeout`` elapses.

    Example:
        Compare two commands and print hyperfine-style output::

            report = run("sleep 0.01", "sleep 0.02", runs=3)
            print(report.summary())
    """
    return Benchmark(*commands, **options).run()


async def arun(*commands: str, **options: Unpack[BenchmarkOptions]) -> BenchmarkReport:
    """Asynchronously benchmark one or more commands with hyperfine.

    Args:
        *commands: The commands to benchmark.
        **options: See :class:`BenchmarkOptions`.

    Returns:
        The parsed report.

    Raises:
        ValidationError: If the options are invalid.
        HyperfineNotFoundError: If the executable cannot be found.
        BenchmarkFailedError: If hyperfine exits with a non-zero status.
        BenchmarkTimeoutError: If ``timeout`` elapses.
    """
    return await Benchmark(*commands, **options).arun()


# --------------------------------------------------------------------------- normalisation


def _commands(commands: tuple[str, ...]) -> tuple[str, ...]:
    if not commands:
        msg = "at least one command is required"
        raise ValidationError(msg)
    return _strings(commands, "commands")


def _strings(values: Sequence[str], option: str) -> tuple[str, ...]:
    if isinstance(values, str):
        msg = f"{option} must be a sequence of strings, not a single string"
        raise ValidationError(msg)
    out = tuple(values)
    for value in out:
        _ = _require_str(value, option)
    return out


def _require_str(value: object, option: str) -> str:
    if not isinstance(value, str):
        msg = f"{option}: expected a string, got {type(value).__name__}"
        raise ValidationError(msg)
    if not value.strip():
        msg = f"{option}: must not be empty"
        raise ValidationError(msg)
    return value


def _command(value: str | None, option: str) -> str | None:
    return None if value is None else _require_str(value, option)


def _per_command(value: str | Sequence[str] | None, option: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (_require_str(value, option),)
    out = _strings(value, option)
    if not out:
        msg = f"{option}: must not be an empty sequence"
        raise ValidationError(msg)
    return out


def _count(value: int | None, option: str, *, minimum: int) -> int | None:
    if value is None:
        return None
    if not _is_strict_int(value):
        msg = f"{option}: expected an integer, got {type(value).__name__}"
        raise ValidationError(msg)
    if value < minimum:
        msg = f"{option}: must be >= {minimum}, got {value}"
        raise ValidationError(msg)
    return value


def _timeout(value: float | None) -> float | None:
    if value is not None and not value > 0:
        msg = f"timeout: must be positive, got {value}"
        raise ValidationError(msg)
    return value


def _enum[E: (Style, SortOrder, TimeUnit)](
    kind: type[E], value: str | None, option: str
) -> E | None:
    if value is None:
        return None
    try:
        return kind(value)
    except ValueError:
        choices = ", ".join(repr(member.value) for member in kind)
        msg = f"{option}: invalid value {value!r}; expected one of {choices}"
        raise ValidationError(msg) from None


def _path(value: StrPath | None) -> Path | None:
    return None if value is None else Path(value)


def _binary(value: StrPath | None) -> str | None:
    return None if value is None else os.fspath(value)


_RESERVED_FILE_NAMES = frozenset({"null", "pipe", "inherit"})


def _file_arg[T](target: T) -> T | str:
    """Keep relative paths named like hyperfine's keywords from being read as keywords."""
    if isinstance(target, Path) and not target.is_absolute():
        text = os.fspath(target)
        if text in _RESERVED_FILE_NAMES:
            return f".{os.sep}{text}"
    return target


def _outputs(value: OutputTarget | Sequence[OutputTarget] | None) -> tuple[Output | Path, ...]:
    if value is None:
        return ()
    if isinstance(value, str | os.PathLike):
        return (_output(value),)
    items = tuple(_output(item) for item in value)
    if not items:
        msg = "output: must not be an empty sequence"
        raise ValidationError(msg)
    return items


def _output(item: str | os.PathLike[str]) -> Output | Path:
    if isinstance(item, str):
        try:
            return Output(item)
        except ValueError:
            msg = (
                f"output: invalid value {item!r}; use one of 'null', 'pipe', 'inherit' "
                "or a pathlib.Path to write to a file"
            )
            raise ValidationError(msg) from None
    return Path(item)


def _parameter_lists(
    value: Mapping[str, Sequence[str | int | float]] | None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if value is None:
        return ()
    out: list[tuple[str, tuple[str, ...]]] = []
    for name, raw in value.items():
        check_placeholder_name(name, "parameter_lists")
        if isinstance(raw, str):
            msg = f"parameter_lists[{name!r}]: expected a sequence of values, not a string"
            raise ValidationError(msg)
        values = tuple(str(item) for item in raw)
        if not values:
            msg = f"parameter_lists[{name!r}]: must contain at least one value"
            raise ValidationError(msg)
        if any("," in item for item in values):
            msg = f"parameter_lists[{name!r}]: values must not contain ','"
            raise ValidationError(msg)
        out.append((name, values))
    return tuple(out)


def _ignore_failure(value: bool | Collection[int]) -> bool | tuple[int, ...]:  # noqa: FBT001 - mirrors option type
    if isinstance(value, bool):
        return value
    codes = tuple(sorted(set(value)))
    if not codes:
        msg = "ignore_failure: pass True or a non-empty collection of exit codes"
        raise ValidationError(msg)
    for code in codes:
        if not _is_strict_int(code):
            msg = f"ignore_failure: exit codes must be integers, got {code!r}"
            raise ValidationError(msg)
    return codes


def _env(value: Mapping[str, str] | None) -> tuple[tuple[str, str], ...] | None:
    if value is None:
        return None
    for key, item in value.items():
        if not (_is_str(key) and _is_str(item)):
            msg = f"env: keys and values must be strings, got {key!r}={item!r}"
            raise ValidationError(msg)
    return tuple(value.items())


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_str(value: object) -> bool:
    return isinstance(value, str)
