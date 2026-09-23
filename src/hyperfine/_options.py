"""Option types accepted by :class:`hyperfine.Benchmark` and :func:`hyperfine.run`."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, TypedDict

from hyperfine._errors import ValidationError

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

__all__ = [
    "BenchmarkOptions",
    "OptionName",
    "Output",
    "OutputTarget",
    "ParameterScan",
    "ParameterValue",
    "SortLike",
    "SortOrder",
    "StrPath",
    "Style",
    "StyleLike",
    "TimeUnit",
    "TimeUnitLike",
]

type StrPath = str | os.PathLike[str]
"""A filesystem path given as a string or path-like object."""

type ParameterValue = str | int | float
"""A value substituted for a ``{name}`` placeholder."""


class Style(StrEnum):
    """Values for hyperfine's ``--style`` option."""

    AUTO = "auto"
    BASIC = "basic"
    FULL = "full"
    NOCOLOR = "nocolor"
    COLOR = "color"
    NONE = "none"


class SortOrder(StrEnum):
    """Values for hyperfine's ``--sort`` option."""

    AUTO = "auto"
    COMMAND = "command"
    MEAN_TIME = "mean-time"


class TimeUnit(StrEnum):
    """Values for hyperfine's ``--time-unit`` option."""

    MICROSECOND = "microsecond"
    MILLISECOND = "millisecond"
    SECOND = "second"


class Output(StrEnum):
    """Special destinations for hyperfine's ``--output`` option.

    To write the benchmarked command's output to a file, pass a :class:`~pathlib.Path`.
    """

    NULL = "null"
    PIPE = "pipe"
    INHERIT = "inherit"


type StyleLike = Style | Literal["auto", "basic", "full", "nocolor", "color", "none"]
"""A :class:`Style` member or its string value."""

type SortLike = SortOrder | Literal["auto", "command", "mean-time"]
"""A :class:`SortOrder` member or its string value."""

type TimeUnitLike = TimeUnit | Literal["microsecond", "millisecond", "second"]
"""A :class:`TimeUnit` member or its string value."""

type OutputTarget = Output | Literal["null", "pipe", "inherit"] | os.PathLike[str]
"""An :class:`Output` member, its string value, or a path to write the output to."""


@dataclass(frozen=True, slots=True)
class ParameterScan:
    """A numeric parameter range, mapped to ``--parameter-scan`` / ``--parameter-step-size``.

    Every occurrence of ``{name}`` in the commands is replaced by each value in
    ``start..=stop``.

    Attributes:
        name: The placeholder name, used as ``{name}`` in commands.
        start: First value of the range (inclusive).
        stop: Last value of the range (inclusive).
        step: Step between values. Required when ``start`` or ``stop`` is a float.

    Example:
        >>> ParameterScan("threads", 1, 8).to_args()
        ('--parameter-scan', 'threads', '1', '8')
    """

    name: str
    start: float
    stop: float
    step: float | None = None

    def __post_init__(self) -> None:
        """Validate the range.

        Raises:
            ValidationError: If the name is empty, the range is empty, the step is not
                positive, a value is not finite, or a floating point range has no step.
        """
        check_placeholder_name(self.name, "parameter_scan")
        bounds = (
            (self.start, self.stop) if self.step is None else (self.start, self.stop, self.step)
        )
        if not all(math.isfinite(value) for value in bounds):
            msg = "parameter_scan: start, stop and step must be finite numbers"
            raise ValidationError(msg)
        if self.stop < self.start:
            msg = f"parameter_scan: stop ({self.stop}) must be >= start ({self.start})"
            raise ValidationError(msg)
        if self.step is None:
            if not (_is_int(self.start) and _is_int(self.stop)):
                msg = "parameter_scan: a step is required when start or stop is a float"
                raise ValidationError(msg)
        elif self.step <= 0:
            msg = f"parameter_scan: step must be positive, got {self.step}"
            raise ValidationError(msg)

    def to_args(self) -> tuple[str, ...]:
        """Return the hyperfine arguments for this scan.

        Returns:
            The ``--parameter-scan`` (and optional ``--parameter-step-size``) arguments.
        """
        args = ("--parameter-scan", self.name, _fmt_number(self.start), _fmt_number(self.stop))
        if self.step is None:
            return args
        return (*args, "--parameter-step-size", _fmt_number(self.step))


class BenchmarkOptions(TypedDict, total=False):
    """Keyword options accepted by :class:`~hyperfine.Benchmark`, ``run()`` and ``arun()``.

    Every key is optional; omitted keys use hyperfine's defaults.

    Attributes:
        warmup: Number of warmup runs before timing (``--warmup``).
        min_runs: Minimum number of timing runs (``--min-runs``).
        max_runs: Maximum number of timing runs (``--max-runs``).
        runs: Exact number of timing runs (``--runs``); excludes ``min_runs``/``max_runs``.
        setup: Command run once before each command's series of runs (``--setup``).
        prepare: Command run before every timing run (``--prepare``); a sequence gives
            one command per benchmarked command.
        conclude: Command run after every timing run (``--conclude``); a sequence gives
            one command per benchmarked command.
        cleanup: Command run after all runs of each command (``--cleanup``).
        parameter_scan: A numeric parameter range (``--parameter-scan``).
        parameter_lists: Mapping of placeholder name to values (``--parameter-list``);
            multiple entries benchmark every combination.
        shell_command: Shell used to run commands (``--shell``), e.g. ``"bash --norc"``;
            ``"default"`` selects the platform default. (Named so that it does not trip
            linters that flag ``shell=`` keyword arguments.)
        no_shell: Run commands directly, without an intermediate shell (``-N``).
        ignore_failure: ``True`` ignores all non-zero exit codes; a collection of ints
            ignores only those codes (``--ignore-failure``).
        command_names: Display name(s) for the commands (``--command-name``).
        reference: Reference command for the relative comparison (``--reference``).
        reference_name: Display name of the reference command (``--reference-name``).
        style: Output style (``--style``). Defaults to ``"none"`` unless ``stream`` is set.
        sort: Sort order of the summary (``--sort``).
        time_unit: Time unit for the terminal output and markup exports (``--time-unit``).
        stdin: File fed to the benchmarked commands' standard input (``--input``).
        output: Destination of the benchmarked commands' output (``--output``): one of
            ``"null"``, ``"pipe"``, ``"inherit"`` (or the :class:`Output` members), or a
            file given as a :class:`pathlib.Path` such as ``Path("out.log")``. Plain
            strings are reserved for the keywords, so a typo is an error rather than a
            file. A sequence gives one destination per benchmarked command.
        export_json: Also keep hyperfine's JSON export at this path (``--export-json``).
        export_csv: Write a CSV export to this path (``--export-csv``).
        export_markdown: Write a Markdown export to this path (``--export-markdown``).
        export_asciidoc: Write an AsciiDoc export to this path (``--export-asciidoc``).
        export_orgmode: Write an Emacs org-mode export to this path (``--export-orgmode``).
        extra_args: Additional raw arguments inserted before the commands.
        cwd: Working directory for hyperfine.
        env: Environment variables added to (and overriding) the current environment.
        timeout: Abort hyperfine after this many seconds.
        binary: Path or name of the hyperfine executable. Defaults to ``$HYPERFINE_BIN``
            or ``hyperfine`` on ``PATH``.
        stream: Let hyperfine's progress output through to the terminal instead of
            capturing it. Hyperfine's error output is then not captured either, so
            :attr:`BenchmarkFailedError.stderr` is empty.
    """

    warmup: int
    min_runs: int
    max_runs: int
    runs: int
    setup: str
    prepare: str | Sequence[str]
    conclude: str | Sequence[str]
    cleanup: str
    parameter_scan: ParameterScan
    parameter_lists: Mapping[str, Sequence[ParameterValue]]
    shell_command: str
    no_shell: bool
    ignore_failure: bool | Collection[int]
    command_names: str | Sequence[str]
    reference: str
    reference_name: str
    style: StyleLike
    sort: SortLike
    time_unit: TimeUnitLike
    stdin: StrPath
    output: OutputTarget | Sequence[OutputTarget]
    export_json: StrPath
    export_csv: StrPath
    export_markdown: StrPath
    export_asciidoc: StrPath
    export_orgmode: StrPath
    extra_args: Sequence[str]
    cwd: StrPath
    env: Mapping[str, str]
    timeout: float
    binary: StrPath
    stream: bool


type OptionName = Literal[
    "warmup",
    "min_runs",
    "max_runs",
    "runs",
    "setup",
    "prepare",
    "conclude",
    "cleanup",
    "parameter_scan",
    "parameter_lists",
    "shell_command",
    "no_shell",
    "ignore_failure",
    "command_names",
    "reference",
    "reference_name",
    "style",
    "sort",
    "time_unit",
    "stdin",
    "output",
    "export_json",
    "export_csv",
    "export_markdown",
    "export_asciidoc",
    "export_orgmode",
    "extra_args",
    "cwd",
    "env",
    "timeout",
    "binary",
    "stream",
]
"""The name of a :class:`BenchmarkOptions` key, as accepted by :meth:`Benchmark.without`."""


def _is_int(value: float) -> bool:
    return isinstance(value, int)


def _fmt_number(value: float) -> str:
    """Format a number the way hyperfine's decimal parser accepts it (never in exponent form).

    Example:
        >>> [_fmt_number(v) for v in (3, 0.1, 1e-05, 1e16)]
        ['3', '0.1', '0.00001', '10000000000000000']
    """
    if isinstance(value, int):
        return str(value)
    return format(Decimal(repr(value)), "f")


def check_placeholder_name(name: str, option: str) -> None:
    if not name or any(char in name for char in "{} \t\n,"):
        msg = f"{option}: invalid parameter name {name!r}"
        raise ValidationError(msg)
