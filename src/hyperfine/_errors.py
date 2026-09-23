"""Exception hierarchy for :mod:`hyperfine`."""

from __future__ import annotations

import errno
from functools import partial
from typing import TYPE_CHECKING, Self, override

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "AmbiguousCommandError",
    "BenchmarkFailedError",
    "BenchmarkTimeoutError",
    "HyperfineError",
    "HyperfineNotFoundError",
    "ReportParseError",
    "ValidationError",
]

_INSTALL_HINT = (
    "Install it from https://github.com/sharkdp/hyperfine#installation "
    "(e.g. `brew install hyperfine`, `cargo install hyperfine` or `apt install hyperfine`), "
    "or point the HYPERFINE_BIN environment variable / the `binary=` argument at the executable."
)


class HyperfineError(Exception):
    """Base class for every error raised by this package."""


class ValidationError(HyperfineError, ValueError):
    """Raised before spawning hyperfine when the requested options are invalid or conflicting."""


class ReportParseError(HyperfineError, ValueError):
    """Raised when hyperfine JSON output cannot be parsed into a report."""


class AmbiguousCommandError(HyperfineError, KeyError):
    """Raised when a command name lookup matches more than one result.

    Happens when the same command is benchmarked twice, or when ``reference=`` names
    a command that is also benchmarked (hyperfine then reports it twice). Look the
    results up by index, or use :meth:`BenchmarkReport.find_all`.

    Attributes:
        command: The command name that was looked up.
        indices: Positions of every result with that command name.
    """

    command: str
    indices: tuple[int, ...]
    _message: str

    def __init__(self, *, command: str, indices: Sequence[int]) -> None:
        """Create the error for an ambiguous command name.

        Args:
            command: The command name that was looked up.
            indices: Positions of every result with that command name.
        """
        self.command = command
        self.indices = tuple(indices)
        positions = ", ".join(map(str, self.indices))
        self._message = (
            f"{len(self.indices)} results share the command {command!r} (at indices "
            f"{positions}); look them up by index or with find_all()"
        )
        super().__init__(self._message)

    @override
    def __str__(self) -> str:
        # KeyError.__str__ would repr() the message.
        return self._message

    @override
    def __reduce__(self) -> tuple[partial[Self], tuple[()]]:
        return (partial(type(self), command=self.command, indices=self.indices), ())


class HyperfineNotFoundError(HyperfineError, FileNotFoundError):
    """Raised when the hyperfine executable cannot be located or executed.

    Attributes:
        binary: The executable name or path that was looked up.
    """

    binary: str

    def __init__(self, binary: str) -> None:
        """Create the error for the executable that could not be found.

        Args:
            binary: The executable name or path that was looked up.
        """
        self.binary = binary
        message = f"hyperfine executable not found: {binary!r}. {_INSTALL_HINT}"
        super().__init__(errno.ENOENT, message, binary)

    @override
    def __str__(self) -> str:
        return self.strerror or ""

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[str]]:
        return (type(self), (self.binary,))


class BenchmarkFailedError(HyperfineError):
    """Raised when hyperfine exits with a non-zero status.

    Attributes:
        returncode: Exit status of the hyperfine process.
        stderr: Captured standard error (empty when output was streamed).
        argv: The full argument vector that was executed.
        streamed: Whether hyperfine's output went to the terminal (``stream=True``)
            instead of being captured.
    """

    returncode: int
    stderr: str
    argv: tuple[str, ...]
    streamed: bool

    def __init__(
        self, *, returncode: int, stderr: str, argv: Sequence[str], streamed: bool = False
    ) -> None:
        """Create the error from the failed process details.

        Args:
            returncode: Exit status of the hyperfine process.
            stderr: Captured standard error (empty when output was streamed).
            argv: The full argument vector that was executed.
            streamed: Whether hyperfine's output was streamed rather than captured.
        """
        self.returncode = returncode
        self.stderr = stderr
        self.argv = tuple(argv)
        self.streamed = streamed
        super().__init__(self._describe())

    @override
    def __reduce__(self) -> tuple[partial[Self], tuple[()]]:
        rebuild = partial(
            type(self),
            returncode=self.returncode,
            stderr=self.stderr,
            argv=self.argv,
            streamed=self.streamed,
        )
        return (rebuild, ())

    @property
    def reason(self) -> str:
        """The most relevant line of hyperfine's error output, if any."""
        lines = [line.strip() for line in self.stderr.splitlines() if line.strip()]
        for line in reversed(lines):
            if line.lower().startswith("error"):
                return line
        return lines[-1] if lines else ""

    def _describe(self) -> str:
        message = f"hyperfine exited with status {self.returncode}"
        reason = self.reason
        if reason:
            return f"{message}: {reason}"
        if self.streamed:
            return f"{message} (stderr was streamed to the terminal, see its output above)"
        return message


class BenchmarkTimeoutError(HyperfineError, TimeoutError):
    """Raised when the hyperfine process exceeds the configured ``timeout``.

    Attributes:
        timeout: The timeout in seconds that was exceeded.
        argv: The full argument vector that was executed.
    """

    timeout: float
    argv: tuple[str, ...]

    def __init__(self, *, timeout: float, argv: Sequence[str]) -> None:
        """Create the error for a timed out run.

        Args:
            timeout: The timeout in seconds that was exceeded.
            argv: The full argument vector that was executed.
        """
        self.timeout = timeout
        self.argv = tuple(argv)
        super().__init__(f"hyperfine did not finish within {timeout:g} seconds")

    @override
    def __reduce__(self) -> tuple[partial[Self], tuple[()]]:
        return (partial(type(self), timeout=self.timeout, argv=self.argv), ())
