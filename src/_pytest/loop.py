"""Prototype implementation of the marker-driven loop/rerun protocol.

See https://github.com/pytest-dev/pytest/pull/14779 for the design document.

A loop controller is attached to a test item via ``@pytest.mark.loop(controller)``
and drives repeated executions of the item, each with fresh function-scoped
fixtures, through :meth:`LoopContext.run_iteration`.
"""

from __future__ import annotations

from collections.abc import Mapping
import contextlib
import dataclasses
from typing import final
from typing import Literal
from typing import Protocol
from typing import TYPE_CHECKING
from typing import Union

from _pytest._code import ExceptionInfo
from _pytest.config import Config
from _pytest.config import hookimpl
from _pytest.outcomes import fail
from _pytest.reports import TestReport
from _pytest.stash import StashKey


if TYPE_CHECKING:
    from collections.abc import Generator

    from _pytest.nodes import Item
    from _pytest.runner import CallInfo
    from _pytest.terminal import TerminalReporter


LoopKey = Union[str, int, tuple["LoopKey", ...]]  # noqa: UP007 -- recursive alias needs Union on 3.11


def pytest_configure(config: Config) -> None:
    config.addinivalue_line(
        "markers",
        "loop(controller, provides=()): drive this test through a loop "
        "controller which may run it multiple times with fresh "
        "function-scoped fixtures per iteration (experimental prototype).",
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class LoopIteration:
    """Structured loop identity stamped on every report of an iteration."""

    key: LoopKey
    #: 0-based true execution counter (gaps in emitted output show discards).
    index: int
    #: Machine-filterable controller name, e.g. "hypothesis", "repeat".
    controller: str

    @property
    def key_repr(self) -> str:
        return _format_key(self.key)


def _format_key(key: LoopKey) -> str:
    if isinstance(key, tuple):
        text = "-".join(_format_key(k) for k in key)
    else:
        text = str(key)
    # Keep the derived "[key]" suffix parseable.
    return text.replace("]", "")


@final
@dataclasses.dataclass(frozen=True)
class IterationResult:
    """The outcome of one loop iteration."""

    key: LoopKey
    #: setup[, call], teardown reports, in execution order.
    reports: tuple[TestReport, ...]
    excinfo: ExceptionInfo[BaseException] | None
    when_failed: Literal["setup", "call", "teardown"] | None
    duration: float
    _context: LoopContext | None = dataclasses.field(
        default=None, repr=False, compare=False
    )

    @property
    def passed(self) -> bool:
        return all(rep.passed for rep in self.reports)

    @property
    def failed(self) -> bool:
        return self.when_failed is not None

    @property
    def skipped(self) -> bool:
        return any(rep.skipped for rep in self.reports)

    def emit(self) -> None:
        """Emit the buffered reports of a deferred iteration."""
        assert self._context is not None
        self._context.emit(self)


class LoopController(Protocol):
    def __call__(self, loop: LoopContext) -> None: ...  # pragma: no cover


class _ActiveIteration:
    """Book-keeping while one iteration's phases execute."""

    def __init__(self, iteration: LoopIteration, mode: str) -> None:
        self.iteration = iteration
        self.mode = mode
        self.excinfo_by_when: dict[str, ExceptionInfo[BaseException] | None] = {}


#: Stash slot holding the currently-active iteration of a looped item, if any.
active_iteration_key = StashKey["_ActiveIteration | None"]()


@final
class LoopContext:
    """Handed to the loop controller; runs iterations of one item."""

    def __init__(self, item: Item, controller_name: str, log: bool) -> None:
        self.item = item
        self.controller_name = controller_name
        self.iteration_count = 0
        self._log = log
        self._escaped = False
        self._emitted_keys: set[object] = set()
        self._any_failed = False
        self._any_emitted = False

    def run_iteration(
        self,
        key: LoopKey | None = None,
        *,
        funcargs: Mapping[str, object] | None = None,
        report: Literal["live", "defer", "discard"] = "live",
    ) -> IterationResult:
        from _pytest.runner import call_and_report

        if self._escaped:
            raise RuntimeError("run_iteration() called on an escaped LoopContext")
        if report not in ("live", "defer", "discard"):
            raise ValueError(f"invalid report mode: {report!r}")
        if key is None:
            key = self.iteration_count
        item = self.item
        live = report == "live"

        if self.iteration_count > 0:
            # Fresh funcargs + request (the rerunfailures contract), and a
            # fresh class instance for class-based tests.
            if getattr(item, "_instance", None) is not None:
                with contextlib.suppress(AttributeError):
                    del item._instance  # type: ignore[attr-defined]
                with contextlib.suppress(AttributeError):
                    del item._obj  # type: ignore[attr-defined]
            item._initrequest()  # type: ignore[attr-defined]

        active = _ActiveIteration(
            LoopIteration(
                key=key, index=self.iteration_count, controller=self.controller_name
            ),
            mode=report,
        )
        item.stash[active_iteration_key] = active
        self.iteration_count += 1
        reports: list[TestReport] = []
        try:
            rep_setup = call_and_report(item, "setup", live)
            reports.append(rep_setup)
            if rep_setup.passed:
                if funcargs:
                    item._loop_funcargs_overlay = dict(funcargs)  # type: ignore[attr-defined]
                try:
                    reports.append(call_and_report(item, "call", live))
                finally:
                    item._loop_funcargs_overlay = None  # type: ignore[attr-defined]
            # Intermediate teardown: the ``nextitem is item`` sentinel pops
            # only the item frame, leaving higher scopes set up.
            reports.append(call_and_report(item, "teardown", live, nextitem=item))
        finally:
            item.stash[active_iteration_key] = None

        when_failed: Literal["setup", "call", "teardown"] | None = None
        excinfo: ExceptionInfo[BaseException] | None = None
        for rep in reports:
            if rep.failed and when_failed is None:
                assert rep.when in ("setup", "call", "teardown")
                when_failed = rep.when
                excinfo = active.excinfo_by_when.get(rep.when)
        result = IterationResult(
            key=key,
            reports=tuple(reports),
            excinfo=excinfo,
            when_failed=when_failed,
            duration=sum(rep.duration for rep in reports),
            _context=self if report == "defer" else None,
        )
        if when_failed is not None:
            self._any_failed = True
        if live:
            self._record_emitted(key)
        return result

    def emit(self, result: IterationResult) -> None:
        """Replay a deferred iteration's reports through the report hooks."""
        if result._context is not self:
            raise RuntimeError("cannot emit a result from a different loop")
        self._record_emitted(result.key)
        if self._log:
            for rep in result.reports:
                self.item.ihook.pytest_runtest_logreport(report=rep)

    def _record_emitted(self, key: LoopKey) -> None:
        if key in self._emitted_keys:
            raise RuntimeError(f"loop key {key!r} emitted more than once")
        self._emitted_keys.add(key)
        self._any_emitted = True


def loop_runtestprotocol(
    item: Item, controller: LoopController, log: bool, nextitem: Item | None
) -> list[TestReport]:
    """Run the loop protocol for a marked item; called from runtestprotocol."""
    from _pytest.runner import call_and_report
    from _pytest.runner import CallInfo
    from _pytest.runner import check_interactive_exception
    from _pytest.runner import get_reraise_exceptions

    controller_name = str(
        getattr(controller, "name", None)
        or getattr(controller, "__name__", type(controller).__name__)
    )
    ctx = LoopContext(item, controller_name, log)

    def invoke_controller() -> None:
        controller(ctx)
        if ctx.iteration_count == 0 and not ctx._any_emitted:
            raise RuntimeError(
                "loop controller executed no iterations and emitted nothing; "
                "call pytest.skip() if there is legitimately nothing to run"
            )
        if ctx._any_failed:
            # Default verdict when the controller doesn't decide itself.
            fail(
                f"{ctx.controller_name}: one or more loop iterations failed",
                pytrace=False,
            )

    try:
        call: CallInfo[None] = CallInfo.from_call(
            invoke_controller,
            when="call",
            reraise=get_reraise_exceptions(item.config),
        )
    except BaseException:
        # Reraise set (KeyboardInterrupt & co): final teardown, then propagate.
        call_and_report(item, "teardown", log, nextitem=None)
        raise
    finally:
        ctx._escaped = True

    ihook = item.ihook
    parent_report: TestReport = ihook.pytest_runtest_makereport(item=item, call=call)
    parent_report.loop_summary = {  # type: ignore[attr-defined]
        "controller": ctx.controller_name,
        "iterations_run": ctx.iteration_count,
        "emitted": len(ctx._emitted_keys),
    }
    reports = [parent_report]
    if log:
        ihook.pytest_runtest_logreport(report=parent_report)
    if check_interactive_exception(call, parent_report):
        ihook.pytest_exception_interact(node=item, call=call, report=parent_report)

    if item.session.shouldfail or item.session.shouldstop:
        nextitem = None
    reports.append(call_and_report(item, "teardown", log, nextitem=nextitem))
    return reports


@hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: Item, call: CallInfo[None]
) -> Generator[None, TestReport, TestReport]:
    report = yield
    active = item.stash.get(active_iteration_key, None)
    if active is not None and isinstance(report, TestReport):
        report.loop = active.iteration  # type: ignore[attr-defined]
        active.excinfo_by_when[call.when] = call.excinfo
    return report


def pytest_terminal_summary(terminalreporter: TerminalReporter) -> None:
    reports = [
        rep
        for key in ("loop failed", "loop error")
        for rep in terminalreporter.stats.get(key, [])
        if rep.longrepr is not None
    ]
    if reports:
        terminalreporter.write_sep("=", "LOOP ITERATION FAILURES")
        for rep in reports:
            loop_info = rep.loop
            msg = f"{rep.nodeid}[{loop_info.key_repr}] ({rep.when})"
            terminalreporter.write_sep("_", msg, red=True, bold=True)
            terminalreporter._outrep_summary(rep)


@hookimpl(tryfirst=True)
def pytest_report_teststatus(report: TestReport) -> tuple[str, str, str] | None:
    loop_info = getattr(report, "loop", None)
    if loop_info is None or getattr(report, "context", None) is not None:
        # Not a loop iteration report, or a SubtestReport (whose own
        # teststatus hook must win).
        return None
    suffix = f"[{loop_info.key_repr}]"
    if report.when == "call":
        if report.failed:
            return "loop failed", "i", f"ITERFAIL{suffix}"
        if report.skipped:
            return "loop skipped", "-", f"ITERSKIP{suffix}"
        return "", "", ""
    else:
        if report.failed:
            return "loop error", "I", f"ITERERROR{suffix}"
        if report.skipped:
            return "loop skipped", "-", f"ITERSKIP{suffix}"
        return "", "", ""
