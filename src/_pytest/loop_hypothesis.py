"""Prototype deep Hypothesis integration built on the loop protocol.

This is a proof of concept for https://github.com/pytest-dev/pytest/pull/14779:
pytest owns the example loop (fresh function-scoped fixtures per example,
per-iteration reports), Hypothesis owns the policy (generation, shrinking,
replay). In a real integration this code would live in Hypothesis itself.

Two integration levels are provided:

**Full integration** (``GivenLoopController``): drives Hypothesis's real
orchestration -- ``StateForActualGivenExecution.run_engine()`` with the
example database, explicit ``@example`` decorators, ``@settings``, health checks,
shrinking, and Flaky detection all intact -- while routing every example
execution through ``LoopContext.run_iteration``:

- the ``@given`` wrapper's ``hypothesis.inner_test`` handle (read at call
  time, by design) is swapped for a shim that runs one loop iteration per
  example: fresh function-scoped fixtures via pytest's own setup/teardown,
  strategy-drawn arguments injected via the funcargs overlay, and the
  *undecorated* inner test as the call target;
- generation and shrinking iterations use ``report="discard"``;
- the minimal-counterexample replay (``execute_once(..., is_final=True)``)
  runs with ``report="live"``, so its reports are emitted and ``--pdb`` /
  ``pytest_exception_interact`` fire on the shrunk example;
- the falsifying-example exception raised by ``run_engine`` becomes the
  loop-less parent verdict, complete with Hypothesis's
  "Falsifying example:" exception notes.

Load this module as a plugin (``-p _pytest.loop_hypothesis``) and every
collected ``@given`` test is automatically driven through the loop
protocol -- no source changes. Hypothesis's own pytest plugin should be
disabled (``-p no:hypothesispytest``): its in-call engine integration is
what this replaces.

**Standalone mini-integration** (``given_loop``): a self-contained
decorator driving ``ConjectureRunner`` directly, kept as a minimal
demonstration of the controller API without the ``@given`` machinery.

Known gaps of the prototype (all fixable in a real integration):

- Hypothesis's ``deadline`` now measures the whole iteration including
  fixture setup/teardown, not just the test call.
- ``st.data()`` draws happen mid-call inside the engine's context and
  work for generation/shrinking, but the drawn ``DataObject`` cannot be
  re-injected meaningfully by other controllers.
- Class-based tests (``self``) are not supported by the auto-marking.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Generator
import contextlib
import functools
import itertools
from typing import Any
from typing import TYPE_CHECKING

from _pytest.python import Function
import pytest


if TYPE_CHECKING:
    from _pytest.loop import IterationResult
    from _pytest.loop import LoopContext


class _AbortLoop(BaseException):
    """Escape hatch out of hypothesis's engine for non-example outcomes.

    Derives from BaseException so the engine does not treat it as a test
    failure to shrink; the controller catches it and re-raises the cause.
    """

    def __init__(self, cause: BaseException | None) -> None:
        self.cause = cause


class GivenLoopController:
    """A LoopController driving a real ``@given`` test's full machinery."""

    name = "hypothesis"

    def __call__(self, loop: LoopContext) -> None:
        from hypothesis.core import StateForActualGivenExecution

        item = loop.item
        wrapped = item.obj
        handle = wrapped.hypothesis  # HypothesisHandle; read at call time
        inner_test = handle.inner_test
        fixture_names = frozenset(item._fixtureinfo.argnames)  # type: ignore[attr-defined]
        counter = itertools.count()
        # Flipped to "live" by the execute_once patch for the final
        # (minimal-counterexample) replay.
        mode = {"report": "discard"}
        # Clean per-iteration tracebacks of live failures, keyed by
        # exception id: propagating back out through the engine extends
        # the exception's traceback with engine frames, so the original
        # is restored when the exception becomes the parent verdict.
        live_tracebacks: dict[int, Any] = {}

        @functools.wraps(inner_test)
        def shim(*args: Any, **kwargs: Any) -> None:
            if args:
                raise NotImplementedError(
                    "loop-protocol hypothesis prototype only supports "
                    "keyword arguments (no self/positional)"
                )
            # Hypothesis merges the (dummy) fixture kwargs we passed to the
            # wrapper with the strategy-drawn arguments; keep only the
            # drawn ones. Real fixture values are injected fresh by
            # pytest inside the iteration.
            drawn = {k: v for k, v in kwargs.items() if k not in fixture_names}
            report = mode["report"]
            prefix = "minimal" if report == "live" else "gen"
            result: IterationResult = loop.run_iteration(
                (prefix, next(counter)), funcargs=drawn, report=report
            )
            if result.skipped or result.when_failed == "setup":
                # Skips apply to the whole item; fixture errors are
                # example-independent. Abort the engine either way.
                raise _AbortLoop(
                    result.excinfo.value if result.excinfo is not None else None
                )
            if result.failed:
                assert result.excinfo is not None
                exc = result.excinfo.value
                if report == "live":
                    live_tracebacks[id(exc)] = exc.__traceback__
                # Re-raise inside the engine: interestingness, dedup by
                # InterestingOrigin, deadline and Flaky handling all see
                # the genuine exception.
                raise exc

        @contextlib.contextmanager
        def patched_execute_once() -> Generator[None]:
            orig = StateForActualGivenExecution.execute_once

            def execute_once(
                self: Any, data: Any, *, is_final: bool = False, **kw: Any
            ) -> Any:
                mode["report"] = "live" if is_final else "discard"
                try:
                    return orig(self, data, is_final=is_final, **kw)
                finally:
                    mode["report"] = "discard"

            StateForActualGivenExecution.execute_once = execute_once  # type: ignore[method-assign]
            try:
                yield
            finally:
                StateForActualGivenExecution.execute_once = orig  # type: ignore[method-assign]

        handle.inner_test = shim
        item._loop_call_target = inner_test  # type: ignore[attr-defined]
        try:
            with patched_execute_once():
                # Dummy values satisfy the wrapper's signature binding; the
                # shim strips them back out of every example call.
                wrapped(**dict.fromkeys(fixture_names))
        except _AbortLoop as abort:
            if abort.cause is not None:
                raise abort.cause from None
            pytest.skip("hypothesis loop iteration was skipped")
        except BaseException as exc:
            # The falsifying-example exception (with hypothesis's notes
            # attached) becomes the parent verdict; restore the clean
            # traceback captured when the live iteration ran.
            clean_tb = live_tracebacks.get(id(exc))
            if clean_tb is not None:
                raise exc.with_traceback(clean_tb) from None
            raise
        finally:
            handle.inner_test = inner_test
            item._loop_call_target = None  # type: ignore[attr-defined]


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-drive every collected ``@given`` test through the loop protocol."""
    for item in items:
        if (
            isinstance(item, Function)
            and getattr(item.obj, "hypothesis", None) is not None
            and item.get_closest_marker("loop") is None
        ):
            item.add_marker(pytest.mark.loop(controller=GivenLoopController()))


class HypothesisLoopController:
    """A LoopController driving Hypothesis's ConjectureRunner directly.

    The standalone mini-integration: no ``@given``, no database, no
    explicit examples -- just generate/shrink/replay through the loop
    protocol, used by :func:`given_loop`.
    """

    name = "hypothesis"

    def __init__(self, strategies: dict[str, Any], max_examples: int = 100) -> None:
        self.strategies = strategies
        self.max_examples = max_examples

    def __call__(self, loop: LoopContext) -> None:
        import random

        from hypothesis import HealthCheck
        from hypothesis import settings
        from hypothesis import strategies as st
        from hypothesis.control import BuildContext
        from hypothesis.internal.conjecture.data import ConjectureData
        from hypothesis.internal.conjecture.engine import ConjectureRunner
        from hypothesis.internal.escalation import InterestingOrigin

        strategy = st.fixed_dictionaries(self.strategies)
        counter = iter(range(10**9))
        skip_outcome: BaseException | None = None

        def draw_kwargs(data: ConjectureData) -> dict[str, Any]:
            with BuildContext(data, wrapped_test=lambda: None):
                kwargs: dict[str, Any] = data.draw(strategy)
            return kwargs

        def test_function(data: ConjectureData) -> None:
            nonlocal skip_outcome
            kwargs = draw_kwargs(data)
            result = loop.run_iteration(
                ("gen", next(counter)), funcargs=kwargs, report="discard"
            )
            if result.skipped:
                # A skip applies to the whole item; stop generating.
                skip_outcome = (
                    result.excinfo.value if result.excinfo is not None else None
                )
                data.mark_invalid()
            if result.failed:
                assert result.excinfo is not None
                data.mark_interesting(
                    InterestingOrigin.from_exception(result.excinfo.value)
                )

        engine_settings = settings(
            database=None,
            max_examples=self.max_examples,
            deadline=None,
            suppress_health_check=list(HealthCheck),
        )
        runner = ConjectureRunner(
            test_function, settings=engine_settings, random=random.Random(0)
        )
        runner.run()

        if skip_outcome is not None:
            raise skip_outcome

        if runner.interesting_examples:
            # Replay each minimal (shrunk) failing example live: fresh
            # fixtures once more, reports emitted, --pdb fires here.
            falsifying: list[dict[str, Any]] = []
            for n, result in enumerate(runner.interesting_examples.values()):
                data = ConjectureData.for_choices(result.choices)
                kwargs = draw_kwargs(data)
                falsifying.append(kwargs)
                loop.run_iteration(("minimal", n), funcargs=kwargs, report="live")
            reprs = "\n".join(
                "  " + ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
                for kwargs in falsifying
            )
            pytest.fail(
                f"hypothesis: falsified after {loop.iteration_count} examples; "
                f"minimal failing example(s):\n{reprs}",
                pytrace=False,
            )


def given_loop(
    *, max_examples: int = 100, **strategies: Any
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """Decorator: property-based testing via the pytest loop protocol.

    Unlike ``@given``, the wrapped function is left untouched -- pytest
    calls it once per example with fresh function-scoped fixtures, and the
    strategy-provided arguments are injected per-iteration via the loop
    mark's ``provides=`` mechanism.
    """
    controller = HypothesisLoopController(strategies, max_examples=max_examples)
    mark = pytest.mark.loop(controller=controller, provides=tuple(strategies))

    def decorate(func: Callable[..., None]) -> Callable[..., None]:
        marked: Callable[..., None] = mark(func)
        return marked

    return decorate
