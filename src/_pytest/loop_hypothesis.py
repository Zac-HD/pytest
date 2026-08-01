"""Prototype deep Hypothesis integration built on the loop protocol.

This is a proof of concept for https://github.com/pytest-dev/pytest/pull/14779:
pytest owns the example loop (fresh function-scoped fixtures per example,
per-iteration reports), Hypothesis owns the policy (generation, shrinking,
replay). In a real integration this module would live in Hypothesis itself
and drive its full engine; here we drive ``ConjectureRunner`` directly with
just enough of the surrounding machinery to demonstrate the protocol:

- every generated example runs through ``LoopContext.run_iteration`` with
  ``report="discard"`` -- a fresh fixture cycle per example, no report spam;
- on failure the engine shrinks (still silent, still fresh fixtures);
- the minimal example is replayed with ``report="live"``, so ``--pdb`` and
  ``pytest_exception_interact`` fire on the shrunk counterexample;
- the parent verdict is a ``pytest.fail()`` naming the minimal example.

Usage::

    from _pytest.loop_hypothesis import given_loop


    @given_loop(x=st.integers(), y=st.integers())
    def test_foo(x, y, tmp_path): ...

Requires ``hypothesis`` to be installed; it is imported lazily.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from _pytest.loop import LoopContext


class HypothesisLoopController:
    """A LoopController driving Hypothesis's ConjectureRunner."""

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
