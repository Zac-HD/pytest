"""Tests for the marker-driven loop/rerun protocol prototype (_pytest.loop)."""

from __future__ import annotations

from _pytest.pytester import Pytester
import pytest


class TestLoopProtocol:
    def test_fresh_function_fixtures_per_iteration(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            events = []

            @pytest.fixture
            def fresh():
                events.append("setup")
                yield []
                events.append("teardown")

            @pytest.fixture(scope="module")
            def shared():
                events.append("module-setup")
                yield object()

            def repeat3(loop):
                for i in range(3):
                    loop.run_iteration(("rep", i))

            @pytest.mark.loop(controller=repeat3)
            def test_repeated(fresh, shared):
                assert fresh == []  # would fail if reused across iterations
                fresh.append(1)

            def test_after():
                assert events.count("setup") == 3
                assert events.count("teardown") == 3
                assert events.count("module-setup") == 1
            """
        )
        result = pytester.runpytest()
        result.assert_outcomes(passed=2)

    def test_live_iteration_failure_reporting(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            def repeat3(loop):
                for i in range(3):
                    loop.run_iteration(("rep", i))

            @pytest.mark.loop(controller=repeat3)
            def test_sometimes_fails(request):
                assert not hasattr(test_sometimes_fails, "ran")
                test_sometimes_fails.ran = True
            """
        )
        result = pytester.runpytest("-v")
        # Two failing iterations plus the parent verdict; totals not inflated.
        result.stdout.fnmatch_lines(
            [
                "*ITERFAIL?rep-1*",
                "*ITERFAIL?rep-2*",
                "*2 loop failed*",
            ]
        )
        result.assert_outcomes(failed=1)

    def test_iteration_reports_carry_loop_key(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            def repeat2(loop):
                for i in range(2):
                    loop.run_iteration(("rep", i))

            @pytest.mark.loop(controller=repeat2)
            def test_looped():
                pass
            """
        )
        result = pytester.runpytest_inprocess("-q")
        reports = [
            rep
            for rep in result.reprec.getreports("pytest_runtest_logreport")  # type: ignore[attr-defined]
            if rep.nodeid.endswith("test_looped")
        ]
        keyed = [rep for rep in reports if getattr(rep, "loop", None) is not None]
        parents = [rep for rep in reports if getattr(rep, "loop", None) is None]
        # 2 iterations x (setup, call, teardown)
        assert len(keyed) == 6
        assert {rep.loop.key for rep in keyed} == {("rep", 0), ("rep", 1)}
        assert {rep.loop.index for rep in keyed} == {0, 1}
        # loop-less parent call + final teardown, call carries the summary
        assert [rep.when for rep in parents] == ["call", "teardown"]
        assert parents[0].loop_summary["iterations_run"] == 2  # type: ignore[attr-defined]

    def test_deferred_reports_emitted_on_demand(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            def retry(loop):
                results = [loop.run_iteration(("attempt", i), report="defer")
                           for i in range(3)]
                results[-1].emit()  # only the deciding attempt

            @pytest.mark.loop(controller=retry)
            def test_retried():
                pass
            """
        )
        result = pytester.runpytest_inprocess("-q")
        reports = [
            rep
            for rep in result.reprec.getreports("pytest_runtest_logreport")  # type: ignore[attr-defined]
            if rep.nodeid.endswith("test_retried")
            and getattr(rep, "loop", None) is not None
        ]
        assert {rep.loop.key for rep in reports} == {("attempt", 2)}
        result.assert_outcomes(passed=1)

    def test_discard_mode_emits_nothing(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            def silent(loop):
                for i in range(5):
                    loop.run_iteration(i, report="discard")

            @pytest.mark.loop(controller=silent)
            def test_silent():
                pass
            """
        )
        result = pytester.runpytest_inprocess("-q")
        reports = [
            rep
            for rep in result.reprec.getreports("pytest_runtest_logreport")  # type: ignore[attr-defined]
            if getattr(rep, "loop", None) is not None
        ]
        assert reports == []
        result.assert_outcomes(passed=1)

    def test_funcargs_injection_with_provides(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            def inject(loop):
                loop.run_iteration("a", funcargs={"x": 1})
                loop.run_iteration("b", funcargs={"x": 2})

            seen = []

            @pytest.mark.loop(controller=inject, provides=("x",))
            def test_injected(x, tmp_path):
                seen.append((x, tmp_path))

            def test_after():
                assert [x for x, _ in seen] == [1, 2]
                # each iteration got its own tmp_path
                assert seen[0][1] != seen[1][1]
            """
        )
        result = pytester.runpytest()
        result.assert_outcomes(passed=2)

    def test_fixture_shadowing_via_funcargs(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            @pytest.fixture
            def value():
                return "fixture"

            def shadow(loop):
                loop.run_iteration("shadowed", funcargs={"value": "injected"})
                loop.run_iteration("plain")

            seen = []

            @pytest.mark.loop(controller=shadow)
            def test_shadowed(value):
                seen.append(value)

            def test_after():
                assert seen == ["injected", "fixture"]
            """
        )
        result = pytester.runpytest()
        result.assert_outcomes(passed=2)

    def test_fresh_class_instance_per_iteration(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            def repeat3(loop):
                for i in range(3):
                    loop.run_iteration(i)

            class TestClass:
                @pytest.mark.loop(controller=repeat3)
                def test_method(self):
                    assert not hasattr(self, "polluted")
                    self.polluted = True
            """
        )
        result = pytester.runpytest()
        result.assert_outcomes(passed=1)

    def test_zero_iterations_is_an_error(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            def lazy(loop):
                pass

            @pytest.mark.loop(controller=lazy)
            def test_never_run():
                pass
            """
        )
        result = pytester.runpytest()
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*executed no iterations*"])

    def test_controller_skip_propagates(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            def skipper(loop):
                pytest.skip("nothing to run")

            @pytest.mark.loop(controller=skipper)
            def test_skipped():
                pass
            """
        )
        result = pytester.runpytest()
        result.assert_outcomes(skipped=1)

    def test_duplicate_emitted_key_rejected(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            def dupe(loop):
                loop.run_iteration("same")
                loop.run_iteration("same")

            @pytest.mark.loop(controller=dupe)
            def test_dupe():
                pass
            """
        )
        result = pytester.runpytest()
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*emitted more than once*"])

    def test_unmarked_tests_unaffected(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            """
            def test_plain():
                pass
            """
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)


class TestHypothesisIntegration:
    """The flagship consumer: hypothesis driving generate/shrink/replay."""

    def test_shrinks_and_reports_minimal_example(self, pytester: Pytester) -> None:
        pytest.importorskip("hypothesis")
        pytester.makepyfile(
            """
            import pytest
            from hypothesis import strategies as st
            from _pytest.loop_hypothesis import given_loop

            setups = []

            @pytest.fixture
            def fresh_list():
                setups.append(1)
                lst = []
                yield lst
                # a stale fixture would carry earlier examples' mutations
                assert len(lst) <= 1

            @given_loop(x=st.integers())
            def test_shrinks(x, fresh_list):
                fresh_list.append(x)
                assert x < 100

            def test_fixture_fresh_per_example():
                assert len(setups) > 10
            """
        )
        result = pytester.runpytest()
        result.assert_outcomes(passed=1, failed=1)
        result.stdout.fnmatch_lines(
            [
                "*minimal failing example(s):*",
                "*x=100*",
                "*LOOP ITERATION FAILURES*",
                "*assert 100 < 100*",
            ]
        )

    def test_passing_property_is_silent(self, pytester: Pytester) -> None:
        pytest.importorskip("hypothesis")
        pytester.makepyfile(
            """
            from hypothesis import strategies as st
            from _pytest.loop_hypothesis import given_loop

            @given_loop(x=st.integers(), max_examples=20)
            def test_ok(x, tmp_path):
                assert isinstance(x, int)
            """
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)
        result.stdout.no_fnmatch_line("*ITERFAIL*")
