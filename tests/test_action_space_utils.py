"""Regression tests for the normal-equation solvers in ``action_space/utils.py``.

``solve_xs_eq_y`` retries factorization with a growing ridge while the system is not
positive definite. Two failure modes made that retry loop spin forever:

* a ROCm PyTorch wheel built without MAGMA raises a plain ``RuntimeError`` on CUDA
  tensors no matter how many times it is retried, and
* ``ridge *= 10`` never changes a ridge of ``0.0`` (the function default).

The solvers must instead fall back to CPU when the backend refuses to factor, and give
up with ``LinAlgError`` after a bounded number of ridge retries.
"""

import threading

import pytest
import torch

from alpamayo2_edge.action_space.utils import (
    _MAX_RIDGE_ATTEMPTS,
    _cholesky_solve_or_cpu,
    solve_xs_eq_y,
)

MAGMA_ERROR = (
    "Calling torch.linalg.cholesky on a CUDA tensor requires compiling PyTorch with MAGMA."
)


def _spd(n: int, seed: int = 0) -> torch.Tensor:
    """A well-conditioned symmetric positive-definite matrix."""
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=gen)
    return a @ a.T + n * torch.eye(n)


def _call_with_timeout(fn, timeout: float = 10.0):
    """Run ``fn`` in a daemon thread and fail the test if it never returns.

    A reverted retry loop would hang the whole suite instead of failing, so the
    termination property itself is what is asserted here.
    """
    box: dict[str, object] = {}

    def target() -> None:
        try:
            box["result"] = fn()
        except BaseException as exc:  # re-raised in the test thread below
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), f"solver did not terminate within {timeout}s"
    if "error" in box:
        raise box["error"]
    return box["result"]


def test_singular_system_with_zero_ridge_terminates():
    """``ridge=0.0`` (the default) must not loop forever on a singular system."""
    s = torch.zeros(4)
    y = torch.ones(4)

    x = _call_with_timeout(
        lambda: solve_xs_eq_y(
            s, y, w_smooth1=None, w_smooth2=None, w_smooth3=None, lam=0.0, ridge=0.0
        )
    )

    assert x.shape == (4,)
    assert torch.isfinite(x).all()


def test_factorization_failure_is_capped(monkeypatch):
    """An unsolvable system raises after a bounded number of retries, not forever."""
    calls = {"n": 0}

    def always_singular(A, *args, **kwargs):
        calls["n"] += 1
        raise torch.linalg.LinAlgError("not positive definite")

    monkeypatch.setattr(torch.linalg, "cholesky", always_singular)

    with pytest.raises(torch.linalg.LinAlgError, match="Failed to factorize"):
        _call_with_timeout(
            lambda: solve_xs_eq_y(
                torch.ones(4),
                torch.ones(4),
                w_smooth1=None,
                w_smooth2=1.0,
                w_smooth3=None,
                lam=1e-4,
                ridge=1e-4,
            )
        )

    assert calls["n"] == _MAX_RIDGE_ATTEMPTS


def test_backend_runtime_error_falls_back_to_cpu(monkeypatch):
    """A MAGMA-style ``RuntimeError`` is retried on CPU and yields the right answer."""
    real_cholesky = torch.linalg.cholesky
    calls = {"n": 0}

    def magma_missing(A, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError(MAGMA_ERROR)
        return real_cholesky(A, *args, **kwargs)

    lhs = _spd(5)
    rhs = torch.randn(5, 1)
    expected = torch.cholesky_solve(rhs, real_cholesky(lhs))

    monkeypatch.setattr(torch.linalg, "cholesky", magma_missing)
    solved = _cholesky_solve_or_cpu(lhs, rhs)

    assert calls["n"] == 2, "expected one failed attempt plus one CPU retry"
    assert torch.allclose(solved, expected)


def test_linalg_error_propagates_without_cpu_retry(monkeypatch):
    """A genuinely singular input raises immediately instead of being masked."""
    calls = {"n": 0}

    def singular(A, *args, **kwargs):
        calls["n"] += 1
        raise torch.linalg.LinAlgError("not positive definite")

    monkeypatch.setattr(torch.linalg, "cholesky", singular)

    with pytest.raises(torch.linalg.LinAlgError, match="not positive definite"):
        _cholesky_solve_or_cpu(_spd(5), torch.randn(5, 1))

    assert calls["n"] == 1
