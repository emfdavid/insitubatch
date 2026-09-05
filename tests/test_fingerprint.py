"""Unit tests for the cross-run cache fingerprint: ``_transform_token`` per transform and
``_pipeline_hash`` per array.

The round-trip behaviour these underpin -- a reopened pool hitting or raising -- is covered
in ``test_pool.py`` (``cache_key`` precedence, cloudpickle vs source, an edited body) and
``test_transform_scope.py`` (which arrays a transform edit invalidates). What is pinned here
is the hash itself, because every one of these properties is a claim the docstrings make and
a silent wrong-data bug if it stops holding.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from insitubatch.pool import _pipeline_hash, _transform_token, transforms_for
from insitubatch.transforms import StandardScaler, applies


@pytest.fixture
def source_path(monkeypatch):
    """Force the best-effort source hash -- the fallback when cloudpickle is absent."""
    monkeypatch.setattr("insitubatch.pool.cloudpickle", None)


def scale(chunk):
    chunk.data = chunk.data * 2.0
    return chunk


def shift(chunk):
    chunk.data = chunk.data + 1.0
    return chunk


@dataclass
class Coarsen:
    """A configured, class-based transform with a stable dataclass repr."""

    factor: int = 2

    def __call__(self, chunk):
        return chunk


class Plain:
    """A callable instance with the *default* repr -- the one carrying a memory address."""

    def __call__(self, chunk):
        return chunk


# -- one transform's token --------------------------------------------------


def test_cache_key_outranks_everything(source_path):
    """The declared key is the whole token: nothing about the body or config can override it."""

    def declared(chunk):
        return chunk

    declared.cache_key = "v1"
    assert _transform_token(declared) == "key:v1"


def test_scope_is_not_in_the_token():
    """`applies` is stripped before hashing. Two scopes of one transform are one identity --
    which is what lets narrowing a scope invalidate the variable that left and no other."""
    bare = _transform_token(scale)
    assert _transform_token(applies(["a"], scale)) == bare
    assert _transform_token(applies(["b", "c"], scale)) == bare


def test_a_callable_instance_hashes_by_class_not_by_address(source_path):
    """Two separately constructed instances must hash alike, or every reopen is a spurious
    miss: the default ``repr`` embeds an address that differs per object and per process."""
    assert _transform_token(Plain()) == _transform_token(Plain())


def test_instance_config_still_reaches_the_token(source_path):
    """A stable dataclass repr is folded in, so a reconfigured transform is a different one."""
    assert _transform_token(Coarsen(2)) == _transform_token(Coarsen(2))
    assert _transform_token(Coarsen(2)) != _transform_token(Coarsen(3))


def test_the_hashing_method_is_encoded_in_the_token(monkeypatch):
    """Toggling cloudpickle re-computes rather than falsely matching: the two methods cannot
    collide, because the token carries which one produced it."""
    pytest.importorskip("cloudpickle")
    pickled = _transform_token(scale)
    monkeypatch.setattr("insitubatch.pool.cloudpickle", None)
    sourced = _transform_token(scale)
    assert pickled.startswith("pickle:") and sourced.startswith("src:")
    assert pickled != sourced


def test_large_statistics_are_invisible_to_the_source_fallback(source_path):
    """A documented hole, pinned so it cannot widen unnoticed (docs/architecture.md).

    numpy summarizes an array over 1000 elements in ``repr``, so a re-fitted scaler carrying
    per-gridpoint stats hashes identically to the values it replaced -- a persisted cache
    reopens as a hit, serving the old normalization. An explicit ``cache_key`` closes it.
    """
    old = np.arange(2000.0)
    new = old.copy()
    new[500] = 999.0
    ones = {"a": np.ones(1)}
    assert repr(old) == repr(new), "numpy stopped summarizing -- re-check the docs claim"
    assert _transform_token(StandardScaler({"a": old}, ones)) == _transform_token(
        StandardScaler({"a": new}, ones)
    )
    assert _transform_token(
        StandardScaler({"a": old}, ones, cache_key="fit-1")
    ) != _transform_token(StandardScaler({"a": new}, ones, cache_key="fit-2"))


def test_cloudpickle_sees_the_statistics_the_repr_hides():
    """The recommended fix works: cloudpickle hashes the values, not their summary."""
    pytest.importorskip("cloudpickle")
    old = np.arange(2000.0)
    new = old.copy()
    new[500] = 999.0
    ones = {"a": np.ones(1)}
    assert _transform_token(StandardScaler({"a": old}, ones)) != _transform_token(
        StandardScaler({"a": new}, ones)
    )


# -- one array's pipeline hash ----------------------------------------------


def test_pipeline_hash_is_ordered():
    """Composition is not commutative, so the identity of the pipeline must not be either."""
    assert _pipeline_hash([scale, shift]) != _pipeline_hash([shift, scale])


def test_pipeline_hash_is_stable_and_fixed_width():
    """16 hex chars, deterministic -- the manifest's ``pipeline`` field is compared verbatim."""
    once = _pipeline_hash([scale])
    assert once == _pipeline_hash([scale])
    assert len(once) == 16 and len(_pipeline_hash([])) == 16


def test_an_empty_pipeline_is_an_identity_like_any_other():
    """An array no transform touches hashes to a stable value distinct from any pipeline's:
    it invalidates once when it leaves every scope, then stays valid across reopens."""
    assert _pipeline_hash([]) == _pipeline_hash([])
    assert _pipeline_hash([]) != _pipeline_hash([scale])


def test_scope_reaches_the_hash_through_membership_not_the_token():
    """The two halves together, through the one function that resolves scope: membership
    decides *which* transforms enter an array's hash, and each token is scope-free. Narrowing
    a scope therefore changes the hash of the array that left it, and of no other."""
    wide = [applies(["a", "b"], scale), shift]
    narrow = [applies(["a"], scale), shift]
    assert _pipeline_hash(transforms_for("a", wide)) == _pipeline_hash(
        transforms_for("a", narrow)
    ), "a kept the same transforms -- its cached bytes are still valid"
    assert _pipeline_hash(transforms_for("b", wide)) != _pipeline_hash(
        transforms_for("b", narrow)
    ), "b lost one -- its cached bytes were transformed by a pipeline no longer configured"
