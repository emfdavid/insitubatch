"""The transform-development CLI: geometry report, output_inner validation, target
resolution, and the GIL-release probe plumbing.

The GIL probe is timing-based, so we assert its *plumbing* (fields + a verdict) directly and
keep the main() integration tests on --no-gil-probe (deterministic exit codes), never a hard
speedup threshold for the held case (flaky on shared CI / variable core counts).
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from insitubatch import ensure_local_dir, obstore_store, open_geometries
from insitubatch.check_transform import gil_probe, load_transform, main, read_chunk
from insitubatch.types import ChunkRead, DecodedChunk

# Geometry + output_inner validation are deterministic; skip the timing-based GIL probe so
# the exit code reflects only the validation (the probe is covered by gil_probe() unit tests).
FAST = ["--no-gil-probe"]

TRANSFORMS_SRC = """
from dataclasses import dataclass

import numpy as np

@dataclass
class Coarsen:  # a dataclass reshaping transform: needs its module registered in sys.modules
    factor: int = 2
    def __call__(self, chunk):
        chunk.data = chunk.data[:, :: self.factor, :: self.factor]
        return chunk
    def output_inner(self, geom):
        lat, lon = geom.inner_shape
        return (-(-lat // self.factor), -(-lon // self.factor)), geom.dtype

class MeanLastAxis:
    def __call__(self, chunk):
        chunk.data = chunk.data.mean(axis=-1)
        return chunk
    def output_inner(self, geom):
        return geom.inner_shape[:-1], geom.dtype

class NeedsArgs:
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, chunk):
        chunk.data = chunk.data * self.factor
        return chunk

def vectorized_scale(chunk):
    chunk.data = chunk.data * 2.0
    return chunk

def reshape_no_declare(chunk):       # reshapes but no output_inner -> not cacheable
    chunk.data = chunk.data.mean(axis=-1)
    return chunk

def wrong_declare(chunk):
    chunk.data = chunk.data.mean(axis=-1)
    return chunk
wrong_declare.output_inner = lambda geom: ((999,), geom.dtype)  # lies about the shape

not_callable = 42
"""


@pytest.fixture
def url(tmp_path):
    url = f"file://{tmp_path}/syn.zarr"
    ensure_local_dir(url)
    g = zarr.open_group(store=obstore_store(url, read_only=False), mode="w")
    arr = g.create_array("t2m", shape=(24, 6, 8), chunks=(8, 6, 8), dtype="f4")
    arr[:] = np.random.default_rng(0).standard_normal((24, 6, 8)).astype("f4")
    return url


@pytest.fixture
def transforms_file(tmp_path):
    p = tmp_path / "xf.py"
    p.write_text(TRANSFORMS_SRC)
    return p


# -- target resolution ------------------------------------------------------


def test_load_transform_from_file(transforms_file):
    fn = load_transform(f"{transforms_file}:vectorized_scale")
    chunk = DecodedChunk(read=ChunkRead("v", 0), data=np.ones((2, 3)), sample_offset=0)
    assert np.array_equal(fn(chunk).data, np.full((2, 3), 2.0))


def test_load_transform_instantiates_zero_arg_class(transforms_file):
    fn = load_transform(f"{transforms_file}:MeanLastAxis")
    assert not isinstance(fn, type) and callable(fn)  # an instance, not the class
    assert hasattr(fn, "output_inner")


def test_load_transform_dataclass_from_file(transforms_file):
    """A @dataclass transform loaded from a file needs its module registered in sys.modules
    (dataclasses resolves annotations via sys.modules[__module__]); regression for that."""
    fn = load_transform(f"{transforms_file}:Coarsen")
    assert callable(fn) and hasattr(fn, "output_inner")
    chunk = DecodedChunk(read=ChunkRead("v", 0), data=np.ones((2, 6, 8)), sample_offset=0)
    assert fn(chunk).data.shape == (2, 3, 4)  # strided by factor=2


def test_load_transform_from_module(transforms_file, monkeypatch):
    monkeypatch.syspath_prepend(str(transforms_file.parent))
    fn = load_transform(f"{transforms_file.stem}:vectorized_scale")  # dotted module:attr
    assert callable(fn)


def test_load_transform_class_needing_args_is_a_clear_error(transforms_file):
    with pytest.raises(TypeError, match="constructor needs arguments|configured instance"):
        load_transform(f"{transforms_file}:NeedsArgs")


@pytest.mark.parametrize(
    "target, exc",
    [
        ("no_colon_here", ValueError),
        ("./missing_file.py:fn", FileNotFoundError),
        ("os:does_not_exist", AttributeError),
    ],
)
def test_load_transform_bad_targets(target, exc):
    with pytest.raises(exc):
        load_transform(target)


def test_load_transform_non_callable(transforms_file):
    with pytest.raises(TypeError, match="not callable"):
        load_transform(f"{transforms_file}:not_callable")


# -- one-chunk read ---------------------------------------------------------


def test_read_chunk_matches_source(url):
    geom = open_geometries(obstore_store(url), variables=["t2m"])["t2m"]
    chunk = read_chunk(obstore_store(url), "t2m", 1, geom)
    assert chunk.data.shape == (8, 6, 8) and chunk.sample_offset == 8


# -- main() integration: exit codes + report --------------------------------


def test_main_passes_for_correct_reshaping_transform(url, transforms_file, capsys):
    rc = main([url, "--var", "t2m", "--transform", f"{transforms_file}:MeanLastAxis", *FAST])
    out = capsys.readouterr().out
    assert rc == 0
    assert "8/chunk" in out and "(6, 8)" in out  # geometry line
    assert "-> (6,)" in out and "[OK]" in out  # declared output validated


def test_main_fails_on_wrong_output_inner(url, transforms_file, capsys):
    rc = main([url, "--var", "t2m", "--transform", f"{transforms_file}:wrong_declare", *FAST])
    out = capsys.readouterr().out
    assert rc == 1
    assert "MISMATCH" in out and "_persist would raise" in out


def test_main_fails_on_undeclared_reshape(url, transforms_file, capsys):
    rc = main([url, "--var", "t2m", "--transform", f"{transforms_file}:reshape_no_declare", *FAST])
    out = capsys.readouterr().out
    assert rc == 1
    assert "does not declare output_inner" in out


def test_main_bad_transform_returns_2(url, capsys):
    rc = main([url, "--var", "t2m", "--transform", "nope_no_colon", *FAST])
    assert rc == 2
    assert "could not load --transform" in capsys.readouterr().err


def test_main_store_auth_flags_pass_through(url, transforms_file, capsys):
    """--skip-signature / --request-payer thread into the url open (so check_transform
    reaches the same public / Requester-Pays stores the loader does)."""
    rc = main(
        [
            url,
            "--var",
            "t2m",
            "--transform",
            f"{transforms_file}:vectorized_scale",
            "--skip-signature",
            "--request-payer",
            *FAST,
        ]
    )
    assert rc == 0


def test_main_chunk_out_of_range(url, transforms_file, capsys):
    rc = main(
        [
            url,
            "--var",
            "t2m",
            "--transform",
            f"{transforms_file}:vectorized_scale",
            "--chunk",
            "99",
            *FAST,
        ]
    )
    assert rc == 2
    assert "out of range" in capsys.readouterr().err


# -- GIL probe plumbing -----------------------------------------------------


def test_gil_probe_reports_fields_and_classifies_vectorized(url, transforms_file):
    geom = open_geometries(obstore_store(url), variables=["t2m"])["t2m"]
    base = read_chunk(obstore_store(url), "t2m", 0, geom)
    fn = load_transform(f"{transforms_file}:vectorized_scale")
    out = gil_probe(fn, base, threads=2, min_seconds=0.02, repeats=1)
    assert {"iters", "serial_s", "parallel_s", "speedup", "per_call_ms", "mb_s"} <= out.keys()
    assert out["iters"] >= 1 and out["serial_s"] > 0 and out["speedup"] > 0


def test_gil_probe_single_thread_speedup_about_one(url, transforms_file):
    # threads=1: serial and parallel are the same work -> speedup ~1 (no scaling claimed).
    geom = open_geometries(obstore_store(url), variables=["t2m"])["t2m"]
    base = read_chunk(obstore_store(url), "t2m", 0, geom)
    fn = load_transform(f"{transforms_file}:vectorized_scale")
    out = gil_probe(fn, base, threads=1, min_seconds=0.02, repeats=1)
    assert out["speedup"] == pytest.approx(1.0, abs=0.6)
