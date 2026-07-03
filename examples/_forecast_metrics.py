"""Framework-neutral training metrics for the forecast examples.

The headline is the **data-stall fraction**: the share of wall-clock the compute
device idles waiting on the loader. Prefetch hiding IO behind compute drives it
toward zero -- the run is compute-bound and the loader is effectively free. It is
self-referential (no competitor baseline), so it replaces an xbatcher A/B.

The paired ``--ceiling`` run feeds the *same* model + loop from a RAM-preloaded
list of batches (no fetch, no decode); its throughput is the compute-only ceiling
and ``insitu_samples_per_s / ceiling_samples_per_s`` is the loader overhead.

This module owns everything framework-independent: the per-(run, epoch) JSONL
schema (:class:`ForecastMetrics`), the stall/throughput timer, and the in-memory
log that patches the final validation RMSE onto the last row before flushing. The
torch loop supplies only the two framework-specific numbers (a synced step and the
GPU peak); a later JAX/TF port reuses the same collector with their own sync.
"""

from __future__ import annotations

import json
import math
import platform
import socket
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Reuse the bench suite's RssAnon reader so host-memory is measured the same way the
# storage benchmarks report it (heap/stack only -- the true bound; mmap cache pages
# are reclaimable and excluded). bench is a repo-root dev package; the examples run
# from the repo root (``python -m examples...``), so this import resolves there.
from bench.result import rss_breakdown_mb
from insitubatch.types import Batch


@dataclass
class ForecastMetrics:
    """One self-describing row per (run, epoch) for the forecast training benchmark.

    ``run`` is ``insitu`` (fed by the loader) or ``ceiling`` (fed from RAM). The
    headline is ``data_stall_fraction``; ``val_*`` are filled only on the final row
    of a run (one training run's forecast skill, a sanity check that it really trained).
    """

    run: str  # insitu | ceiling
    framework: str  # torch | jax | tf
    source: str  # synthetic | wb2 | arraylake
    device: str  # cpu | cuda
    epoch: int
    n_samples: int
    n_batches: int
    batch_size: int
    wall_s: float
    data_wait_s: float
    compute_s: float
    data_stall_fraction: float  # data_wait_s / wall_s -- the HEADLINE
    samples_per_s: float
    mb_per_s: float  # decoded volume delivered to the model
    mean_data_wait_ms: float  # per batch
    mean_compute_ms: float  # per batch
    ttfb_ms: float  # wait for the first batch of the epoch (cold start on epoch 0)
    peak_rss_anon_mb: float  # heap/stack high-water at epoch end (the real host bound)
    peak_gpu_mem_mb: float  # torch.cuda peak this epoch (0 on cpu)
    val_model_rmse: float = math.nan  # set on the last row of a run only
    val_persistence_rmse: float = math.nan
    host: str = field(default_factory=socket.gethostname)
    platform: str = field(default_factory=platform.platform)
    ts: float = field(default_factory=time.time)


class StallTimer:
    """Accumulate data-stall vs compute time over one epoch by wrapping its iterator.

    ``wrap`` yields each batch, timing the gap *before* it arrives as ``data_wait``
    (the loader stall) and the caller's loop body after it as ``compute``. This is
    only valid when the compute body synchronizes every step -- the torch loop's
    ``loss.item()`` does; a JAX/TF port must block equivalently.
    """

    def __init__(self) -> None:
        self.data_wait = 0.0
        self.compute = 0.0
        self.ttfb = 0.0
        self.n_batches = 0
        self.n_samples = 0
        self.total_bytes = 0

    def wrap(self, source: Iterable[Batch]) -> Iterator[Batch]:
        t = time.perf_counter()
        for i, batch in enumerate(source):
            now = time.perf_counter()
            wait = now - t
            self.data_wait += wait
            if i == 0:
                self.ttfb = wait
            self.n_batches += 1
            self.n_samples += len(batch.sample_indices)
            self.total_bytes += sum(a.nbytes for a in batch.arrays.values())
            yield batch  # caller runs (and synchronizes) the compute step here
            after = time.perf_counter()
            self.compute += after - now
            t = after

    def metrics(
        self,
        *,
        run: str,
        framework: str,
        source: str,
        device: str,
        epoch: int,
        batch_size: int,
        peak_gpu_mem_mb: float,
    ) -> ForecastMetrics:
        """Finalize the epoch into a row. Sample host RSS now (epoch-end ~= steady-state
        peak for a bounded prefetch buffer -- kept out of the timed loop so the /proc read
        never counts as compute)."""
        wall = self.data_wait + self.compute
        nb = max(self.n_batches, 1)
        return ForecastMetrics(
            run=run,
            framework=framework,
            source=source,
            device=device,
            epoch=epoch,
            n_samples=self.n_samples,
            n_batches=self.n_batches,
            batch_size=batch_size,
            wall_s=wall,
            data_wait_s=self.data_wait,
            compute_s=self.compute,
            data_stall_fraction=self.data_wait / wall if wall else 0.0,
            samples_per_s=self.n_samples / wall if wall else 0.0,
            mb_per_s=(self.total_bytes / 1e6) / wall if wall else 0.0,
            mean_data_wait_ms=self.data_wait / nb * 1e3,
            mean_compute_ms=self.compute / nb * 1e3,
            ttfb_ms=self.ttfb * 1e3,
            peak_rss_anon_mb=rss_breakdown_mb()[0],
            peak_gpu_mem_mb=peak_gpu_mem_mb,
        )


class MetricsLog:
    """Collect rows in memory, print a one-line summary each, flush to JSONL at the end.

    Rows are held (not streamed) so the final validation RMSE -- known only after all
    epochs and the eval pass -- can be patched onto the last row of the ``insitu`` run
    before writing. ``path=None`` prints only (the CPU-validation path leaves no file).
    """

    def __init__(self, path: str | None) -> None:
        self.path = path
        self.rows: list[ForecastMetrics] = []

    def add(self, m: ForecastMetrics) -> None:
        self.rows.append(m)
        print(
            f"  [{m.run:>7}] epoch {m.epoch}  stall {m.data_stall_fraction:5.1%}  "
            f"{m.samples_per_s:7.1f} samp/s  {m.mb_per_s:6.1f} MB/s  "
            f"wait {m.mean_data_wait_ms:5.1f}ms  compute {m.mean_compute_ms:5.1f}ms  "
            f"ttfb {m.ttfb_ms:6.1f}ms  rss {m.peak_rss_anon_mb:6.0f}MB  "
            f"gpu {m.peak_gpu_mem_mb:6.0f}MB"
        )

    def set_val(self, run: str, model_rmse: float, persistence_rmse: float) -> None:
        """Attach the run's forecast skill to its final row (last epoch of that run)."""
        for m in reversed(self.rows):
            if m.run == run:
                m.val_model_rmse = model_rmse
                m.val_persistence_rmse = persistence_rmse
                return

    def steady_samples_per_s(self, run: str) -> float:
        """Mean throughput over the run's rows, excluding epoch 0 (cold TTFB) when possible."""
        rows = [m for m in self.rows if m.run == run]
        warm = [m for m in rows if m.epoch > 0] or rows
        return sum(m.samples_per_s for m in warm) / len(warm) if warm else 0.0

    def summary(self) -> None:
        """Print the headline: insitu stall and its % of the in-memory ceiling (if run)."""
        insitu = self.steady_samples_per_s("insitu")
        ceiling = self.steady_samples_per_s("ceiling")
        stall = sum(m.data_stall_fraction for m in self.rows if m.run == "insitu" and m.epoch > 0)
        n = max(len([m for m in self.rows if m.run == "insitu" and m.epoch > 0]), 1)
        line = f"\ninsitu: {insitu:.1f} samp/s (steady), stall {stall / n:.1%}"
        if ceiling:
            line += f"  ->  {insitu / ceiling:.1%} of the {ceiling:.1f} samp/s in-memory ceiling"
        print(line)

    def flush(self) -> None:
        if not self.path:
            return
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            for m in self.rows:
                f.write(json.dumps(asdict(m)) + "\n")
        print(f"wrote {len(self.rows)} rows -> {self.path}")
