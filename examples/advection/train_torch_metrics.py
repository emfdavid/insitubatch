"""Loader stall / in-memory-ceiling benchmark for the advection torch loop (backlog #9).

Wraps the *same* model and step as ``train_torch.py`` (imported, not duplicated) in the
:class:`~examples._forecast_metrics.StallTimer` so each epoch records the **data-stall
fraction** -- the share of wall-clock the GPU idles waiting on the loader. ``--ceiling``
trains a second, identical model on a RAM-preloaded epoch (no fetch/decode) to get the
compute-only in-memory ceiling the loader run is scored against; ``--metrics-out`` appends
per-(run, epoch) JSONL.

Kept out of ``train_torch.py`` so that file stays a clean usage example. Run with::

    python -m examples.advection.train_torch_metrics --source wb2 --device cuda \\
        --epochs 5 --ceiling --metrics-out bench/results/advection_torch_wb2_gpu.jsonl
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable

import torch
from torch import nn

from insitubatch import Batch, InSituDataset, to_torch

from .._forecast_metrics import MetricsLog, StallTimer, preload_epoch
from .data import build_datasets, build_parser, evaluate
from .train_torch import AdvectionCNN, _forecast


def _fit(
    epochs_source: Callable[[int], Iterable[Batch]],
    *,
    epochs: int,
    dev: torch.device,
    log: MetricsLog,
    run: str,
    source: str,
    batch_size: int,
) -> AdvectionCNN:
    """Train a fresh CNN over ``epochs``, recording one metrics row per epoch.

    ``epochs_source(epoch)`` yields that epoch's batches -- the loader (``insitu``) or the
    RAM-preloaded list (``ceiling``). The step syncs via ``loss.item()`` so the
    :class:`StallTimer`'s data-wait / compute split is real wall-clock, not queued async."""
    cuda = dev.type == "cuda"
    model = AdvectionCNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        if cuda:
            torch.cuda.reset_peak_memory_stats(dev)
        timer = StallTimer()
        for batch in timer.wrap(epochs_source(epoch)):
            target = to_torch(batch)["target"][:, None].to(dev)
            loss = nn.functional.mse_loss(_forecast(model, batch, dev), target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss.item()  # sync: makes the timer's compute window real GPU wall-clock
        gpu_mb = torch.cuda.max_memory_allocated(dev) / 1e6 if cuda else 0.0
        log.add(
            timer.metrics(
                run=run,
                framework="torch",
                source=source,
                device=dev.type,
                epoch=epoch,
                batch_size=batch_size,
                peak_gpu_mem_mb=gpu_mb,
            )
        )
    return model


def _val_rmse(model: AdvectionCNN, ds: InSituDataset, dev: torch.device) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        return evaluate(ds.val, lambda b: _forecast(model, b, dev).detach().cpu().numpy())


def train(
    ds: InSituDataset,
    *,
    epochs: int,
    device: str = "cpu",
    source: str = "synthetic",
    batch_size: int = 32,
    ceiling: bool = False,
    log: MetricsLog | None = None,
) -> tuple[float, float]:
    """Train the CNN and record stall/throughput metrics to ``log``; return the val skill.

    With ``ceiling=True`` it then trains a second, identical model on a RAM-preloaded epoch
    (no IO) -- the compute-only ceiling the ``insitu`` run is scored against."""
    dev = torch.device(device)
    log = log or MetricsLog(None)

    def loader_epoch(epoch: int) -> Iterable[Batch]:
        ds.set_epoch(epoch)
        return ds.train

    model = _fit(
        loader_epoch,
        epochs=epochs,
        dev=dev,
        log=log,
        run="insitu",
        source=source,
        batch_size=batch_size,
    )
    model_rmse, persistence_rmse = _val_rmse(model, ds, dev)
    log.set_val("insitu", model_rmse, persistence_rmse)

    if ceiling:
        preloaded = preload_epoch(ds)  # one shuffled epoch in RAM; replayed each epoch
        _fit(
            lambda epoch: preloaded,
            epochs=epochs,
            dev=dev,
            log=log,
            run="ceiling",
            source=source,
            batch_size=batch_size,
        )
    return model_rmse, persistence_rmse


def cli() -> argparse.Namespace:
    """The shared example CLI plus the two benchmark-only flags."""
    p = build_parser()
    p.add_argument(
        "--ceiling",
        action="store_true",
        help="also run the compute-only in-memory ceiling (RAM-preloaded batches, no IO)",
    )
    p.add_argument(
        "--metrics-out",
        default=None,
        metavar="PATH",
        help="append per-(run,epoch) JSONL metrics here (default: print only, no file)",
    )
    return p.parse_args()


def main() -> None:
    args = cli()
    ds = build_datasets(args)
    log = MetricsLog(args.metrics_out)
    model_rmse, persistence_rmse = train(
        ds,
        epochs=args.epochs,
        device=args.device,
        source=args.source,
        batch_size=args.batch_size,
        ceiling=args.ceiling,
        log=log,
    )
    log.summary()
    log.flush()
    print(
        f"\n24 h forecast RMSE on held-out data: model {model_rmse:.3f}  vs  "
        f"persistence {persistence_rmse:.3f}  ({persistence_rmse / model_rmse:.1f}x better)"
    )


if __name__ == "__main__":
    main()
