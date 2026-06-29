# GCP ops — GCE + GCS for the cloud benchmark

The GCP counterpart to [`ops_aws.md`](ops_aws.md): stand up a GCE box that reads zarr from a
co-located **GCS** bucket, so the benchmark runs against Google's object store the same way
it runs against S3. The dataset matrix and per-story run commands live in
[`benchmark_plan.md`](benchmark_plan.md).

> **Status: best-effort, not yet run end-to-end.** These commands are a starting point
> derived from the AWS runbook and the GCP docs, not a validated transcript. Verify each
> step on first run — especially [§7 Rapid Storage](#7-rapid-storage-grpc--the-obstore-high-performance-experiment)
> and [§8 GPU / G2](#8-gpu-box--g2--nvidia-l4-pending-quota), which are blocked on a pending
> quota request. Lines that need checking are marked **VERIFY**.

## Contents

- [Access model](#read-first--the-access-model)
- [Variables](#variables)
- [1. Project + APIs](#1-project--apis)
- [2. GCS bucket](#2-gcs-bucket--regional-public-read)
- [3. Service account + access](#3-service-account--bucket-access)
- [4. Firewall](#4-firewall-ssh-from-your-ip-only)
- [5. Launch the CPU box](#5-launch-the-cpu-box-local-ssd-nvme)
- [6. On the box: install + bench](#6-on-the-box--mount-nvme-install-generate-bench)
- [7. Rapid Storage (gRPC)](#7-rapid-storage-grpc--the-obstore-high-performance-experiment)
- [8. GPU box (G2 / L4)](#8-gpu-box--g2--nvidia-l4-pending-quota)
- [Teardown](#teardown) · [Cost](#cost) · [Notes](#notes)

## Read first — the access model

- **Public read = anonymous.** Unlike S3 Requester-Pays, the simplest public GCS pattern is
  an anonymous-readable bucket (`allUsers` → `roles/storage.objectViewer`); obstore reads it
  with `skip_signature=True` (the same flag the WeatherBench2 example uses for `gs://`). The
  owner pays egress, so keep readers in-region.
- **Requester Pays exists** on GCS too (`--requester-pays`), but obstore's billing-project
  pass-through for GCS is **VERIFY** — confirm before relying on it. For the public benchmark,
  anonymous read is the tested path.
- **Co-locate** bucket and instance in one region (and one **zone** for local SSD and Rapid
  Storage). Cross-region egress is slow and billed.

## Variables

```bash
export PROJECT=$(gcloud config get-value project)
export REGION=us-central1
export ZONE=us-central1-a
export BUCKET="insitubatch-bench-${PROJECT}"
export INSTANCE=insitubatch-bench
```

## 1. Project + APIs

```bash
gcloud services enable compute.googleapis.com storage.googleapis.com
```

## 2. GCS bucket — regional, public-read

```bash
gcloud storage buckets create "gs://$BUCKET" \
  --location="$REGION" \
  --uniform-bucket-level-access

# anonymous public read (so external reproducers need no credentials)
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member=allUsers --role=roles/storage.objectViewer
```

## 3. Service account + bucket access

The GCE instance reads/writes the bucket via its attached service account — no keys on disk,
the GCP analogue of an IAM instance profile. The default compute service account with the
`cloud-platform` scope is enough for the owner's own bucket; scope it down for shared infra.

```bash
export SA=$(gcloud iam service-accounts list \
  --filter="displayName:Compute Engine default" --format="value(email)")
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member="serviceAccount:$SA" --role=roles/storage.objectAdmin
```

## 4. Firewall (SSH from your IP only)

```bash
export MYIP=$(curl -s https://checkip.amazonaws.com)
gcloud compute firewall-rules create insitubatch-ssh \
  --network=default --allow=tcp:22 --source-ranges="${MYIP}/32"
```

## 5. Launch the CPU box (local SSD NVMe)

`n2-standard-32` ≈ the `c6id.8xlarge` (32 vCPU); one local SSD gives ephemeral NVMe for the
mmap cache spill. **VERIFY** the machine family supports the local-SSD count you ask for.

```bash
gcloud compute instances create "$INSTANCE" \
  --zone="$ZONE" \
  --machine-type=n2-standard-32 \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --local-ssd=interface=NVME \
  --scopes=cloud-platform \
  --provisioning-model=SPOT --instance-termination-action=STOP

gcloud compute ssh "$INSTANCE" --zone="$ZONE"
```

## 6. On the box — mount NVMe, install, generate, bench

```bash
export REGION=us-central1
export BUCKET=insitubatch-bench-<PROJECT>

# mount the local SSD (ephemeral scratch for the mmap cache)
# find the device (local SSD shows as /dev/nvme0n... via google-local-nvme-ssd-0)
lsblk
sudo mkfs.ext4 -F /dev/nvme0n1
sudo mkdir -p /mnt/nvme && sudo mount /dev/nvme0n1 /mnt/nvme && sudo chown "$USER" /mnt/nvme

# install
curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.bashrc"
sudo apt-get update && sudo apt-get install -y git
git clone https://github.com/emfdavid/insitubatch.git && cd insitubatch
uv sync --extra torch --extra bench

# sanity-check the install before the long run
uv run pytest -q

# generate the chunk-size family (instance SA has objectAdmin -> no extra auth)
for spc in 1 2 4 8 16 32; do
  uv run python bench/make_dataset.py --url "gs://$BUCKET/era5_c${spc}.zarr" \
    --sample-chunk "$spc" --n-samples 6000 --inner 721,1440
done

# run the suite + render Plotly graphs (mmap cache on the local SSD)
uv run python -m bench --full --url-prefix "gs://$BUCKET/era5" \
  --cache-dir /mnt/nvme/cache
```

obstore reads `gs://` through the same `store_from_url` shim; credentials come from the
instance service account (Application Default Credentials) automatically. A public reader on
another account passes `skip_signature=True` instead — **VERIFY** the kwarg name flows
through `make_dataset`/`open_geometries` for `gs://` as it does for the WB2 example.

## 7. Rapid Storage (gRPC) — the obstore high-performance experiment

**Goal:** test obstore against **Google Cloud Rapid Storage** — a *zonal* GCS bucket served
over the high-throughput **gRPC** Cloud Storage API (the GCS analogue of S3 Express One Zone).
The open question is whether obstore (via Rust `object_store`) can drive the gRPC endpoint at
all, and if so how close it gets to the raw-GET ceiling — exactly the story-4 efficiency
measurement, on a faster floor.

> **All of this section is UNVERIFIED.** Rapid Storage is new and the exact `gcloud` flags,
> the gRPC endpoint wiring, and obstore's gRPC support all need checking before the numbers
> mean anything. Treat it as an experiment plan, not a runbook.

Open questions to resolve first:

- **Does obstore/`object_store` speak the GCS gRPC API?** The S3-Express win came for free
  because it is still the S3 HTTP API; GCS gRPC is a *different protocol*. If `object_store`
  is HTTPS/JSON-only for GCS, this needs an obstore feature or a different client — confirm
  before provisioning.
- **Endpoint selection.** How obstore is told to use the gRPC endpoint (a URL scheme, a store
  kwarg, or an env var) is **VERIFY** — there may be no hook yet.

Provisioning sketch (**VERIFY** every flag against current docs — Rapid Storage requires a
zonal location and hierarchical namespace):

```bash
export RAPID_BUCKET="insitubatch-rapid-${PROJECT}"
gcloud storage buckets create "gs://$RAPID_BUCKET" \
  --location="$ZONE" \
  --enable-hierarchical-namespace \
  --storage-class=RAPID
```

Then mirror the story-4 datasets (the GRIB end + a fat-spatial grid) onto the Rapid bucket
and run the probe's raw-GET-vs-decoded comparison, as in `benchmark_plan.md` story 4:

```bash
uv run python bench/make_dataset.py --url "gs://$RAPID_BUCKET/era5_fat_g16.zarr" \
  --n-samples 3200 --inner 361,720 --sample-chunk 200 --inner-chunks 91,180 --variables t2m

uv run python -m bench.probe_decode --url "gs://$RAPID_BUCKET/era5_fat_g16.zarr" \
  --max-inflight 64 --concurrency 8,16,32,64 --max-chunks 256 --repeats 5
```

If obstore cannot reach the gRPC endpoint, record that as the finding (a gap to file upstream)
rather than silently falling back to HTTPS, which would measure the wrong thing.

## 8. GPU box — G2 / NVIDIA L4 (pending quota)

The advection examples (`examples/advection/train_{torch,jax,tf}.py`) train the forecast CNN
on real ERA5 read in place; a single **L4** is plenty, the win is fast local SSD for the chunk
cache. This is the GCP twin of [`ops_aws.md` §10](ops_aws.md) (AWS G6 / L4).

> **Pending a GPU quota request** (`NVIDIA L4 GPUs` in the target region). Commands below are
> **UNVERIFIED** until that lands.

```bash
export ZONE=us-central1-a
export GPU_INSTANCE=insitubatch-gpu

# g2-standard-8 = 1x L4, 8 vCPU. The Deep Learning VM image ships the driver + CUDA;
# we still bring torch/jax/tf via uv (as on AWS). VERIFY the current image family.
gcloud compute instances create "$GPU_INSTANCE" \
  --zone="$ZONE" \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=common-cu123-debian-11 --image-project=deeplearning-platform-release \
  --local-ssd=interface=NVME \
  --scopes=cloud-platform \
  --boot-disk-size=100GB

gcloud compute ssh "$GPU_INSTANCE" --zone="$ZONE"
```

On the box, mount the local SSD (§6), install with the GPU extra, and run the same advection
training as the AWS GPU section — `--device cuda`, `--cache-dir /mnt/nvme/cache`, a finite
`--sample-range`:

```bash
# confirm the driver sees the L4 first
nvidia-smi

uv sync --extra torch --extra arraylake
# CUDA torch (the Deep Learning image already has CUDA; swap the CPU wheel)
uv pip install torch --torch-backend=auto
uv run python -c "import torch; print(torch.cuda.is_available())"  # expect True

uv run python -m examples.advection.train_torch \
  --source arraylake --sample-range 0,2920 \
  --device cuda --epochs 8 --batch-size 32 \
  --cache-dir /mnt/nvme/cache
```

Same sizing caveats as the AWS GPU run apply (see [`ops_aws.md` §10](ops_aws.md)): a finite
`--sample-range` for the real store, full epochs to warm the cross-epoch cache, and high cold
TTFB over a high-latency network is expected.

## Teardown

```bash
gcloud compute instances delete "$INSTANCE" --zone="$ZONE"
# keep the bucket for reproducers; to remove later:
#   gcloud storage rm --recursive "gs://$BUCKET"
#   gcloud compute firewall-rules delete insitubatch-ssh
```

## Cost

- `n2-standard-32`: Spot ~$0.3–0.4/hr, on-demand ~$1.5/hr. **Stop or delete when done.**
- `g2-standard-8` (1x L4): Spot ~$0.2–0.3/hr, on-demand ~$0.7–0.9/hr (**VERIFY** current
  pricing). Local SSD is billed while the instance exists and wiped on stop.
- GCS storage: ~$0.02/GB-mo (standard, regional). Rapid Storage is priced higher — **VERIFY**.

## Notes

- Local SSD is ephemeral (wiped on stop/delete); the dataset stays in GCS, the mmap cache
  spills to `/mnt/nvme`.
- The multi-engine `python -m bench` suite needs torch (`workers`/`xbatcher` engines); the
  core probe runs without it.
- This file is a draft. As steps are validated on a real run, drop the **VERIFY** /
  **UNVERIFIED** markers and record the actual numbers in `benchmark_plan.md`.
