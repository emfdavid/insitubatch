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

Two mutually exclusive read models; **anonymous-public is the only one that works
end-to-end today** — see the Requester Pays blocker below.

- **Anonymous public read (tested).** The simplest public GCS pattern is an
  anonymous-readable bucket (`allUsers` → `roles/storage.objectViewer`); obstore reads it
  with `skip_signature=True` (the same flag the WeatherBench2 example uses for `gs://`). The
  owner pays egress, so keep readers in-region.
- **Requester Pays (bucket side works, client side BLOCKED).** GCS supports
  `--requester-pays`, but unlike S3 there is **no owner exemption**: *every* non-exempt
  request — yours from the GCE box included — must carry a billing project
  (`x-goog-user-project`). obstore 0.10.1's `GCSConfig` exposes **no** user-project / billing
  knob (just auth options like `skip_signature`, `service_account`, `token`), so obstore cannot send it and
  both owner and reproducer reads 400. This is why S3's `request_payer=True` has no GCS
  equivalent here. Do not enable Requester Pays on the bench bucket until that upstream knob
  lands — see [§2](#2-gcs-bucket--regional-public-read).
- **Co-locate** bucket and instance in one region (and one **zone** for local SSD and Rapid
  Storage). Cross-region egress is slow and billed.

## Variables

> **zsh (macOS default):** run `setopt interactive_comments` once before pasting — the
> blocks below contain `#` comments, and interactive zsh otherwise runs `#` as a command
> (`zsh: command not found: #`). bash honors in-block comments by default.

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

### Requester Pays — the AWS analogue (currently BLOCKED on obstore)

The GCS counterpart to [`ops_aws.md` §2](ops_aws.md#2-s3-bucket--regional-public-read-requester-pays).
The bucket-side setup is a two-liner — any authenticated Google account may read, and the
reader's own project is billed for egress:

```bash
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member=allAuthenticatedUsers --role=roles/storage.objectViewer
gcloud storage buckets update "gs://$BUCKET" --requester-pays
```

> **obstore cannot read Requester-Pays GCS; use `fsspec_store` instead.** obstore 0.10.1 has
> no GCS billing-project config (verified: `GCSConfig` carries auth options like
> `skip_signature` / `service_account` but no `user_project`), and GCS — unlike S3 — grants
> the owner **no** Requester-Pays exemption, so enabling it breaks *your own* GCE bench reads
> too (`... is a requester pays bucket but no user project provided`). The path is
> `insitubatch.fsspec_store` (gcsfs), which forwards a billing project via `storage_options`:
> `fsspec_store("gs://bucket/ds.zarr", project="my-billing-project", requester_pays=True)`.
> obstore stays the default for non-Requester-Pays reads; anonymous-public (above) needs no
> billing project on either backend.

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

### Reuse an existing SSH key (skip gcloud's auto-keygen)

GCE has no importable key-pair object like EC2's `import-key-pair`; left to itself
`gcloud compute ssh` generates and manages a dedicated `~/.ssh/google_compute_engine` pair.
To connect with an existing key instead — the analogue of
[`ops_aws.md` §1](ops_aws.md#1-import-your-ssh-key-from-ssh-agent) — inject its **public**
half as instance metadata at launch. The metadata format is `USERNAME:KEY`, and `USERNAME`
becomes your login user on the box:

```bash
export SSH_USER="$USER"
printf '%s:%s\n' "$SSH_USER" "$(ssh-add -L | grep emfdavid_ed25519)" \
  > /tmp/gce-ssh-keys.txt
```

Then launch with the key wired in via `--metadata-from-file`, and connect with plain
`ssh -A` (the AWS-style flow), not `gcloud compute ssh`:

```bash
gcloud compute instances create "$INSTANCE" \
  --zone="$ZONE" \
  --machine-type=n2-standard-32 \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --local-ssd=interface=NVME \
  --scopes=cloud-platform \
  --provisioning-model=SPOT --instance-termination-action=STOP \
  --metadata-from-file=ssh-keys=/tmp/gce-ssh-keys.txt

IP=$(gcloud compute instances describe "$INSTANCE" --zone="$ZONE" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
ssh -A "$SSH_USER@$IP"
```

The key comes from your **ssh-agent** (as in the AWS flow) — no `-i` and no on-disk key
file needed. First connect to a fresh box prompts to accept the host key; if instead you get
`Host key verification failed`, GCP has likely recycled the external IP and a stale
`known_hosts` entry is colliding — clear it and retry:

```bash
ssh-keygen -R "$IP"
ssh -A "$SSH_USER@$IP" -o StrictHostKeyChecking=accept-new
```

> **OS Login caveat.** If OS Login is enabled (project/instance metadata
> `enable-oslogin=TRUE`), metadata `ssh-keys` are *ignored* — register the key with
> `gcloud compute os-login ssh-keys add --key-file="$HOME/.ssh/emfdavid_ed25519.pub"`
> instead. It is off by default; check with
> `gcloud compute project-info describe --format='value(commonInstanceMetadata.items)' | grep -i oslogin`.

### (Optional) Static external IP — stable address across stop/start

The GCE counterpart to the AWS Elastic IP
([`ops_aws.md` §6](ops_aws.md#optional-elastic-ip--stable-address-across-stopstart)): an
ephemeral external IP is released on stop and a new one assigned on start, so reserve a
**static external IP** to keep the box's address fixed. Reserve a regional address and swap
the instance's external interface onto it:

```bash
# reserve a regional static IP
gcloud compute addresses create "${INSTANCE}-ip" --region="$REGION"
STATIC_IP=$(gcloud compute addresses describe "${INSTANCE}-ip" \
  --region="$REGION" --format='get(address)')

# swap the instance's ephemeral access-config for the reserved IP (look up the existing
# access-config name first; the default created at launch is "External NAT")
CONFIG=$(gcloud compute instances describe "$INSTANCE" --zone="$ZONE" \
  --format='get(networkInterfaces[0].accessConfigs[0].name)')
gcloud compute instances delete-access-config "$INSTANCE" --zone="$ZONE" \
  --access-config-name="$CONFIG"
gcloud compute instances add-access-config "$INSTANCE" --zone="$ZONE" \
  --access-config-name="external-nat" --address="$STATIC_IP"
echo "ssh -A $SSH_USER@$STATIC_IP"
```

To assign it at launch instead, reserve the address first and add `--address="$STATIC_IP"`
to the §5 `instances create`. A static IP on a running VM is a few cents a day; a
**reserved-but-unused** one is billed at a higher rate — delete it at
[teardown](#teardown) rather than leaving it reserved.

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

obstore reads `gs://` through `obstore_store`; credentials come from the instance service
account (Application Default Credentials) automatically. A public reader on another account
passes `skip_signature=True` instead — **VERIFY** the kwarg name flows through
`make_dataset`/`open_geometries` for `gs://` as it does for the WB2 example. For Rapid/zonal or
Requester-Pays buckets, build the store with `fsspec_store` (gcsfs) instead — see §7 and §2.

## 7. Rapid Storage (gRPC) — the fsspec/gcsfs high-performance experiment

**Goal:** test **Google Cloud Rapid Storage** — a *zonal* GCS bucket served over the
high-throughput **gRPC** Cloud Storage API (the GCS analogue of S3 Express One Zone) — as an
insitubatch read path. The measurement is the story-4 raw-GET-vs-decoded efficiency, on a
faster floor.

> **Backend: gcsfs, not obstore.** Rust `object_store` (obstore) speaks GCS over HTTPS/JSON
> only — it has no gRPC path — so it cannot drive the Rapid endpoint. gcsfs *does* (it has
> dedicated Rapid/zonal support: True Appends over gRPC BidiWriteObject, streaming I/O). The
> read path is therefore `insitubatch.fsspec_store("gs://…", …)` (zarr `FsspecStore` over
> gcsfs), which is exactly why fsspec_store exists. The headline experiment is **fsspec-over-
> gcsfs on Rapid vs obstore-over-HTTP on standard GCS** — does zonal gRPC beat the raw-GET
> ceiling enough to justify fsspec as a co-equal fast path (see DESIGN.md M-GCS)?

> **Still UNVERIFIED end-to-end.** Rapid Storage is new; the exact `gcloud` flags, the gcsfs
> Rapid `storage_options`, and the measured numbers all need checking on the live box before
> they mean anything. Treat provisioning below as an experiment plan, not a settled runbook.

### 7a — Standard-GCS A/B (obstore vs fsspec): run this first

Before Rapid, establish that fsspec is **not a regression on standard GCS** — the
prerequisite for co-equal status, and the same harness you re-point at the Rapid bucket
below. Runnable on any box (recipe validated on an `n2-standard-8` with a 375 GB local SSD
at `/mnt/nvme`). The one box limit that matters here is **decode saturation** (few cores →
the full pipeline goes decode-bound, which is backend-invariant and *hides* the fetch
difference), so the sharpest backend signal is the **decode-free raw GET** (`probe_decode`
section 2). The **ChunkPool cache is post-decode and backend-invariant** — obstore and
fsspec share the same decoded bytes — so it does *not* discriminate the backends; the NVMe
is used to (a) run the end-to-end suite as designed (mmap tier, not a RAM-only degraded
mode) and (b) measure the decode-once ceiling (cold vs warm, `probe_decode` §1c), which
tells you whether you are fetch-bound enough for the backend to matter at all. **Gotcha:**
a cache is keyed by chunk, not by backend, so never point both backends at the *same*
`--cache-dir` — the second run would hit the first's decoded chunks and measure the cache,
not its own reads. The discriminating A/B (Step 1) is therefore cache-free; cached runs use
a fresh per-backend dir.

`probe_decode` prints human-readable tables → capture as `.log`; the `bench` suite writes
JSONL natively → `--out …jsonl`. All logs land in `bench/results/`.

```bash
export PREFIX="gs://$BUCKET/era5"          # the _c{spc}.zarr family from §6
export NVME=/mnt/nvme                      # the local SSD (fixed at 1 drive for this class)
mkdir -p bench/results "$NVME/insitu-cache"
{ hostname; nproc; date -u; } | tee bench/results/BOX.txt   # stamp the hardware

# Step 0 — characterize THIS box (don't trust the spec sheet). One obstore run gives:
#   section 1  = where decode saturates on your cores;
#   section 1c = the decode-once ceiling (cold vs warm epoch off NVMe) -> if warm >> cold
#                you're fetch/decode-bound (backend can matter); if warm ~= cold, delivery-bound;
#   section 2  = the NIC ceiling (where raw-GET MB/s stops rising).
uv run python -m bench.probe_decode --url "${PREFIX}_c8.zarr" --backend obstore \
  --max-chunks 64 --repeats 3 --decode-threads 1,2,4,8,0 \
  --concurrency 4,8,16,32,64 --max-inflight 32 --cache-dir "$NVME/insitu-cache/calib" \
  2>&1 | tee bench/results/probe_calib_obstore_c8.log

# Step 1 — the A/B: raw transfer floor (section 2) + bridge scaling (section 1b), both
#   backends, grib->mid->fat. The grib end (c1) is the discriminating case (per-request Python cost).
for BK in obstore fsspec; do
  for SPC in 1 4 16; do
    uv run python -m bench.probe_decode --url "${PREFIX}_c${SPC}.zarr" --backend "$BK" \
      --max-chunks 128 --repeats 3 --no-decode-sweep \
      --max-inflight 8,16,32 --concurrency 4,8,16,32 \
      2>&1 | tee "bench/results/probe_${BK}_c${SPC}.log"
  done
done

# Step 1b (optional sharpener) — pure request-rate: tiny objects, all per-call overhead,
#   no bandwidth. Where obstore's Rust path would win most, if anywhere.
uv run python bench/make_dataset.py --url "gs://$BUCKET/tiny_c1.zarr" \
  --n-samples 4000 --inner 64,64 --sample-chunk 1 --variables t2m
for BK in obstore fsspec; do
  uv run python -m bench.probe_decode --url "gs://$BUCKET/tiny_c1.zarr" --backend "$BK" \
    --max-chunks 256 --repeats 3 --no-decode-sweep --max-inflight 16,32 --concurrency 8,16,32 \
    2>&1 | tee "bench/results/probe_${BK}_tiny.log"
done

# Step 2 — end-to-end confirmation, mmap cache tier on NVMe. Fresh per-backend cache dir
#   (never shared — see the gotcha above) so each backend's reads are its own.
for BK in obstore fsspec; do
  uv run python -m bench --url-prefix "$PREFIX" --backend "$BK" \
    --chunk-sizes 1,2,4,8,16,32 --engines insitu,naive --repeats 3 --max-batches 100 \
    --cache-dir "$NVME/insitu-cache/e2e_${BK}" \
    --out "bench/results/gcs_e2e_${BK}.jsonl"
done
```

**Read it:** fsspec earns co-equal *on standard GCS* if raw-GET (Step 1 section 2, read
*below* your Step-0 NIC ceiling — especially c1/tiny) is within ~10–15 % of obstore, the
section-1b curves don't flatten fsspec early (the cross-loop bridge isn't throttling), and
end-to-end insitu tracks obstore. A large c1/tiny raw-GET gap is the one red flag (the
per-request Python tax). This proves *not a regression* — **not** the Rapid win, which
needs the gRPC bucket below (DESIGN.md M-GCS).

> **Sustained-rate caveat — big-chunk raw-GET reads low, for *both* backends.** `--max-chunks`
> fixes the *object count*, so a larger `sample-chunk` moves proportionally more bytes per
> point (single-inner-chunk: c16 ≈ 16× the bytes of c1). Those points therefore measure
> **sustained** throughput after GCP/GCS burst credits deplete, not the **burst** rate the
> small-chunk points catch — so a dip in MB/s at c16/c32 is this byte-volume artifact, *not*
> a backend effect (it hits obstore and fsspec equally). The verdict is the **gap between
> backends at a fixed chunk size**, which is invariant to it. To make per-chunk-size points
> directly comparable, hold bytes ~constant by scaling `--max-chunks` *down* as chunk size
> rises (e.g. 128 at c1 → 8 at c16).

Open questions to resolve first:

- **gcsfs Rapid `storage_options`.** How gcsfs is told to use the zonal gRPC endpoint (an
  endpoint override, a bucket-type flag, or automatic on a Rapid bucket) is **VERIFY** —
  confirm against the gcsfs Rapid docs, then thread it through `fsspec_store(...)`.
- **Write path.** `bench/make_dataset.py` writes via `obstore_store` (HTTP); confirm whether it
  must switch to gcsfs to *populate* a Rapid bucket, or whether HTTP writes + gRPC reads work.

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

# g2-standard-16 = 1x L4, 16 vCPU. The Deep Learning VM image ships the driver + CUDA;
# we still bring torch/jax/tf via uv (as on AWS). VERIFY the current image family.
# vCPU count is load-bearing here: the async decode path scales with cores, so the
# advection sweep numbers were taken on 16 vCPU -- an 8-vCPU box will read lower.
gcloud compute instances create "$GPU_INSTANCE" \
  --zone="$ZONE" \
  --machine-type=g2-standard-16 \
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
# if you reserved a static IP, delete it (a reserved-but-unused address keeps billing):
#   gcloud compute addresses delete "${INSTANCE}-ip" --region="$REGION"
# keep the bucket for reproducers; to remove later:
#   gcloud storage rm --recursive "gs://$BUCKET"
#   gcloud compute firewall-rules delete insitubatch-ssh
```

## Cost

- `n2-standard-32`: Spot ~$0.3–0.4/hr, on-demand ~$1.5/hr. **Stop or delete when done.**
- `g2-standard-16` (1x L4, 16 vCPU): Spot ~$0.4–0.6/hr, on-demand ~$1.3–1.6/hr (**VERIFY**
  current pricing). Local SSD is billed while the instance exists and wiped on stop.
- GCS storage: ~$0.02/GB-mo (standard, regional). Rapid Storage is priced higher — **VERIFY**.
- Static external IP (if used): a few cents a day while attached to a running VM; a
  reserved-but-unused address is billed at a higher rate — delete it at teardown.

## Notes

- Local SSD is ephemeral (wiped on stop/delete); the dataset stays in GCS, the mmap cache
  spills to `/mnt/nvme`.
- The multi-engine `python -m bench` suite needs torch (`workers`/`xbatcher` engines); the
  core probe runs without it.
- This file is a draft. As steps are validated on a real run, drop the **VERIFY** /
  **UNVERIFIED** markers and record the actual numbers in `benchmark_plan.md`.
