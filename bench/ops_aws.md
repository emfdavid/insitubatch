# Phase 1 ops — EC2 + S3 for the cloud benchmark

Stand up a CPU EC2 box in **us-east-1** that reads ~200 GB of zarr from a
**regional, public, Requester-Pays** S3 bucket, so anyone with an AWS account can
reproduce the benchmark paying only their own egress.

> You run these; nothing here is run for you. Commands are copy-paste with the
> variables in the first block.

## Read first — the access model
- **Requester Pays is not anonymous-public.** Anonymous access is *denied* on a
  Requester-Pays bucket. "Public" here means **any authenticated AWS account** can
  read, and **the reader pays** their own GET + egress; the bucket owner pays
  storage only. Reproducers need an AWS account (which they need for EC2 anyway).
- **obstore supports it.** External readers pass `request_payer=True`, which flows
  through `store_from_url(...)` / `InSituDataset(..., request_payer=True)`. The
  bucket **owner** is *not* charged and does not need the flag for their own
  reads/writes on their own bucket.
- **Co-locate** bucket and instance in `us-east-1`. Cross-region egress is slow
  and billed.

## Variables

* c6id.8xlarge: 32 vCPU / 64 GiB + ~1.9 TB local NVMe (DiskCache)
* bucket names are global; suffix keeps it unique
```bash
export AWS_REGION=us-east-1
export ACCT=$(aws sts get-caller-identity --query Account --output text)
export BUCKET="insitubatch-bench-${ACCT}"
export KEY_NAME=emfdavid_ed25519
export INSTANCE_TYPE=c6id.8xlarge      
```

## 1. Import your SSH key (from ssh-agent)
`emfdavid_ed25519` is in your agent; export its *public* material and import it
(EC2 supports ed25519):
```bash
ssh-add -L | grep emfdavid_ed25519 > /tmp/${KEY_NAME}.pub
aws ec2 import-key-pair --region "$AWS_REGION" --key-name "$KEY_NAME" \
  --public-key-material "fileb:///tmp/${KEY_NAME}.pub"
```

## 2. S3 bucket — regional, public-read, Requester Pays
```bash
aws s3api create-bucket --bucket "$BUCKET" --region "$AWS_REGION"   # us-east-1 needs no LocationConstraint

# allow a public-read policy (disable Block Public Access on this bucket)
aws s3api put-public-access-block --bucket "$BUCKET" \
  --public-access-block-configuration \
  BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false

cat > /tmp/bucket-policy.json <<JSON
{ "Version": "2012-10-17", "Statement": [{
    "Sid": "PublicReadRequesterPays", "Effect": "Allow", "Principal": "*",
    "Action": ["s3:GetObject", "s3:ListBucket"],
    "Resource": ["arn:aws:s3:::$BUCKET", "arn:aws:s3:::$BUCKET/*"] }] }
JSON
aws s3api put-bucket-policy --bucket "$BUCKET" --policy file:///tmp/bucket-policy.json

aws s3api put-bucket-request-payment --bucket "$BUCKET" \
  --request-payment-configuration Payer=Requester
```

## 3. IAM instance profile (box reads/writes the bucket; no keys on disk)
```bash
cat > /tmp/trust.json <<'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
 "Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
aws iam create-role --role-name insitubatch-bench \
  --assume-role-policy-document file:///tmp/trust.json

cat > /tmp/s3-policy.json <<JSON
{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
 "Action":["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:ListBucket"],
 "Resource":["arn:aws:s3:::$BUCKET","arn:aws:s3:::$BUCKET/*"]}]}
JSON
aws iam put-role-policy --role-name insitubatch-bench \
  --policy-name s3-bench --policy-document file:///tmp/s3-policy.json

aws iam create-instance-profile --instance-profile-name insitubatch-bench
aws iam add-role-to-instance-profile \
  --instance-profile-name insitubatch-bench --role-name insitubatch-bench
```

## 4. Security group (SSH from your IP only) + default VPC lookups
```bash
export MYIP=$(curl -s https://checkip.amazonaws.com)
# If the next line prints "None", your account has no default VPC in this region.
# Create one (recreates default subnets + IGW + route table), then re-run it:
#   aws ec2 create-default-vpc --region "$AWS_REGION"
export VPC=$(aws ec2 describe-vpcs --region "$AWS_REGION" \
  --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
export SG=$(aws ec2 create-security-group --region "$AWS_REGION" \
  --group-name insitubatch-bench --description "insitubatch bench SSH" \
  --vpc-id "$VPC" --query GroupId --output text)
aws ec2 authorize-security-group-ingress --region "$AWS_REGION" \
  --group-id "$SG" --protocol tcp --port 22 --cidr "${MYIP}/32"
```

## 5. (Recommended) free S3 gateway endpoint — keeps S3 traffic on AWS, no NAT/egress
```bash
RT=$(aws ec2 describe-route-tables --region "$AWS_REGION" \
  --filters Name=vpc-id,Values=$VPC --query 'RouteTables[0].RouteTableId' --output text)
aws ec2 create-vpc-endpoint --region "$AWS_REGION" --vpc-id "$VPC" \
  --service-name "com.amazonaws.${AWS_REGION}.s3" --route-table-ids "$RT"
```

## 6. Launch (Spot, latest Amazon Linux 2023, public IP)
```bash
AMI=$(aws ssm get-parameters --region "$AWS_REGION" \
  --names /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
  --query 'Parameters[0].Value' --output text)
# Don't pin an AZ. In a default VPC, leaving --subnet-id off lets EC2 launch into
# the default subnet of whatever AZ has capacity — the fix for "InsufficientInstance-
# Capacity" in one zone. Nothing here ties you to an AZ: default subnets auto-assign a
# public IP, the SG is VPC-scoped, and the S3 gateway endpoint (step 5) is on the VPC
# main route table, so it applies in every AZ. To force a zone instead, set SUBNET to
# that AZ's default subnet, e.g.:
#   SUBNET=$(aws ec2 describe-subnets --region "$AWS_REGION" \
#     --filters Name=vpc-id,Values=$VPC Name=availability-zone,Values=us-east-1a \
#     --query 'Subnets[0].SubnetId' --output text)
SUBNET=""   # empty => EC2 picks an AZ with capacity

IID=$(aws ec2 run-instances --region "$AWS_REGION" \
  --image-id "$AMI" --instance-type "$INSTANCE_TYPE" --key-name "$KEY_NAME" \
  --security-group-ids "$SG" ${SUBNET:+--subnet-id "$SUBNET"} \
  --iam-instance-profile Name=insitubatch-bench \
  --instance-market-options 'MarketType=spot' \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=30,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=insitubatch-bench}]' \
  --query 'Instances[0].InstanceId' --output text)
echo "instance: $IID"

aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$IID"
IP=$(aws ec2 describe-instances --region "$AWS_REGION" --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "ssh -A ec2-user@$IP"
```

## 7. On the box — mount NVMe, install, generate, bench
```bash
export AWS_REGION=us-east-1            # obstore/object_store needs the region
export BUCKET=insitubatch-bench-808047988126

# --- mount the instance-store NVMe (ephemeral scratch for the DiskCache) ---
lsblk                                  # find the instance store (usually /dev/nvme1n1; root = nvme0n1)
sudo mkfs -t xfs /dev/nvme1n1
sudo mkdir -p /mnt/nvme && sudo mount /dev/nvme1n1 /mnt/nvme && sudo chown "$USER" /mnt/nvme

# --- install ---
curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.bashrc"
sudo yum install git
git clone git@github.com:emfdavid/insitubatch.git && cd insitubatch
uv sync --extra torch --extra bench

# --- generate the chunk-size family (owner creds -> no request_payer needed) ---
# (n, 721, 1440) f4 ~= 4.15 MB/sample; n=6000 ~= 25 GB per chunking, ~150 GB total.
for spc in 1 2 4 8 16 32; do
  uv run python bench/make_dataset.py --url "s3://$BUCKET/era5_c${spc}.zarr" \
    --sample-chunk "$spc" --n-samples 6000 --inner 721,1440
done

# --- run the suite + render Plotly graphs (DiskCache on the NVMe) ---
uv run python -m bench --full --url-prefix "s3://$BUCKET/era5" \
  --cache-dir /mnt/nvme/cache
```


### Get results:
```bash
scp ec2-user@$IP:/home/ec2-user/insitubatch/bench/results/suite.jsonl bench/results/exp_a.jsonl


```

## 8 back to local
```bash
scp ec2-user@$IP:insitubatch/bench/results/suite.jsonl /tmp/suite.jsonl

# in the repo
uv run python -m bench.plot --in /tmp/suite.jsonl --out docs/figures --cdn
```

## External reproducers (a different AWS account)
```python
from insitubatch import open_geometries, split_by_chunk
from insitubatch.source import InSituDataset

url = "s3://insitubatch-bench-<ACCT>/era5_fat.zarr"
geoms = open_geometries(url, request_payer=True)          # they pay their egress
manifest = split_by_chunk(geoms["t2m"], fractions=(0.8, 0.1, 0.1))
ds = InSituDataset(url, manifest, request_payer=True)      # store kwargs pass through
```

## Teardown
```bash

aws ec2 stop-instances --region "$AWS_REGION" --instance-ids "$IID"
aws ec2 start-instances --region "$AWS_REGION" --instance-ids "$IID"

aws ec2 terminate-instances --region "$AWS_REGION" --instance-ids "$IID"
# keep the bucket for reproducers; to remove later:
#   aws s3 rb "s3://$BUCKET" --force
# and tear down the IAM/SG if done:
#   aws iam remove-role-from-instance-profile --instance-profile-name insitubatch-bench --role-name insitubatch-bench
#   aws iam delete-instance-profile --instance-profile-name insitubatch-bench
#   aws iam delete-role-policy --role-name insitubatch-bench --policy-name s3-bench
#   aws iam delete-role --role-name insitubatch-bench
#   aws ec2 delete-security-group --region "$AWS_REGION" --group-id "$SG"
```

## Cost (us-east-1, approximate)
- `c6id.8xlarge`: Spot ~$0.5-0.7/hr, On-Demand ~$1.6/hr. **Terminate when done.**
  (Instance-store NVMe is included and ephemeral — wiped on stop/terminate.)
- S3 storage: ~$0.023/GB-mo → ~150 GB ≈ **$3.5/mo** (owner pays storage only).
- Requester Pays: reproducers pay their own GET + egress; owner pays $0 for reads.
- Optional: an AWS Budgets alarm to cap surprises.

## Notes
- The suite reads each `era5_c<spc>.zarr` once per config; the chunk-size sweep
  (`--full`) and `num_workers`/`compute_ms` sweeps come from `python -m bench`.
- The DiskCache lives on `/mnt/nvme` (instance store) — fast and ephemeral; the
  dataset stays in S3.
