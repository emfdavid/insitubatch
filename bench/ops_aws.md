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
```bash
export AWS_REGION=us-east-1
export ACCT=$(aws sts get-caller-identity --query Account --output text)
export BUCKET="insitubatch-bench-${ACCT}"      # bucket names are global; suffix keeps it unique
export KEY_NAME=emfdavid_ed25519
export INSTANCE_TYPE=c7i.4xlarge               # 16 vCPU / 32 GiB; scale up for more NIC
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
MYIP=$(curl -s https://checkip.amazonaws.com)
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
SUBNET=$(aws ec2 describe-subnets --region "$AWS_REGION" \
  --filters Name=vpc-id,Values=$VPC Name=default-for-az,Values=true \
  --query 'Subnets[0].SubnetId' --output text)

IID=$(aws ec2 run-instances --region "$AWS_REGION" \
  --image-id "$AMI" --instance-type "$INSTANCE_TYPE" --key-name "$KEY_NAME" \
  --security-group-ids "$SG" --subnet-id "$SUBNET" --associate-public-ip-address \
  --iam-instance-profile Name=insitubatch-bench \
  --instance-market-options 'MarketType=spot' \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=30,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=insitubatch-bench}]' \
  --query 'Instances[0].InstanceId' --output text)
echo "instance: $IID"

aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$IID"
IP=$(aws ec2 describe-instances --region "$AWS_REGION" --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "ssh ec2-user@$IP"
```

## 7. On the box — install, generate ~200 GB, bench
```bash
ssh ec2-user@$IP
export AWS_REGION=us-east-1            # obstore/object_store needs the region
export BUCKET=insitubatch-bench-<ACCT> # same value as above

curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.bashrc"
git clone https://github.com/emfdavid/insitubatch && cd insitubatch
uv sync --extra torch

# Generate (owner creds via instance profile -> no request_payer needed).
# Sizing: (n, 721, 1440) f4 ~= 4.15 MB/sample. n=24000 ~= 100 GB per dataset.
uv run python bench/make_dataset.py --url "s3://$BUCKET/era5_fat.zarr"  --regime fat  --n-samples 24000 --inner 721,1440
uv run python bench/make_dataset.py --url "s3://$BUCKET/era5_grib.zarr" --regime grib --n-samples 24000 --inner 721,1440

uv run python bench/bench_throughput.py --url "s3://$BUCKET/era5_fat.zarr"
uv run python bench/bench_throughput.py --url "s3://$BUCKET/era5_grib.zarr"
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
- `c7i.4xlarge`: Spot ~$0.30/hr, On-Demand ~$0.71/hr. **Terminate when done.**
- S3 storage: ~$0.023/GB-mo → 200 GB ≈ **$4.6/mo** (owner pays storage only).
- Requester Pays: reproducers pay their own GET + egress; owner pays $0 for reads.
- Optional: an AWS Budgets alarm to cap surprises.

## Known rough edges (Phase-1 code work, not ops)
- `bench_throughput.py --url` currently runs both regime *configs* against the one
  URL you pass; for clean cloud numbers it wants a small refinement to take its
  config from the dataset and emit per-dataset rows.
- The bench has no compute step, so prefetch overlap won't show as throughput;
  add a simulated step (or a real model) to surface the GPU-fed win.
