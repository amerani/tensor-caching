# Sources for costs and model parameters

Retrieved on 2 October 2026. Each value is labeled:

- **Sourced**: taken from a published figure.
- **Derived**: computed from sourced figures.
- **Assumption**: chosen by the author with no authoritative source found. Change these first if your situation differs.

Most price pages below are third-party aggregators that reproduce list prices, not the providers' own pages. Check the official pricing pages before relying on any number for a real decision.

## Model shape and KV size

| Value | Used | Status | Source |
|---|---|---|---|
| Layers, KV heads for Llama 3 70B | 80 layers, 8 KV heads | Sourced | Llama 3 paper, Table 3: https://arxiv.org/pdf/2407.21783 |
| Model dimension and attention heads | 8192 and 64, so head dimension 128 | Sourced, then derived | Same table; 8192 / 64 = 128 |
| KV bytes per token in fp16 | 327,680 (about 320 KB) | Derived | 2 x 80 x 8 x 128 x 2. Same result worked out at https://blog.prompt20.com/posts/kv-cache/ and https://blogs.vmware.com/cloud-foundation/2024/09/25/llm-inference-sizing-and-performance-guidance/ |
| 4x KV compression | Factor of 4 | Assumption | Nominal 4-bit versus 16-bit storage; ignores scale and metadata overhead and any quality loss |

## GPU cost

| Value | Used | Status | Source |
|---|---|---|---|
| Neocloud H100 price | $2.95 per GPU-hour | Sourced (secondary) | A May 2026 median across 58 providers from the AIMultiple index, as summarized at https://alatirok.com/gpu-cloud-pricing-2026/ |
| AWS H100 on-demand | $6.88 per GPU-hour | Sourced | https://www.spheron.network/blog/h100-price-per-hour-2026/ and https://www.thundercompute.com/blog/nvidia-h100-pricing (AWS p5.48xlarge listed at $55.04 per hour for 8 GPUs) |
| Marketplace low | $1.50 per GPU-hour | Assumption | Chosen from within the $1.38 to $1.49 marketplace floor reported at https://intuitionlabs.ai/articles/data-center-gpu-pricing-2026 and the $0.67 to $1.85 listings at https://gpufinder.dev/gpu/h100, rounded up |

Published H100 prices span roughly 4x to 8x depending on provider type (marketplace, neocloud, hyperscaler). The GPU price selector in the widget exists because this one input moves the answer more than almost any other.

## Prefill throughput (assumption, derived from FLOPs)

| Value | Used | Status |
|---|---|---|
| Prefill rate for one 8-GPU node | 20,000 tokens per second | Assumption, checked against a FLOP bound |
| Quadratic slowdown | Throughput divided by (1 + tokens / 200,000) | Assumption |

I could not find a reliable published prefill benchmark for Llama-3-70B on 8 H100s, so the rate is checked against a compute bound:

- A dense 70B model needs about 2 x 70 billion = 140 GFLOP per prefilled token.
- An H100 SXM is commonly quoted at roughly 989 TFLOPS dense BF16 peak (from NVIDIA's datasheet; this figure was not re-fetched when writing, so verify it). Eight GPUs give about 7.9 PFLOPS.
- 20,000 tokens per second x 140 GFLOP = 2.8 PFLOPS, around 35% of peak. That is a plausible utilization for large-batch prefill.

One published table (https://www.spheron.network/blog/llm-serving-optimization-continuous-batching-paged-attention/) lists 200 to 400 ms to prefill 32k tokens on a 70B model. That is below the compute bound even at full utilization on 8 GPUs, so it was not used.

The `200,000` constant in the quadratic term is a rough shape chosen so that prefill slows noticeably at several hundred thousand tokens. It has no source.

## Storage prices

| Tier | Used | Status | Source |
|---|---|---|---|
| Object storage | $0.023 per GB-month | Sourced | S3 Standard, US East, first 50 TB: https://www.cloudzero.com/blog/s3-pricing/ and https://www.usage.ai/blogs/aws/storage-cost/s3/cost-calculator/. Official: https://aws.amazon.com/s3/pricing/ |
| NVMe SSD | $0.08 per GB-month | Proxy | EBS gp3 list price, US East: https://cloudburn.io/blog/amazon-ebs-pricing and https://www.factualminds.com/tools/aws-ebs-pricing-calculator/ (rates dated 9 July 2026) |
| CPU RAM | $6.00 per GB-month | Derived, upper bound | AWS r7i memory-optimized instances price at about $6.04 per GiB-month all-in: r7i.large 16 GiB at $96.58 per month, r7i.4xlarge 128 GiB at $772.63, r7i.48xlarge 1,536 GiB at $9,271.58 (https://www.economize.cloud/resources/aws/pricing/ec2/r7i.4xlarge/) |

Caveats on these:

- **The NVMe price is a proxy.** EBS gp3 is network-attached block storage with throughput well below the 7 GB/s modeled for the NVMe tier. The widget pairs gp3's capacity price with local-NVMe bandwidth because local NVMe capacity is bundled into the instance price and has no separate per-GB rate. Treat the NVMe row as illustrative.
- **The RAM price includes CPUs.** Instance prices cover vCPUs too, so $6.00 overstates the cost of memory alone. On a GPU node, host RAM is often already paid for, which would make the marginal cost lower.
- **Request fees are ignored.** S3 GET requests are priced per thousand requests, which is negligible next to the storage cost for a handful of large objects per conversation.

## Load bandwidth and latency (assumptions)

| Tier | Bandwidth | Fixed latency | Status |
|---|---|---|---|
| CPU RAM | 50 GB/s | 2 ms | Assumption. PCIe 5.0 x16 is about 64 GB/s one way in theory; 50 is a practical figure |
| NVMe SSD | 7 GB/s | 5 ms | Assumption. Typical sequential read for a PCIe 4.0 x4 enterprise drive |
| Object storage | 1.5 GB/s | 150 ms | Assumption. Sustained single-node S3 download with parallel range requests varies widely with instance network and tuning |

None of these were sourced. Measure on your own hardware if latency matters, since the verdict line's latency comparison depends directly on them.

## Related systems

These projects implement tiered KV storage and are useful starting points for real numbers. They were not consulted for the model's parameters.

- LMCache: https://github.com/LMCache/LMCache
- Mooncake: https://github.com/kvcache-ai/Mooncake
- vLLM KV offloading: https://docs.vllm.ai
