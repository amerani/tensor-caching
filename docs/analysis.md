# Re-prefill vs KV persistence for archived conversations

## Question

When a conversation goes idle, should a serving system keep its KV cache so a returning user skips prefill, or delete it and recompute on return?

This analysis restricts itself to one case: **archived or abandoned conversations**. These sit idle for hours to weeks, and each belongs to a single user. That is the case where persistence is hardest to justify, because nothing amortizes the storage cost across many requests.

All numbers below come from the model in `js/model.js` with the defaults in `js/config.js`, so you can reproduce them in the widget. Treat them as shapes and orders of magnitude. See `methodology.md` for the model and `sources.md` for how much each input can be trusted.

## The tradeoff in one rule

Persistence pays only if the daily storage cost of the KV cache is smaller than `P x C / tau`, where:

- `P` is the chance the user ever returns,
- `C` is the dollar cost of one re-prefill,
- `tau` is the mean number of days until a returning user comes back.

The right side is the rate at which you expect to save re-prefill cost right after archiving. If storage costs more than that per day, deleting immediately is cheaper. If it costs less, there is a retention window after which the saving rate falls below the storage rate, and the cache should be deleted.

## Results at the default scenario

Defaults: 128k-token conversation, 40% chance of return, 2-day mean return time, Llama-3-70B-class KV (320 KB per token in fp16), $2.95 per GPU-hour on an 8-GPU node.

### Storage tier and KV precision

| Tier / precision | KV size | Re-prefill cost | Storage per day | Best retention | Saving vs always re-prefill |
|---|---|---|---|---|---|
| RAM, fp16 | 41.9 GB | 6.88¢ | 838.9¢ | none | n/a |
| RAM, 4-bit | 10.5 GB | 6.88¢ | 209.7¢ | none | n/a |
| NVMe, fp16 | 41.9 GB | 6.88¢ | 11.2¢ | none | n/a |
| NVMe, 4-bit | 10.5 GB | 6.88¢ | 2.8¢ | none | n/a |
| Object storage, fp16 | 41.9 GB | 6.88¢ | 3.2¢ | none | n/a |
| Object storage, fp8 | 21.0 GB | 6.88¢ | 1.6¢ | none | n/a |
| Object storage, 4-bit | 10.5 GB | 6.88¢ | 0.8¢ | 1.5 days | 14% |

Only one of the nine combinations has a sweet spot. At these settings a single day of storage costs more than the whole re-prefill for everything except compressed KV on the cheapest tier.

### Conversation length (object storage, 4-bit)

| Tokens | KV size | Re-prefill time | Re-prefill cost | Load time | Best retention | Saving |
|---|---|---|---|---|---|---|
| 8k | 0.7 GB | 0.4 s | 0.27¢ | 0.6 s | about 0.5 days | ~0% |
| 32k | 2.6 GB | 1.9 s | 1.22¢ | 1.9 s | 0.5 days | 2% |
| 128k | 10.5 GB | 10.5 s | 6.88¢ | 7.1 s | 1.5 days | 14% |
| 512k | 41.9 GB | 91.1 s | 59.74¢ | 28.1 s | 3.5 days | 45% |

Storage grows linearly with length, but prefill cost grows faster because attention is quadratic. Longer conversations therefore favor persistence, and short ones almost never justify it.

### Return behavior (128k, object storage, 4-bit)

| Chance of return | Mean time to return | Best retention | Saving |
|---|---|---|---|
| 10% | 0.5 days | 0.5 days | 7% |
| 10% | 2 days | none | n/a |
| 40% | 0.5 days | 1 day | 64% |
| 40% | 2 days | 1.5 days | 14% |
| 40% | 7 days | none | n/a |
| 80% | 0.5 days | 2 days | 87% |
| 80% | 2 days | 5 days | 56% |
| 80% | 7 days | none | n/a |

Return behavior dominates everything else. A slow return rate (a week or more) kills retention even when the chance of return is high, because you pay storage for days while the savings arrive diffusely.

### GPU price (128k, object storage, 4-bit, 40% return, 2 days)

| GPU price | Re-prefill cost | Best retention | Saving |
|---|---|---|---|
| $1.50 per GPU-hr | 3.50¢ | none | n/a |
| $2.95 per GPU-hr | 6.88¢ | 1.5 days | 14% |
| $6.88 per GPU-hr | 16.05¢ | 3.5 days | 48% |

Cheap compute pushes toward re-prefill and expensive compute toward retention. A hyperscaler on-demand price makes retention look much better than a neocloud price does, which shows how much this decision depends on what your GPUs actually cost you.

## Findings

1. **For cold archives, re-prefilling is usually cheaper.** Raw fp16 KV on RAM or SSD costs more per day than recomputing. Persistence needs cheap storage, compressed KV, long contexts, and fast returns together.
2. **Latency is the better argument, not cost.** Fast tiers (RAM, NVMe) load a 128k-token cache in under two seconds versus about ten seconds to re-prefill, but they are the expensive tiers. Cheap object storage loads in about seven seconds, barely faster than recomputing.
3. **Compression changes the answer.** Going from fp16 to 4-bit shrinks storage cost fourfold. In the default scenario it is the difference between "never keep" and "keep for a day and a half" on object storage. The model does not charge for any quality loss from quantized KV.
4. **Time-to-return matters more than probability of return.** Both enter the rule through `P / tau`, but `tau` has a wider realistic range.
5. **The window is short.** Even when persistence wins, the optimum here is hours to a few days. This points to a tiered policy: a fast tier for the first hours, a cheap tier for a day or two, then deletion.

## Practical policy for archived conversations

1. Measure your own return-time distribution. It is the most sensitive input and the one the model can only assume.
2. Default to delete on archive for short conversations (under about 32k tokens here).
3. For long conversations, keep compressed KV on the cheapest tier for a window set by `T* = -tau x ln(c(1-P) / (P(C/tau - c)))`, then delete.
4. If returning users are latency-sensitive, price the latency separately. A faster tier may be worth its cost even where this model says it loses.

## What this analysis leaves out

- **Shared prefixes.** A system prompt or document shared by many conversations amortizes one cached prefill across all of them. That is the strongest case for persistence and is not modeled.
- **Peak-load effects.** Re-prefill is costed at an average GPU price. A burst of returning users at peak load stresses the cluster more than an average-price model shows.
- **Fees beyond storage.** Request charges, minimum storage durations, and data-transfer costs are ignored.
- **Heavy-tailed returns.** Real users often return quickly or not at all, which is not an exponential distribution.
- **Other architectures.** Models with much smaller KV per token (for example multi-head latent attention) would shift every row toward retention.
- **Quality cost of KV quantization.**

## Reproducing these tables

```bash
node -e "
const m = require('./js/model.js');
console.log(m.evaluate({tokens:128000,pReturn:0.4,tauDays:2,tierId:'object',compression:4,gpuUsdPerHour:2.95}));
"
```

Change the fields to match any row above. The widget runs the same function.
