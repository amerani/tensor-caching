# Methodology

This document describes the model behind the widget in `index.html`. The code is in `js/model.js`, and every constant is in `js/config.js`.

## The decision being modeled

When a conversation goes idle, its KV cache (the per-token key and value tensors the model computed during prefill) can be handled two ways:

1. **Delete it.** If the user returns, run prefill over the whole conversation again.
2. **Persist it** in some storage tier for up to `T` days, then delete it. If the user returns within `T`, load the cache instead of recomputing. If they return later, re-prefill.

The widget searches for the `T` that minimizes expected cost per conversation. `T = 0` means "always re-prefill."

## Inputs

| Input | Meaning | Default |
|---|---|---|
| Conversation length | Tokens in the archived conversation | 128k |
| Chance the user ever returns (`P`) | Probability the conversation is ever resumed | 40% |
| Typical time until return (`tau`) | Mean delay before a returning user comes back, in days | 2 |
| Storage tier | Price per GB-month, load bandwidth, fixed latency | Object storage |
| KV precision | Compression factor applied to stored KV: 1x, 2x, 4x | 4x |
| GPU price | Dollars per GPU-hour used to cost re-prefill | $2.95 |

## Derived quantities

**KV size.** `size_GB = tokens x kvBytesPerToken / compression / 1e9`, where `kvBytesPerToken = 2 x 80 x 8 x 128 x 2 = 327,680` for a Llama-3-70B-class model in fp16 (2 for K and V, 80 layers, 8 KV heads, head dimension 128, 2 bytes per element).

**Prefill time.** `t_prefill = (tokens / R) x (1 + tokens / Q)`, with `R = 20,000` tokens per second for one 8-GPU node and `Q = 200,000`. The second factor is a crude correction for attention's quadratic cost at long contexts.

**Prefill cost.** `C = t_prefill x (gpuPrice x 8) / 3600`. Re-prefill is billed as occupying the whole node for its duration.

**Storage cost rate.** `c = size_GB x pricePerGBMonth / 30` dollars per day.

**Load time.** `t_load = size_GB / bandwidth + fixedLatency`.

## Return-time model

With probability `P` the user returns, and the delay is exponentially distributed with mean `tau`. With probability `1 - P` they never return. So the chance of a return after day `T` is `P x exp(-T/tau)`.

## Expected cost for a retention window `T`

Storage is paid until the user returns or `T` expires, whichever is first. Users who never return pay for the full `T`.

```
storage(T)   = c x ( P x tau x (1 - exp(-T/tau)) + (1 - P) x T )
recompute(T) = P x exp(-T/tau) x C
total(T)     = storage(T) + recompute(T)
```

The chart shows all three, multiplied by 1,000 and evaluated every 0.5 days from 0 to 30 days. The marker is the minimum of `total(T)` on that grid.

## Closed-form optimum

Setting `d total / dT = 0` gives:

```
c x ( P x exp(-T/tau) + 1 - P )  =  (P x C / tau) x exp(-T/tau)
```

Let `x = exp(-T/tau)`. Then:

```
x* = c (1 - P) / ( P (C/tau - c) )
T* = -tau x ln(x*)
```

Retention pays at all only if `x* < 1`, which reduces to a simple rule:

> **Keep the KV cache only if the daily storage cost is less than `P x C / tau`**, the daily rate at which you expect to save re-prefill cost right after archiving.

The unit tests check the grid-search optimum against this formula.

## What the verdict line says

- If the best `T` is 0, the widget says to delete on archive and re-prefill on return.
- Otherwise it reports `T`, the savings versus always re-prefilling, and (when loading is faster than prefill) the latency a returning user saves.
- If loading from the chosen tier is no faster than re-prefilling, it says so, because then retention buys no latency either.

## Assumptions and simplifications

1. **Independent conversations.** No shared prefixes (system prompts, common documents), which is where persistence usually pays off most.
2. **Exponential return times.** Real return behavior is often heavier-tailed (many quick returns, a long tail of late ones).
3. **Average GPU price.** Re-prefill is costed at the same hourly rate whether the cluster is idle or saturated. In practice a burst of re-prefills at peak load costs more.
4. **No queueing.** Wait times assume an idle node.
5. **Storage billed linearly per day.** Real services have minimum durations, request fees, and tiered pricing, none modeled here.
6. **Load does not consume GPU time.** Copying KV back is treated as free of compute cost.
7. **One model shape.** Costs assume a Llama-3-70B-class GQA model. Models with multi-head latent attention have far smaller KV per token and would shift the result toward retention.
8. **Prefill throughput is derived, not measured.** See `sources.md`.
9. **Retention window capped at 30 days.** If the best point sits on the edge, the widget reports "30+ days".

## Validation

`node --test` runs unit tests that check:

- KV bytes per token equals 327,680.
- KV size scales linearly with tokens and inversely with compression.
- Prefill time is superlinear in tokens.
- `total(0) = P x C` exactly, and the optimum is never worse than that baseline.
- The grid-search optimum agrees with a 0.01-day brute-force search and with the closed-form `T*`.

These confirm the code implements the equations above. They do not validate the equations against real traffic; that would need production return-time data.
