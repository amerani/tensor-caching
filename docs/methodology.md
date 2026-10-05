# Methodology

This document describes the model behind the widget in `index.html`. The code is in `js/model.js`, every constant is in `js/config.js`, and the widget wiring is in `js/app.js`. Tests are in `test/model.test.js`.

The model has one organizing idea: **the fundamental object is a KV block, not a conversation.** A conversation is a set of blocks, some shared and some private, and cost is attributed per block. The original single-conversation model is the special case where nothing is shared.

## The decision being modeled

When a conversation goes idle, its KV cache (the per-token key and value tensors the model computed during prefill) can be handled two ways:

1. **Delete it.** If the user returns, run prefill again for whatever is not already cached.
2. **Persist it** in some storage tier for up to `T` days, then delete it. If the user returns within `T`, load the cache instead of recomputing. If they return later, re-prefill.

The widget searches for the `T` that minimizes expected cost per conversation. `T = 0` means "always re-prefill."

## Block-level KV accounting

### Blocks

A **block** holds the keys and values for `beta` consecutive tokens (default 16) across all layers and KV heads. Each block is identified by a chained hash:

```
id_j = H(id_(j-1), tokens_j)
```

Because `id_j` depends on every earlier token, two blocks are identical only if their entire prefixes are identical. Blocks therefore form a prefix tree, and a conversation is a root-to-leaf path through it. This is why only **prefix** content is shareable. A RAG document placed after a conversation-specific turn is not shared, even if its text matches another conversation's.

### Conversation as a set of blocks

Replace `conversation = KV object` with `conversation = set of KV blocks`:

```
B_i = S_i  U  U_i
```

- `S_i`: shared prefix blocks (system prompt, tool definitions, common skills, shared RAG documents, agent scaffolding). Other conversations hold references to the same physical blocks.
- `U_i`: conversation-specific blocks (the divergent tail: private history, unique documents, the partial final block).

For a conversation of `n` tokens, let `p` be the shared prefix length rounded down to a whole number of blocks, and `u = n - p`. Only complete blocks match, so any partial block at the boundary counts as unique.

### Shared-prefix economics

Separate reusable shared KV from private conversation KV. With `k = kvBytesPerToken / compression`:

```
shared_bytes = p x k
unique_bytes = u x k
```

The old model charged every conversation `tokens x bytes_per_token`. The block model charges:

```
effective_bytes = unique_bytes + shared_bytes x marginal_share
```

where `marginal_share` is `m` in `[0, 1]`: the fraction of a shared block's storage cost that this conversation is responsible for.

### Choosing `marginal_share`

There are two defensible definitions, and they answer different questions.

| Mode | Definition | Use it for |
| --- | --- | --- |
| **Average share** (default) | `m = 1 / n_s`, where `n_s` is the average number of conversations referencing each shared block | Fair attribution. Summed over all holders it equals the physical bytes stored, so nothing is double counted or lost. |
| **Marginal share** | `m = (1 - a)^(n_s - 1)`, the probability that none of the other `n_s - 1` sharers keeps the block alive through the window, where `a` is the chance each one does (`holdProb`) | The individual keep-or-delete decision: what extra bytes does keeping this one conversation cause? Approaches 0 when the pool is hot. |

Limiting cases:

- `m = 1` (nobody else shares it): shared bytes are charged in full. The model reduces to the single-conversation case.
- `m -> 0` (a very widely shared prefix): shared bytes are free to this conversation, and only `unique_bytes` matter.

The default is average share because it is simple and conserves bytes. Marginal share is usually smaller than average share when the pool is hot, so average share is the more conservative choice for the retain decision.

### Recompute under sharing

Prefill is causal: the work for the first `p` tokens is identical whether or not a later suffix exists. So the incremental cost of prefilling only the suffix, given a resident prefix, is:

```
C_suffix = C(n) - C(p)
```

where `C(.)` is the prefill cost function defined below. This is exact under the model's prefill-time formula, including the quadratic correction.

Let `h` be the probability that the shared prefix is still resident (in GPU memory or a warm tier) when the user returns, **independent of this conversation's retention**. Then the expected re-prefill cost is:

```
C_eff = h x (C(n) - C(p)) + (1 - h) x C(n)
      = C(n) - h x C(p)
```

The same split applies to load time, since only non-resident bytes need loading:

```
t_load = (unique_bytes + (1 - h) x shared_bytes) / bandwidth + fixedLatency
```

### Effective storage cost

```
size_eff_GB = (unique_bytes + shared_bytes x m) / 1e9
c_eff       = size_eff_GB x pricePerGBMonth / 30      (dollars per day)
```

## Inputs

| Input | Meaning | Default |
| --- | --- | --- |
| Conversation length | Total tokens `n` in the archived conversation | 128k |
| Shared prefix length | Tokens `p` that are an exact prefix shared with other conversations (rounded down to whole blocks) | 0 |
| Sharers (`n_s`) | Average conversations referencing each shared block; sets `m = 1/n_s` in average-share mode | 1 |
| Other-sharer hold probability (`a`) | Marginal-share mode only: chance each other sharer keeps a shared block alive | 50% |
| Prefix residency (`h`) | Probability the shared prefix is still cached when the user returns | 0% |
| Block size (`beta`) | Tokens per KV block | 16 |
| Chance the user ever returns (`P`) | Probability the conversation is ever resumed | 40% |
| Typical time until return (`tau`) | Mean delay before a returning user comes back, in days | 2 |
| Storage tier | Price per GB-month, load bandwidth, fixed latency | Object storage |
| KV precision | Compression factor applied to stored KV: 1x, 2x, 4x | 4x |
| GPU price | Dollars per GPU-hour used to cost re-prefill | $2.95 |

With the defaults for sharing (`p = 0`, `n_s = 1`, `h = 0`), the model is identical to the single-conversation model.

## Derived quantities

**KV bytes per token.** `kvBytesPerToken = 2 x 80 x 8 x 128 x 2 = 327,680` for a Llama-3-70B-class model in fp16 (2 for K and V, 80 layers, 8 KV heads, head dimension 128, 2 bytes per element). `k = kvBytesPerToken / compression`.

**Prefill time.** `t_prefill(m) = (m / R) x (1 + m / Q)` for `m` tokens, with `R = 20,000` tokens per second for one 8-GPU node and `Q = 200,000`. The second factor is a crude correction for attention's quadratic cost at long contexts.

**Prefill cost.** `C(m) = t_prefill(m) x (gpuPrice x 8) / 3600`. Re-prefill is billed as occupying the whole node for its duration.

**Load time.** See above; it depends on `h`.

## Return-time model

With probability `P` the user returns, and the delay is exponentially distributed with mean `tau`. With probability `1 - P` they never return. So the chance of a return after day `T` is `P x exp(-T/tau)`.

## Expected cost for a retention window `T`

Storage is paid until the user returns or `T` expires, whichever is first. Users who never return pay for the full `T`. The form is unchanged from the single-conversation model; only `c` and `C` are replaced by their effective values:

```
storage(T)   = c_eff x ( P x tau x (1 - exp(-T/tau)) + (1 - P) x T )
recompute(T) = P x exp(-T/tau) x C_eff
total(T)     = storage(T) + recompute(T)
```

The chart shows all three, multiplied by 1,000 and evaluated every 0.5 days from 0 to 30 days. The marker is the minimum of `total(T)` on that grid.

## Closed-form optimum

Setting `d total / dT = 0` gives:

```
c_eff x ( P x exp(-T/tau) + 1 - P )  =  (P x C_eff / tau) x exp(-T/tau)
```

Let `x = exp(-T/tau)`. Then:

```
x* = c_eff (1 - P) / ( P (C_eff/tau - c_eff) )
T* = -tau x ln(x*)
```

Retention pays at all only if `x* < 1`, which reduces to a simple rule:

> **Keep the KV blocks only if the daily effective storage cost is less than `P x C_eff / tau`**, the daily rate at which you expect to save re-prefill cost right after archiving.

The unit tests check the grid-search optimum against this formula.

### How sharing moves the decision

Define the storage-to-benefit ratio `rho = c_eff / (P x C_eff / tau)`. Retention pays when `rho < 1`. Relative to the single-conversation model:

```
rho_block / rho_single = (storage factor) / (recompute factor)

storage factor   = (u + m x p) / n
recompute factor = (C(n) - h x C(p)) / C(n)
```

- If sharing shrinks storage more than recompute (typical when `m` is small and `h` is moderate), retention gets **more** attractive.
- If the shared prefix is almost always resident (`h -> 1`), recompute collapses to `C(n) - C(p)` while storage may still include an unshared remainder. Retention gets **less** attractive.
- Because prefill cost is superlinear, the unique suffix is more expensive per token than the prefix it extends, which makes `C(n) - C(p)` larger than a linear estimate.

The direction depends on the workload, which is why both quantities are inputs.

### Worked example

Inputs: `n = 40,000`, `p = 30,000`, `u = 10,000`, 4x compression, `n_s = 1,000` (so `m = 0.001`), `h = 0.95`, `P = 0.4`, `tau = 2`, GPU price $2.95/hr, and an illustrative storage price of $0.02 per GB-month.

```
k           = 327,680 / 4 = 81,920 bytes per token
shared      = 30,000 x 81,920 = 2.4576 GB
unique      = 10,000 x 81,920 = 0.8192 GB
size_eff    = 0.8192 + 0.001 x 2.4576 = 0.8217 GB     (single model: 3.2768 GB)

cost per node-second = 2.95 x 8 / 3600 = $0.006556
t_prefill(40k) = (40,000/20,000) x (1 + 0.2)   = 2.400 s  -> C(n) = $0.01573
t_prefill(30k) = (30,000/20,000) x (1 + 0.15)  = 1.725 s  -> C(p) = $0.01131
C_eff = 0.01573 - 0.95 x 0.01131 = $0.00499               (single model: $0.01573)

c_eff (block)  = 0.8217 x 0.02 / 30 = $0.000548 per day   (single: $0.002185)
break-even     = 0.4 x 0.00499 / 2  = $0.000998 per day   (single: $0.003147)
rho            = 0.55                                     (single: 0.69)
```

Both models say persist. Sharing cuts storage about 4x and recompute about 3x, so the margin widens modestly. The storage price here is illustrative, not the widget's default.

## The shared pool's own retention decision

The shared blocks have an economics of their own, separate from any one conversation. Let `lambda` be the rate (per day) at which requests across all conversations touch the shared prefix, `S_GB` its size, and `c_s = S_GB x pricePerGBMonth / 30` its storage cost per day. Keeping it resident in a given tier pays off when:

```
lambda x C(p)  >  c_s
```

With a resident window `T_s` and Poisson arrivals, the hit probability is `h = 1 - exp(-lambda x T_s)`. This is where `h` in the per-conversation model comes from when you want to derive it rather than supply it. For a prefix touched thousands of times a day, `h` is effectively 1 and the shared bytes are nearly free relative to what they save. The tier matters: the same prefix may be worth holding in cheap storage but not in GPU memory.

## What the verdict line says

- If the best `T` is 0, the widget says to delete on archive and re-prefill on return.
- Otherwise it reports `T`, the savings versus always re-prefilling, and (when loading is faster than prefill) the latency a returning user saves.
- If loading from the chosen tier is no faster than re-prefilling, it says so, because then retention buys no latency either.
- When sharing is enabled, the verdict also reports how many bytes were attributed to this conversation versus the total, so it is clear how much of the answer comes from sharing.

## Assumptions and simplifications

1. **Prefix-only, exact-match sharing.** Blocks are shared only when the entire preceding token sequence is identical. Content that is the same but appears after divergent tokens (for example, the same RAG chunk in a different position) is not shared. Techniques that blend or re-position KV are out of scope.
2. **Sharing parameters are inputs.** `p`, `n_s`, and `h` are supplied by the user. The model does not simulate a population of conversations, evictions, or prefix-tree contention.
3. **Average-share attribution by default.** `m = 1/n_s` assumes sharers are symmetric and live over the same window. Real prefix popularity is skewed, and `n_s` varies by block depth (system prompt blocks have more sharers than document blocks deeper in the tree).
4. **Residency is independent of this conversation.** `h` does not depend on whether this particular conversation was retained. This is reasonable for a widely shared prefix and weaker for a prefix shared by only a few conversations.
5. **Block-boundary rounding only.** Per-block metadata, hash-table overhead, and fragmentation are ignored.
6. **Exponential return times.** Real return behavior is often heavier-tailed (many quick returns, a long tail of late ones).
7. **Average GPU price.** Re-prefill is costed at the same hourly rate whether the cluster is idle or saturated. In practice a burst of re-prefills at peak load costs more.
8. **No queueing.** Wait times assume an idle node.
9. **Storage billed linearly per day.** Real services have minimum durations, request fees, and tiered pricing, none modeled here.
10. **Load does not consume GPU time.** Copying KV back is treated as free of compute cost.
11. **One model shape.** Costs assume a Llama-3-70B-class GQA model. Models with multi-head latent attention have far smaller KV per token and would shift the result toward retention.
12. **Prefill throughput is derived, not measured.** See `sources.md`.
13. **Retention window capped at 30 days.** If the best point sits on the edge, the widget reports "30+ days".

## Validation

`node --test` runs unit tests that check:

- KV bytes per token equals 327,680.
- KV size scales linearly with tokens and inversely with compression.
- Prefill time is superlinear in tokens.
- `total(0) = P x C_eff` exactly, and the optimum is never worse than that baseline.
- The grid-search optimum agrees with a 0.01-day brute-force search and with the closed-form `T*`.

Tests to add with the block model:

- **Reduction.** With `p = 0`, or with `m = 1` and `h = 0`, every output equals the single-conversation model.
- **Conservation.** In average-share mode, summing `m x s_b` over all holders of a block recovers the block's physical size.
- **Prefill additivity.** `C(p) + (C(n) - C(p)) = C(n)`, and `C_eff` lies between `C(n) - C(p)` (at `h = 1`) and `C(n)` (at `h = 0`).
- **Monotonicity.** `c_eff` is non-increasing in `n_s`, and `C_eff` is non-increasing in `h`.
- **Rounding.** Shared prefix length is always a multiple of `beta`, and the remainder is charged as unique.

These confirm the code implements the equations above. They do not validate the equations against real traffic; that would need production return-time and prefix-sharing data.
