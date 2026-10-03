# tensors-cache

An interactive cost model for one narrow question: **for an idle LLM conversation (whose tensors are a single KV block), is it cheaper to keep its KV cache around, or to delete it and re-prefill if the user comes back? Persist or Recompute?**

The widget plots expected cost per 1,000 archived conversations against how long you retain the KV cache, and marks the sweet spot (or tells you there isn't one).

![Screenshot](https://raw.githubusercontent.com/amerani/tensors-cache/main/output.png)

> **TL;DR: Persist when daily storage cost is less than expected daily avoided recompute cost**

## Background: What is a KV cache?

A transformer processes a conversation as a sequence of tokens. At each layer, every token's hidden state (its intermediate activations) is projected into three vectors: a **query**, a **key**, and a **value**. Attention works by having each new token's query compare against the keys of all earlier tokens, then use those scores to blend their values.

The keys and values of earlier tokens never change once computed, so the model saves them instead of recomputing them. That saved set of tensors is the **KV cache**. It is not part of the model's weights. It is per-conversation state derived from that conversation's tokens.

Serving a request happens in two phases:

- **Prefill:** the model reads the whole prompt (system prompt, history, new message) in parallel and builds the KV cache for every token at every layer. This is compute-heavy and its cost grows faster than linearly with context length.
- **Decode:** the model generates one token at a time, reading the cache and appending one new key and value per layer per step. This is memory-bandwidth-bound.

**How big is it?** The cache holds a key and a value for every token, at every layer, for every KV head. Its size is therefore linear in conversation length:

```
bytes per token = 2 (K and V) × layers × KV heads × head dim × bytes per element
```

For a Llama-3-70B-class model in fp16 that is 2 × 80 × 8 × 128 × 2 = **327,680 bytes per token**. A 128k-token conversation is roughly 42 GB uncompressed, or about 10 GB at 4x compression. That is why this question is worth modeling: the cache is large, and it is expensive both to store and to rebuild.

**Why does it matter when a conversation goes idle?** GPU memory is scarce, so a conversation's cache is normally evicted soon after its last turn. When the user comes back, the server must either rebuild the cache with another prefill, or load a previously saved copy from slower storage (CPU memory, disk, object storage). This repo models that trade-off.

## Audience: Who is this for?

This is a back-of-envelope tool for building intuition, aimed at people making or reasoning about LLM serving decisions:

- **ML platform engineers** deciding whether a KV persistence tier is worth building.
- **Managers and stakeholders** who need a rough sense of the cost and latency trade-offs before committing resources.
- **Researchers and students** who want a concrete, inspectable example of the compute-versus-storage trade-off in LLM serving.

It is **not** a capacity-planning or billing tool. It assumes independent conversations, a single Llama-3-70B-class model shape, and idealized pricing, and it has not been validated against production traffic. If your workload has heavy prefix sharing (common system prompts, shared documents), very different return-time patterns, or a different attention architecture, treat the output as a starting point and replace the constants in `js/config.js` with your own measurements.

A basic familiarity with LLM inference is helpful. You do not need to know transformer internals beyond the refresher above.

## How the caching model works?

When a conversation goes idle, you can either **delete** its KV cache and re-prefill the whole conversation if the user returns, or **persist** it for up to `T` days and load it instead. The widget finds the `T` that minimizes expected cost per conversation. `T = 0` means "always re-prefill."

**Inputs:** conversation length, the chance the user ever returns (`P`), the typical time until return (`tau`), a storage tier (price, bandwidth, latency), KV compression (1x/2x/4x), and GPU price.

**Costs:**
- **Storage** is KV size × price per GB-month. KV size comes from tokens × bytes per token (327,680 for a Llama-3-70B-class fp16 model), divided by compression.
- **Re-prefill** is prefill time × the hourly price of a full 8-GPU node. Prefill time is linear in tokens with a mild quadratic correction for long contexts.
- **Return behavior:** with probability `P` the user returns after an exponentially distributed delay (mean `tau`). Otherwise they never come back and you pay storage for the full window.

**Expected cost** for a window `T` is the storage paid until return or expiry, plus the re-prefill you still owe if the user returns after `T`:

```
total(T) = c · (P·tau·(1 − e^(−T/tau)) + (1 − P)·T)  +  P·e^(−T/tau)·C
```

where `c` is storage cost per day and `C` is re-prefill cost.

**Rule of thumb:** a closed-form optimum exists, and retention pays only if the daily storage cost is below `P × C / tau`. The widget also reports the latency a returning user saves, and says so when loading isn't faster than re-prefilling.

**Validation:** unit tests check the code against the closed-form optimum and a brute-force search. These confirm the implementation matches the equations, not that the equations match real traffic.

**Key simplifications:** independent conversations (no shared prefixes), exponential return times, a flat average GPU price, no queueing, linear storage billing, free KV loading, a single model shape, and a 30-day cap on the window.

Full derivations and the complete assumptions list are in [`docs/methodology.md`](docs/methodology.md).

## View it in a browser

No build step and no internet connection are needed. Chart.js is vendored in `vendor/`.

**Option 1: open the file directly**

Double-click `index.html`, or from a terminal:

```bash
# macOS
open index.html
# Linux
xdg-open index.html
# Windows (PowerShell)
start index.html
```

**Option 2: serve it locally** (use this if your browser blocks something on `file://`, or you want to click through to the docs as rendered files)

```bash
python3 -m http.server 8000
# then visit http://localhost:8000
```

or, with Node:

```bash
npx serve .
```

## Run the tests

Requires Node 18 or newer. There are no dependencies to install.

```bash
node --test
```

The tests check the cost model against its own closed-form optimum and against a brute-force search.

## Licenses

Chart.js is MIT licensed; its license is in `vendor/CHARTJS_LICENSE.md`. 