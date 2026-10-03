# tensors-cache

An interactive widget for one narrow question: **for an archived or abandoned LLM conversation, is it cheaper to keep its KV cache around, or to delete it and re-prefill if the user comes back?**

The widget plots expected cost per 1,000 archived conversations against how long you retain the KV cache, and marks the sweet spot (or tells you there isn't one).

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

## Layout

```
tensors-cache/
  index.html          markup only
  css/styles.css      styling (light and dark mode)
  js/config.js        every tunable constant: model shape, prices, bandwidths, defaults
  js/model.js         pure cost model, no DOM access, also loadable from Node
  js/app.js           DOM wiring and Chart.js rendering
  vendor/             Chart.js 4.4.1 (MIT) and its license
  test/model.test.js  unit tests for the model
  docs/
    analysis.md       the re-prefill vs KV persistence analysis and results
    methodology.md    how the widget's model works, assumptions, limitations
    sources.md        where each GPU, storage, and model figure comes from
```

Scripts are plain classic `<script>` tags that share a `window.TC` namespace, not ES modules. That is deliberate: browsers block module scripts on `file://`, and this keeps Option 1 working.

## Changing the assumptions

Edit `js/config.js`. Prices, storage-tier bandwidths, prefill throughput, the KV size per token, and the slider defaults are all there. Reload the page. Before trusting a number, read `docs/sources.md`, which marks each value as sourced, derived, or an assumption.

## Caveats

This is a back-of-envelope model for building intuition, not a capacity-planning tool. It treats each conversation independently, assumes an exponential return-time distribution, and ignores queueing, peak-load effects, request fees, and shared prefixes. See `docs/methodology.md` for the full list.

## Licenses

Chart.js is MIT licensed; its license is in `vendor/CHARTJS_LICENSE.md`. This repository has no license of its own yet, so add one before inviting reuse.
