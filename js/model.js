/* Block-level KV caching cost model. See docs/methodology.md.
 *
 * A conversation is a set of KV blocks: shared prefix blocks + unique blocks.
 *   effective_bytes = unique_bytes + shared_bytes x marginal_share
 *   C_eff           = C(n) - h x C(p)
 * With p = 0 (or m = 1 and h = 0) this reduces to the single-conversation model.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./config.js'));
  else root.TCModel = factory(root.TC_CONFIG);
})(typeof self !== 'undefined' ? self : this, function (CONFIG) {
  'use strict';

  const GB = 1e9;
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  /* ---------- primitives ---------- */

  function kvBytesPerToken(kv) {
    kv = kv || CONFIG.kv;
    return 2 * kv.layers * kv.kvHeads * kv.headDim * kv.bytesPerElement;
  }

  function prefillSeconds(tokens, pf) {
    pf = pf || CONFIG.prefill;
    if (tokens <= 0) return 0;
    return (tokens / pf.tokensPerSecond) * (1 + tokens / pf.quadraticTokens);
  }

  function prefillCost(tokens, gpuPrice, pf) {
    pf = pf || CONFIG.prefill;
    return (prefillSeconds(tokens, pf) * gpuPrice * pf.gpusPerNode) / 3600;
  }

  /* ---------- block accounting ---------- */

  // Only whole blocks can match, so the shared prefix is rounded down to a
  // multiple of the block size; the remainder is charged as unique.
  function splitTokens(total, shared, blockTokens) {
    const n = Math.max(0, Math.floor(total));
    const b = Math.max(1, Math.floor(blockTokens));
    const wanted = clamp(Math.floor(shared || 0), 0, n);
    const p = Math.floor(wanted / b) * b;
    return { total: n, shared: p, unique: n - p, blockTokens: b, sharedBlocks: p / b, totalBlocks: Math.ceil(n / b) };
  }

  // Fraction of a shared block's storage cost charged to one conversation.
  //  average : 1 / n_s            (conserves bytes across holders)
  //  marginal: (1 - a)^(n_s - 1)  (no other sharer keeps the block alive)
  function marginalShare(opts) {
    const ns = Math.max(1, opts.sharers || 1);
    if (opts.shareMode === 'marginal') {
      const a = clamp(opts.holdProb == null ? 0.5 : opts.holdProb, 0, 1);
      return Math.pow(1 - a, ns - 1);
    }
    return 1 / ns;
  }

  function withDefaults(inputs) {
    const x = Object.assign({}, CONFIG.defaults, inputs || {});
    x.tier = x.tier || CONFIG.tiers.find(t => t.id === x.tierId) || CONFIG.tiers[0];
    x.residency = clamp(x.residency, 0, 1);
    x.pReturn = clamp(x.pReturn, 0, 1);
    x.tauDays = Math.max(1e-9, x.tauDays);
    return x;
  }

  // Effective (per-conversation) quantities from the inputs.
  function effective(inputs) {
    const x = withDefaults(inputs);
    const s = splitTokens(x.tokens, x.sharedTokens, x.blockTokens);
    const k = kvBytesPerToken() / x.compression;
    const sharedBytes = s.shared * k;
    const uniqueBytes = s.unique * k;
    const m = marginalShare(x);
    const h = x.residency;

    const effGB = (uniqueBytes + sharedBytes * m) / GB;
    const totalGB = (uniqueBytes + sharedBytes) / GB;
    const perDay = gb => (gb * x.tier.pricePerGBMonth) / CONFIG.daysPerMonth;

    const Cn = prefillCost(s.total, x.gpuPrice);
    const Cp = prefillCost(s.shared, x.gpuPrice);
    const tPrefillEff = prefillSeconds(s.total) - h * prefillSeconds(s.shared);
    const loadGB = (uniqueBytes + (1 - h) * sharedBytes) / GB;

    return {
      inputs: x, split: s, marginalShare: m,
      sharedGB: sharedBytes / GB, uniqueGB: uniqueBytes / GB, effGB, totalGB,
      cEff: perDay(effGB), cSingle: perDay(totalGB),
      CEff: Cn - h * Cp, CSingle: Cn, Cprefix: Cp,
      tPrefillEff, tLoad: loadGB / x.tier.bandwidthGBps + x.tier.latencySeconds
    };
  }

  /* ---------- expected cost ---------- */

  function costAt(T, c, C, P, tau) {
    const storage = c * (P * tau * (1 - Math.exp(-T / tau)) + (1 - P) * T);
    const recompute = P * Math.exp(-T / tau) * C;
    return { T, storage, recompute, total: storage + recompute };
  }

  // Retention pays only if c < P*C/tau.
  function shouldRetain(c, C, P, tau) { return c < (P * C) / tau; }

  // Closed-form optimum. Returns 0 when retention never pays; Infinity if P = 1.
  function closedFormT(c, C, P, tau) {
    if (!shouldRetain(c, C, P, tau)) return 0;
    if (P >= 1) return Infinity;
    const x = (c * (1 - P)) / (P * (C / tau - c));
    if (x <= 0) return Infinity;
    return Math.max(0, -tau * Math.log(x));
  }

  function grid(maxDays, step) {
    const out = [];
    const n = Math.round(maxDays / step);
    for (let i = 0; i <= n; i++) out.push(i * step);
    return out;
  }

  function curve(inputs, step) {
    const e = effective(inputs), x = e.inputs;
    return grid(CONFIG.windowDays, step || CONFIG.stepDays)
      .map(T => costAt(T, e.cEff, e.CEff, x.pReturn, x.tauDays));
  }

  function bruteForceT(inputs, step) {
    const pts = curve(inputs, step);
    return pts.reduce((b, p) => (p.total < b.total - 1e-15 ? p : b), pts[0]);
  }

  function optimize(inputs) {
    const e = effective(inputs), x = e.inputs;
    const pts = curve(inputs);
    const best = pts.reduce((b, p) => (p.total < b.total - 1e-15 ? p : b), pts[0]);
    const baseline = pts[0];
    const mult = CONFIG.perConversations;
    const loadFaster = e.tLoad < e.tPrefillEff;
    return {
      eff: e, points: pts, best, baseline,
      retain: shouldRetain(e.cEff, e.CEff, x.pReturn, x.tauDays),
      closedFormT: closedFormT(e.cEff, e.CEff, x.pReturn, x.tauDays),
      atCap: best.T >= CONFIG.windowDays,
      savingsPerBatch: (baseline.total - best.total) * mult,
      loadFaster,
      latencySavedSeconds: loadFaster ? e.tPrefillEff - e.tLoad : 0,
      attribution: { effGB: e.effGB, totalGB: e.totalGB, fraction: e.totalGB > 0 ? e.effGB / e.totalGB : 1 },
      multiplier: mult
    };
  }

  /* ---------- shared pool's own retention decision ---------- */

  // Poisson arrivals at lambda requests/day, resident window Ts days.
  function hitProbability(lambdaPerDay, windowDays) {
    return 1 - Math.exp(-Math.max(0, lambdaPerDay) * Math.max(0, windowDays));
  }

  // Keep the shared prefix resident iff lambda x C(p) > c_s.
  function poolRetention(inputs) {
    const e = effective(inputs), x = e.inputs;
    const cS = (e.sharedGB * x.tier.pricePerGBMonth) / CONFIG.daysPerMonth;
    const benefit = (x.requestsPerDay || 0) * e.Cprefix;
    return { storagePerDay: cS, benefitPerDay: benefit, keep: benefit > cS };
  }

  /* ---------- verdict ---------- */

  function describe(r) {
    const f = (d, n) => d.toFixed(n == null ? 1 : n);
    const money = v => '$' + (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2));
    if (!r.retain || r.best.T === 0) {
      return 'Delete on archive and re-prefill on return. Storage costs more than the re-prefill it would save.';
    }
    const window = r.atCap ? CONFIG.windowDays + '+ days' : f(r.best.T) + ' days';
    let s = 'Persist for about ' + window + ', saving ' + money(r.savingsPerBatch) +
            ' per ' + r.multiplier.toLocaleString('en-US') + ' conversations versus always re-prefilling.';
    s += r.loadFaster
      ? ' A returning user saves about ' + f(r.latencySavedSeconds, 1) + ' s.'
      : ' Loading from this tier is no faster than re-prefilling, so retention buys no latency.';
    if (r.eff.split.shared > 0) {
      s += ' Charged ' + f(r.attribution.effGB, 2) + ' GB of ' + f(r.attribution.totalGB, 2) + ' GB (shared prefix discounted).';
    }
    return s;
  }

  return {
    kvBytesPerToken, prefillSeconds, prefillCost,
    splitTokens, marginalShare, effective,
    costAt, shouldRetain, closedFormT, curve, bruteForceT, optimize,
    hitProbability, poolRetention, describe
  };
});
