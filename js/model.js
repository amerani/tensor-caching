(function (root) {
  'use strict';

  var CONFIG = (typeof module !== 'undefined' && module.exports)
    ? require('./config.js')
    : root.TC.CONFIG;

  // Seconds to prefill `tokens` on one serving node. Throughput degrades as context grows
  // because attention cost is quadratic in sequence length.
  function prefillSeconds(tokens) {
    var p = CONFIG.prefill;
    return (tokens / p.nodeTokensPerSecond) * (1 + tokens / p.quadraticScaleTokens);
  }

  // Dollars to prefill once, occupying a whole node for the duration.
  function prefillCostUSD(tokens, gpuUsdPerHour) {
    var nodeUsdPerHour = gpuUsdPerHour * CONFIG.prefill.gpusPerNode;
    return prefillSeconds(tokens) * nodeUsdPerHour / 3600;
  }

  function kvSizeGB(tokens, compression) {
    return tokens * CONFIG.model.kvBytesPerToken / compression / 1e9;
  }

  function storageUsdPerDay(sizeGB, tier) {
    return sizeGB * tier.usdPerGBMonth / CONFIG.curve.daysPerMonth;
  }

  // Seconds to bring a cached KV block back to GPU memory.
  function loadSeconds(sizeGB, tier) {
    return sizeGB / tier.bandwidthGBps + tier.fixedLatencySec;
  }

  // Expected cost per conversation for a retention window of T days.
  //   Return model: with probability P the user returns, after an Exponential(mean tau) delay.
  //   Storage is paid until the user returns or T expires, whichever is first.
  //   Re-prefill is paid if the user returns after T.
  function expectedCost(T, P, tau, storagePerDay, prefillCost) {
    var storage = storagePerDay * (P * tau * (1 - Math.exp(-T / tau)) + (1 - P) * T);
    var recompute = P * Math.exp(-T / tau) * prefillCost;
    return { storage: storage, recompute: recompute, total: storage + recompute };
  }

  // params: { tokens, pReturn (0..1), tauDays, tierId, compression, gpuUsdPerHour }
  function evaluate(params) {
    var tier = CONFIG.tiers[params.tierId];
    var scale = CONFIG.conversationsScale;
    var sizeGB = kvSizeGB(params.tokens, params.compression);
    var perDay = storageUsdPerDay(sizeGB, tier);
    var pfSec = prefillSeconds(params.tokens);
    var pfCost = prefillCostUSD(params.tokens, params.gpuUsdPerHour);
    var ldSec = loadSeconds(sizeGB, tier);

    var points = [];
    var best = null;
    var c = CONFIG.curve;
    for (var T = 0; T <= c.maxRetentionDays + 1e-9; T += c.stepDays) {
      var e = expectedCost(T, params.pReturn, params.tauDays, perDay, pfCost);
      var pt = { day: T, storage: e.storage * scale, recompute: e.recompute * scale, total: e.total * scale };
      points.push(pt);
      if (best === null || pt.total < best.total - 1e-9) best = pt;
    }

    var baseline = params.pReturn * pfCost * scale; // retention window of zero days
    return {
      sizeGB: sizeGB,
      storageUsdPerDay: perDay,
      prefillSeconds: pfSec,
      prefillCostUSD: pfCost,
      loadSeconds: ldSec,
      points: points,
      best: best,
      baselineTotal: baseline,
      savingsPct: baseline > 0 ? (1 - best.total / baseline) * 100 : 0,
      retentionPays: best.day > 0,
      hitsMaxWindow: best.day >= c.maxRetentionDays,
      loadBeatsPrefill: ldSec < pfSec
    };
  }

  var api = {
    prefillSeconds: prefillSeconds,
    prefillCostUSD: prefillCostUSD,
    kvSizeGB: kvSizeGB,
    storageUsdPerDay: storageUsdPerDay,
    loadSeconds: loadSeconds,
    expectedCost: expectedCost,
    evaluate: evaluate
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  } else {
    root.TC = root.TC || {};
    root.TC.model = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
