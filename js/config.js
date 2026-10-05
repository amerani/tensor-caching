/* Every constant used by the model lives here.
 * Works as a classic <script> (sets window.TC_CONFIG) and as a Node module. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.TC_CONFIG = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  return {
    // Llama-3-70B-class GQA model, fp16: 2 x 80 x 8 x 128 x 2 = 327,680 bytes/token
    kv: { layers: 80, kvHeads: 8, headDim: 128, bytesPerElement: 2 },

    // Prefill: t(m) = (m / R) * (1 + m / Q) for one 8-GPU node
    prefill: { tokensPerSecond: 20000, quadraticTokens: 200000, gpusPerNode: 8 },

    // Storage tiers. NOTE: replace these with your own measured/quoted values.
    tiers: [
      { id: 'cpu',    label: 'Host memory (CPU RAM)', pricePerGBMonth: 3.0,   bandwidthGBps: 25,  latencySeconds: 0.01 },
      { id: 'nvme',   label: 'Local NVMe SSD',        pricePerGBMonth: 0.10,  bandwidthGBps: 5,   latencySeconds: 0.05 },
      { id: 'object', label: 'Object storage',        pricePerGBMonth: 0.023, bandwidthGBps: 1.5, latencySeconds: 0.2 }
    ],

    compressionOptions: [1, 2, 4],

    // Retention-window grid
    windowDays: 30,
    stepDays: 0.5,
    daysPerMonth: 30,
    perConversations: 1000,

    defaults: {
      tokens: 128000,        // n: total tokens in the archived conversation
      sharedTokens: 0,       // p: exact shared prefix (rounded down to whole blocks)
      sharers: 1,            // n_s: conversations referencing each shared block
      shareMode: 'average',  // 'average' (m = 1/n_s) or 'marginal'
      holdProb: 0.5,         // marginal mode: chance each other sharer keeps the block alive
      residency: 0,          // h: chance the shared prefix is still cached on return
      blockTokens: 16,       // beta: tokens per KV block
      pReturn: 0.4,          // P
      tauDays: 2,            // tau
      tierId: 'object',
      compression: 4,
      gpuPrice: 2.95         // dollars per GPU-hour
    }
  };
});
