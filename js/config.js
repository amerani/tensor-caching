(function (root) {
  'use strict';

  // All tunable constants live here. See docs/sources.md for where each value comes from
  // and which ones are assumptions rather than sourced figures.
  var CONFIG = {
    model: {
      name: 'Llama-3-70B-class (GQA, 8 KV heads)',
      layers: 80,
      kvHeads: 8,
      headDim: 128,
      bytesPerElement: 2, // fp16 / bf16
      // 2 (K and V) x layers x kvHeads x headDim x bytes = 327,680 bytes (~320 KB) per token
      kvBytesPerToken: 2 * 80 * 8 * 128 * 2
    },

    prefill: {
      gpusPerNode: 8,
      nodeTokensPerSecond: 20000, // assumption, derived from FLOPs; see docs/sources.md
      quadraticScaleTokens: 200000 // assumption: prefill slows by (1 + tokens / this)
    },

    gpuPricesPerHour: [
      { id: 'neocloud', label: 'Neocloud median ($2.95 per GPU-hr)', usd: 2.95 },
      { id: 'aws', label: 'AWS p5 ($6.88 per GPU-hr)', usd: 6.88 },
      { id: 'marketplace', label: 'Marketplace low ($1.50 per GPU-hr)', usd: 1.5 }
    ],

    tiers: {
      ram: {
        label: 'CPU RAM ($6.00/GB-mo, 50 GB/s)',
        usdPerGBMonth: 6.0,
        bandwidthGBps: 50,
        fixedLatencySec: 0.002
      },
      ssd: {
        label: 'NVMe SSD ($0.08/GB-mo proxy, 7 GB/s)',
        usdPerGBMonth: 0.08,
        bandwidthGBps: 7,
        fixedLatencySec: 0.005
      },
      object: {
        label: 'Object ($0.023/GB-mo, 1.5 GB/s)',
        usdPerGBMonth: 0.023,
        bandwidthGBps: 1.5,
        fixedLatencySec: 0.15
      }
    },

    compression: [
      { factor: 1, label: 'fp16 (1x)' },
      { factor: 2, label: 'fp8 (2x smaller)' },
      { factor: 4, label: '4-bit (4x smaller)' }
    ],

    defaults: {
      tokensK: 128,
      pReturnPct: 40,
      tauDays: 2,
      gpuPriceId: 'neocloud',
      tierId: 'object',
      compression: 4
    },

    curve: { maxRetentionDays: 30, stepDays: 0.5, daysPerMonth: 30 },
    conversationsScale: 1000
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = CONFIG;
  } else {
    root.TC = root.TC || {};
    root.TC.CONFIG = CONFIG;
  }
})(typeof window !== 'undefined' ? window : globalThis);
