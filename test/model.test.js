'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const CONFIG = require('../js/config.js');
const model = require('../js/model.js');

const base = { tokens: 128000, pReturn: 0.4, tauDays: 2, tierId: 'object', compression: 4, gpuUsdPerHour: 2.95 };

test('KV bytes per token matches Llama-3-70B GQA-8 fp16 (327,680)', () => {
  assert.equal(CONFIG.model.kvBytesPerToken, 327680);
});

test('KV size scales linearly with tokens and inversely with compression', () => {
  const a = model.kvSizeGB(100000, 1);
  assert.ok(Math.abs(model.kvSizeGB(200000, 1) - 2 * a) < 1e-9);
  assert.ok(Math.abs(model.kvSizeGB(100000, 4) - a / 4) < 1e-9);
});

test('prefill is superlinear in tokens', () => {
  assert.ok(model.prefillSeconds(200000) > 2 * model.prefillSeconds(100000));
});

test('zero retention costs exactly P x prefill cost', () => {
  const C = model.prefillCostUSD(base.tokens, base.gpuUsdPerHour);
  const e = model.expectedCost(0, base.pReturn, base.tauDays, 123, C);
  assert.ok(Math.abs(e.total - base.pReturn * C) < 1e-12);
  assert.equal(e.storage, 0);
});

test('long retention with free storage drives re-prefill cost to ~0', () => {
  const e = model.expectedCost(1000, 0.5, 1, 0, 1);
  assert.ok(e.recompute < 1e-12);
});

test('best point is never worse than the zero-retention baseline', () => {
  const r = model.evaluate(base);
  assert.ok(r.best.total <= r.baselineTotal + 1e-9);
});

test('default scenario has an interior sweet spot', () => {
  const r = model.evaluate(base);
  assert.ok(r.retentionPays);
  assert.ok(!r.hitsMaxWindow);
});

test('expensive RAM storage means do not retain', () => {
  const r = model.evaluate({ ...base, tierId: 'ram', compression: 1 });
  assert.equal(r.best.day, 0);
});

test('optimum matches brute-force search over a fine grid', () => {
  const r = model.evaluate(base);
  const C = model.prefillCostUSD(base.tokens, base.gpuUsdPerHour);
  const perDay = model.storageUsdPerDay(model.kvSizeGB(base.tokens, base.compression), CONFIG.tiers[base.tierId]);
  let bestT = 0, bestV = Infinity;
  for (let T = 0; T <= 30; T += 0.01) {
    const v = model.expectedCost(T, base.pReturn, base.tauDays, perDay, C).total;
    if (v < bestV) { bestV = v; bestT = T; }
  }
  assert.ok(Math.abs(bestT - r.best.day) <= 0.5);
});

test('closed-form optimum x* = c(1-P) / (P(C/tau - c)) matches the grid search', () => {
  const C = model.prefillCostUSD(base.tokens, base.gpuUsdPerHour);
  const c = model.storageUsdPerDay(model.kvSizeGB(base.tokens, base.compression), CONFIG.tiers[base.tierId]);
  const P = base.pReturn, tau = base.tauDays;
  assert.ok(c < (P * C) / tau, 'retention-pays condition should hold for the default scenario');
  const x = (c * (1 - P)) / (P * (C / tau - c));
  const tStar = -tau * Math.log(x);
  const r = model.evaluate(base);
  assert.ok(Math.abs(tStar - r.best.day) <= CONFIG.curve.stepDays);
});
