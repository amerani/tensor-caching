const test = require('node:test');
const assert = require('node:assert');
const CONFIG = require('../js/config.js');
const M = require('../js/model.js');

const near = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b)), `${a} != ${b}`);
const cheap = { id: 't', pricePerGBMonth: 0.02, bandwidthGBps: 1.5, latencySeconds: 0.2 };
const shared = { tokens: 40000, sharedTokens: 30000, sharers: 1000, residency: 0.95, tier: cheap };

/* ---- single-conversation properties (original suite) ---- */

test('KV bytes per token is 327,680', () => assert.strictEqual(M.kvBytesPerToken(), 327680));

test('KV size scales linearly with tokens and inversely with compression', () => {
  const a = M.effective({ tokens: 10000, compression: 1 }).totalGB;
  near(M.effective({ tokens: 20000, compression: 1 }).totalGB, 2 * a);
  near(M.effective({ tokens: 10000, compression: 4 }).totalGB, a / 4);
});

test('prefill time is superlinear in tokens', () => {
  assert.ok(M.prefillSeconds(200000) > 2 * M.prefillSeconds(100000));
});

test('total(0) = P x C_eff and optimum is never worse', () => {
  for (const inp of [{}, shared, { tokens: 8000, pReturn: 0.9 }]) {
    const r = M.optimize(inp), x = r.eff.inputs;
    near(r.baseline.total, x.pReturn * r.eff.CEff);
    assert.ok(r.best.total <= r.baseline.total + 1e-15);
  }
});

test('grid optimum matches 0.01-day brute force and closed form', () => {
  const inp = { tokens: 128000, pReturn: 0.8, tauDays: 3, tier: cheap };
  const r = M.optimize(inp), bf = M.bruteForceT(inp, 0.01);
  assert.ok(r.closedFormT > 0 && r.closedFormT < CONFIG.windowDays, 'test needs an interior optimum');
  assert.ok(Math.abs(r.best.T - bf.T) <= CONFIG.stepDays / 2 + 1e-9);
  assert.ok(Math.abs(bf.T - r.closedFormT) <= 0.02);
});

/* ---- block model ---- */

test('reduction: p = 0 equals the single-conversation model', () => {
  const e = M.effective({ tokens: 64000, sharedTokens: 0, sharers: 500, residency: 0.9 });
  near(e.effGB, e.totalGB);
  near(e.CEff, e.CSingle);
  near(e.cEff, e.cSingle);
});

test('reduction: m = 1 and h = 0 equals the single-conversation model', () => {
  const e = M.effective({ tokens: 40000, sharedTokens: 30000, sharers: 1, residency: 0 });
  near(e.effGB, e.totalGB);
  near(e.CEff, e.CSingle);
});

test('conservation: average share over all holders recovers physical bytes', () => {
  const ns = 37;
  const e = M.effective({ tokens: 40000, sharedTokens: 30000, sharers: ns });
  near(ns * e.marginalShare * e.sharedGB, e.sharedGB);
});

test('prefill additivity and C_eff bounds', () => {
  const lo = M.effective({ ...shared, residency: 1 }), hi = M.effective({ ...shared, residency: 0 });
  near(lo.CEff, lo.CSingle - lo.Cprefix);
  near(hi.CEff, hi.CSingle);
  const mid = M.effective(shared);
  assert.ok(mid.CEff <= hi.CEff && mid.CEff >= lo.CEff);
});

test('monotonicity: c_eff falls with sharers, C_eff falls with residency', () => {
  let prev = Infinity;
  for (const ns of [1, 2, 10, 100, 10000]) {
    const c = M.effective({ ...shared, sharers: ns }).cEff;
    assert.ok(c <= prev); prev = c;
  }
  prev = Infinity;
  for (const h of [0, 0.25, 0.5, 0.75, 1]) {
    const C = M.effective({ ...shared, residency: h }).CEff;
    assert.ok(C <= prev); prev = C;
  }
});

test('rounding: shared prefix is a multiple of the block size, rest is unique', () => {
  const s = M.splitTokens(1000, 517, 16);
  assert.strictEqual(s.shared % 16, 0);
  assert.strictEqual(s.shared, 512);
  assert.strictEqual(s.shared + s.unique, 1000);
  assert.strictEqual(M.splitTokens(100, 5000, 16).shared, 96); // clamped to n, then rounded
});

test('marginal share is <= average share and equals 1 for a lone holder', () => {
  const avg = M.marginalShare({ shareMode: 'average', sharers: 10 });
  const mar = M.marginalShare({ shareMode: 'marginal', sharers: 10, holdProb: 0.5 });
  assert.ok(mar <= avg);
  near(M.marginalShare({ shareMode: 'marginal', sharers: 1, holdProb: 0.9 }), 1);
});

test('persist rule agrees with the optimizer', () => {
  for (const price of [0.01, 0.1, 1, 10, 100]) {
    const r = M.optimize({ ...shared, tier: { ...cheap, pricePerGBMonth: price } });
    assert.strictEqual(r.retain, r.best.T > 0 || r.closedFormT > 0);
    if (!r.retain) assert.strictEqual(r.best.T, 0);
  }
});

test('worked example from docs/methodology.md', () => {
  const e = M.effective(shared);
  near(e.sharedGB, 2.4576, 1e-6);
  near(e.uniqueGB, 0.8192, 1e-6);
  near(e.effGB, 0.8217, 1e-3);
  near(e.CSingle, 0.01573, 1e-3);
  near(e.CEff, 0.00499, 2e-3);
  near(e.cEff, 0.000548, 2e-3);
  const r = M.optimize(shared);
  near(r.eff.cEff / (r.eff.inputs.pReturn * r.eff.CEff / r.eff.inputs.tauDays), 0.55, 1e-2);
});

test('shared pool: hit probability and keep rule', () => {
  near(M.hitProbability(0, 5), 0);
  assert.ok(M.hitProbability(100, 1) > 0.999);
  assert.strictEqual(M.poolRetention({ ...shared, requestsPerDay: 1e5 }).keep, true);
  assert.strictEqual(M.poolRetention({ ...shared, requestsPerDay: 0 }).keep, false);
});

test('load time counts only non-resident shared bytes', () => {
  const cold = M.effective({ ...shared, residency: 0 }).tLoad;
  const warm = M.effective({ ...shared, residency: 1 }).tLoad;
  assert.ok(warm < cold);
});
