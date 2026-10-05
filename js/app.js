/* Widget wiring: reads controls, calls TCModel, renders stats + SVG chart. */
(function () {
  'use strict';
  const C = window.TC_CONFIG, M = window.TCModel;
  const $ = id => document.getElementById(id);
  const D = C.defaults;
  const pct = ['residency', 'pReturn', 'holdProb'];

  const fmt = {
    tokens: v => (v >= 1000 ? Math.round(v / 1000) + 'k' : String(v)),
    sharedTokens: v => (v >= 1000 ? Math.round(v / 1000) + 'k' : String(v)),
    sharers: v => Number(v).toLocaleString('en-US'),
    residency: v => v + '%', pReturn: v => v + '%', holdProb: v => v + '%',
    tauDays: v => v + ' d', gpuPrice: v => '$' + Number(v).toFixed(2)
  };
  const money = v => '$' + (v >= 100 ? v.toFixed(0) : v >= 1 ? v.toFixed(2) : v.toFixed(4));

  // populate selects
  C.tiers.forEach(t => $('tierId').add(new Option(t.label + ' ($' + t.pricePerGBMonth + '/GB-mo)', t.id)));
  C.compressionOptions.forEach(c => $('compression').add(new Option(c + 'x', c)));

  // initial values
  const initial = Object.assign({}, D, { residency: D.residency * 100, pReturn: D.pReturn * 100, holdProb: D.holdProb * 100 });
  Object.keys(initial).forEach(k => { if ($(k)) $(k).value = initial[k]; });

  function read() {
    const x = {};
    ['tokens', 'sharedTokens', 'sharers', 'residency', 'holdProb', 'pReturn', 'tauDays', 'gpuPrice', 'compression'].forEach(k => {
      x[k] = Number($(k).value) / (pct.includes(k) ? 100 : 1);
    });
    x.tierId = $('tierId').value;
    x.shareMode = $('shareMode').value;
    x.blockTokens = D.blockTokens;
    return x;
  }

  function stat(label, value, note) {
    return '<div class="stat"><small>' + label + '</small><b>' + value + '</b><small>' + (note || '') + '</small></div>';
  }

  function drawChart(r) {
    const W = 720, H = 340, L = 64, R = 16, T = 16, B = 40;
    const pts = r.points, k = r.multiplier;
    const maxY = Math.max.apply(null, pts.map(p => Math.max(p.storage, p.recompute, p.total))) * k * 1.08 || 1;
    const sx = d => L + (d / C.windowDays) * (W - L - R);
    const sy = v => H - B - (v / maxY) * (H - T - B);
    const path = key => pts.map((p, i) => (i ? 'L' : 'M') + sx(p.T).toFixed(1) + ' ' + sy(p[key] * k).toFixed(1)).join('');
    let g = '';
    for (let i = 0; i <= 4; i++) {
      const v = (maxY * i) / 4, y = sy(v);
      g += '<line class="axis" x1="' + L + '" x2="' + (W - R) + '" y1="' + y + '" y2="' + y + '"/>' +
           '<text x="' + (L - 6) + '" y="' + (y + 4) + '" text-anchor="end">' + money(v) + '</text>';
    }
    for (let d = 0; d <= C.windowDays; d += 5) {
      g += '<text x="' + sx(d) + '" y="' + (H - B + 16) + '" text-anchor="middle">' + d + '</text>';
    }
    g += '<text x="' + (W / 2) + '" y="' + (H - 6) + '" text-anchor="middle">Retention window T (days)</text>';
    const bx = sx(r.best.T), by = sy(r.best.total * k);
    const marker = '<circle cx="' + bx + '" cy="' + by + '" r="5" fill="var(--accent)"/>' +
      '<text class="best" x="' + (bx - 20) + '" y="' + (by - 20) + '" style="fill:var(--accent)">' +
      (r.best.T === 0 ? 'best: T = 0' : 'best: T = ' + r.best.T + (r.atCap ? '+' : '') + ' d') + '</text>';
    $('chart').innerHTML = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Expected cost versus retention window">' + g +
      '<path d="' + path('storage') + '" fill="none" stroke="var(--storage)" stroke-width="1"/>' +
      '<path d="' + path('recompute') + '" fill="none" stroke="var(--recompute)" stroke-width="1"/>' +
      '<path d="' + path('total') + '" fill="none" stroke="var(--total)" stroke-width="1.5"/>' + marker + '</svg>';
  }

  function render() {
    const inp = read();
    // the shared prefix can't exceed the conversation
    $('sharedTokens').max = String(Math.min(200000, inp.tokens));
    if (inp.sharedTokens > inp.tokens) { inp.sharedTokens = inp.tokens; $('sharedTokens').value = inp.tokens; }
    Object.keys(fmt).forEach(k => { if ($('o-' + k)) $('o-' + k).textContent = fmt[k](Number($(k).value)); });
    $('holdRow').hidden = inp.shareMode !== 'marginal';

    const r = M.optimize(inp), e = r.eff;
    const v = $('verdict');
    v.textContent = M.describe(r);
    v.className = r.retain && r.best.T > 0 ? '' : 'delete';

    const shared = e.split.shared > 0;
    $('stats').innerHTML =
      stat('KV charged to this conversation', e.effGB.toFixed(2) + ' GB', shared ? 'of ' + e.totalGB.toFixed(2) + ' GB total (share m = ' + e.marginalShare.toFixed(4) + ')' : 'no shared prefix') +
      stat('Storage per day', money(e.cEff), shared ? 'vs ' + money(e.cSingle) + ' unshared' : '') +
      stat('Expected re-prefill', money(e.CEff), shared ? 'vs ' + money(e.CSingle) + ' unshared' : '') +
      stat('Break-even storage / day', money((inp.pReturn * e.CEff) / inp.tauDays), 'persist if storage is below this') +
      stat('Closed-form T*', isFinite(r.closedFormT) ? r.closedFormT.toFixed(1) + ' d' : '30+ d', 'grid best: ' + r.best.T + ' d');
    drawChart(r);
  }

  document.querySelectorAll('input, select').forEach(el => el.addEventListener('input', render));
  render();
})();
