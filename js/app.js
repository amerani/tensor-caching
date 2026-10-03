(function () {
  'use strict';

  var CONFIG = window.TC.CONFIG;
  var model = window.TC.model;
  var $ = function (id) { return document.getElementById(id); };

  var css = getComputedStyle(document.documentElement);
  var color = function (name) { return css.getPropertyValue(name).trim(); };

  function formatTokens(k) { return k >= 1000 ? (k / 1000) + 'M' : k + 'k'; }

  function formatCents(usd) {
    var c = usd * 100;
    var digits = c < 0.1 ? 3 : c < 10 ? 2 : 1;
    return c.toFixed(digits) + '\u00A2';
  }

  function formatSeconds(s) {
    if (s < 1) return Math.round(s * 1000) + ' ms';
    if (s < 60) return s.toFixed(1) + ' s';
    return (s / 60).toFixed(1) + ' min';
  }

  function formatMoney(v) { return '$' + (v < 10 ? v.toFixed(2) : Math.round(v)); }

  function fillSelect(el, options, selectedValue) {
    options.forEach(function (o) {
      var opt = document.createElement('option');
      opt.value = o.value;
      opt.textContent = o.label;
      if (String(o.value) === String(selectedValue)) opt.selected = true;
      el.appendChild(opt);
    });
  }

  // --- initialise controls from config -------------------------------------------------
  var d = CONFIG.defaults;
  $('tok').value = d.tokensK;
  $('p').value = d.pReturnPct;
  $('t').value = d.tauDays;
  fillSelect($('tier'), Object.keys(CONFIG.tiers).map(function (id) {
    return { value: id, label: CONFIG.tiers[id].label };
  }), d.tierId);
  fillSelect($('comp'), CONFIG.compression.map(function (c) {
    return { value: c.factor, label: c.label };
  }), d.compression);
  fillSelect($('gpu'), CONFIG.gpuPricesPerHour.map(function (g) {
    return { value: g.id, label: g.label };
  }), d.gpuPriceId);

  // --- chart ---------------------------------------------------------------------------
  var chart = new Chart($('ch'), {
    type: 'line',
    data: {
      datasets: [
        { label: 'Total', data: [], borderColor: color('--total'), borderWidth: 2, pointRadius: 0, tension: 0.2, order: 2 },
        { label: 'Storage', data: [], borderColor: color('--storage'), borderWidth: 2, borderDash: [6, 4], pointRadius: 0, tension: 0.2, order: 3 },
        { label: 'Re-prefill', data: [], borderColor: color('--recompute'), borderWidth: 2, borderDash: [2, 3], pointRadius: 0, tension: 0.2, order: 3 },
        { label: 'Best', type: 'scatter', data: [], backgroundColor: color('--total'), borderColor: color('--bg'), borderWidth: 2, pointRadius: 6, order: 1 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: function (items) { return 'Retain for ' + items[0].parsed.x + ' days'; },
            label: function (item) { return item.dataset.label + ': $' + item.parsed.y.toFixed(2); }
          }
        }
      },
      scales: {
        x: {
          type: 'linear', min: 0, max: CONFIG.curve.maxRetentionDays,
          title: { display: true, text: 'Days KV is kept before deleting', color: color('--muted') },
          ticks: { color: color('--muted'), stepSize: 5 },
          grid: { display: false }
        },
        y: {
          min: 0,
          title: { display: true, text: '$ per ' + CONFIG.conversationsScale.toLocaleString() + ' archived conversations', color: color('--muted') },
          ticks: { color: color('--muted'), callback: function (v) { return '$' + v; } },
          grid: { color: color('--grid') }
        }
      }
    }
  });

  // --- render --------------------------------------------------------------------------
  function readParams() {
    var gpu = CONFIG.gpuPricesPerHour.filter(function (g) { return g.id === $('gpu').value; })[0];
    return {
      tokens: Number($('tok').value) * 1000,
      pReturn: Number($('p').value) / 100,
      tauDays: Number($('t').value),
      tierId: $('tier').value,
      compression: Number($('comp').value),
      gpuUsdPerHour: gpu.usd
    };
  }

  function render() {
    var params = readParams();
    var r = model.evaluate(params);

    $('o-tok').textContent = formatTokens(Number($('tok').value)) + ' tokens';
    $('o-p').textContent = Math.round(params.pReturn * 100) + '%';
    $('o-t').textContent = params.tauDays + (params.tauDays === 1 ? ' day' : ' days');

    var toXY = function (key) {
      return r.points.map(function (p) { return { x: p.day, y: p[key] }; });
    };
    chart.data.datasets[0].data = toXY('total');
    chart.data.datasets[1].data = toXY('storage');
    chart.data.datasets[2].data = toXY('recompute');
    chart.data.datasets[3].data = [{ x: r.best.day, y: r.best.total }];
    chart.update();

    $('m-c').textContent = formatCents(r.prefillCostUSD);
    $('m-cs').textContent = formatSeconds(r.prefillSeconds) + ' on one ' + CONFIG.prefill.gpusPerNode + '-GPU node';
    $('m-s').textContent = formatCents(r.storageUsdPerDay) + '/day';
    $('m-ss').textContent = r.sizeGB.toFixed(1) + ' GB per conversation';
    $('m-w').textContent = formatSeconds(r.loadSeconds);
    $('m-ws').textContent = 'vs ' + formatSeconds(r.prefillSeconds) + ' to re-prefill';

    var verdict;
    if (!r.retentionPays) {
      $('m-b').textContent = 'Don\u2019t keep it';
      $('m-bs').textContent = 're-prefill on return';
      verdict = 'At these settings storage costs more than recomputing, even for the first day. Delete the KV on archive and re-prefill if the user comes back.';
    } else {
      var days = r.hitsMaxWindow ? CONFIG.curve.maxRetentionDays + '+' : String(r.best.day);
      $('m-b').textContent = days + ' days';
      $('m-bs').textContent = 'saves ' + r.savingsPct.toFixed(0) + '% vs always re-prefill';
      verdict = 'Keeping KV for about ' + days + ' days minimizes expected cost (' +
        formatMoney(r.best.total) + ' vs ' + formatMoney(r.baselineTotal) +
        ' per ' + CONFIG.conversationsScale.toLocaleString() + ' conversations if you always re-prefill). ' +
        'After that window, delete it and fall back to re-prefill.';
    }
    if (!r.loadBeatsPrefill) {
      verdict += ' Loading from this tier is also no faster than re-prefilling, so retention buys no latency here.';
    } else if (r.retentionPays) {
      verdict += ' Returning users also wait ' + formatSeconds(r.loadSeconds) + ' instead of ' + formatSeconds(r.prefillSeconds) + '.';
    }
    $('verdict').textContent = verdict;
  }

  ['tok', 'p', 't', 'tier', 'comp', 'gpu'].forEach(function (id) {
    $(id).addEventListener('input', render);
  });
  render();
})();
