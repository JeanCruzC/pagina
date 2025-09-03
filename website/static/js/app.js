// Tema Bootstrap usando data-bs-theme en <html>
(function () {
  const root = document.documentElement;
  const btn  = document.getElementById('themeToggle');
  const saved = localStorage.getItem('theme') || 'light';

  function apply(theme){
    root.setAttribute('data-bs-theme', theme);
    localStorage.setItem('theme', theme);
    // icono
    if(btn){
      btn.innerHTML = theme === 'dark'
        ? '<i class="bi bi-sun" aria-hidden="true"></i>'
        : '<i class="bi bi-moon" aria-hidden="true"></i>';
    }
  }
  apply(saved);

  if(btn){
    btn.addEventListener('click', () => {
      const next = (root.getAttribute('data-bs-theme') === 'dark') ? 'light' : 'dark';
      apply(next);
    });
  }
})();

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('genForm');
  if (form) {
    form.addEventListener('submit', () => {
      const btn = document.getElementById('btnGenerar');
      const spinner = document.getElementById('spinner');
      const progress = document.getElementById('progressContainer');
      if (btn && spinner) {
        btn.disabled = true;
        spinner.classList.remove('d-none');
      }
      if (progress) {
        progress.classList.remove('d-none');
      }
      setTimeout(() => {
        const toastEl = document.getElementById('warningToast');
        if (toastEl) {
          const toast = new bootstrap.Toast(toastEl);
          toast.show();
        }
      }, 240000);
    });
  }
});

// ------------------- PuLP Charts -------------------
let _covChart, _defChart, _excChart;

function renderMatrixChart(canvasId, labels, matrix, title) {
  const ctx = document.getElementById(canvasId);
  const days = labels.days || [];
  const hours = labels.hours || [];
  const datasets = (matrix || []).map((row, i) => ({
    label: days[i] ?? `Día ${i+1}`,
    data: row,
  }));
  return new Chart(ctx, {
    type: 'line',
    data: { labels: hours, datasets },
    options: { responsive: true, plugins: { title: { display: true, text: title } } }
  });
}

function updatePulpCharts(pr) {
  if (!pr || !pr.charts) return;

  const { labels, demand, coverage, deficit, excess } = pr.charts;

  if (_covChart) _covChart.destroy();
  if (_defChart) _defChart.destroy();
  if (_excChart) _excChart.destroy();

  _covChart = renderMatrixChart('chartCoverage', labels, coverage, 'Cobertura');
  _defChart = renderMatrixChart('chartDeficit', labels, deficit, 'Déficit');
  _excChart = renderMatrixChart('chartExcess', labels, excess, 'Exceso');
}

function onRefreshPayload(payload) {
  if (!payload) return;
  if (payload.pulp_results) {
    updatePulpCharts(payload.pulp_results);

    const m = payload.pulp_results.metrics;
    if (m && typeof m.coverage_percentage === 'number') {
      const el = document.getElementById('pulp-coverage-pct');
      if (el) el.textContent = `${m.coverage_percentage}%`;
    }
  }
}

