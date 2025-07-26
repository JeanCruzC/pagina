const profile = document.getElementById('profile');
const personalizado = document.getElementById('personalizado');
const jean = document.getElementById('jean');

function updateVisibility() {
  const val = profile.value;
  personalizado.style.display = val === 'Personalizado' ? 'block' : 'none';
  jean.style.display = val === 'JEAN Personalizado' ? 'block' : 'none';
}
profile.addEventListener('change', updateVisibility);
updateVisibility();

function bindRange(id, target) {
  const el = document.getElementById(id);
  const span = document.getElementById(target);
  if (el && span) {
    span.textContent = el.value;
    el.addEventListener('input', () => span.textContent = el.value);
  }
}

bindRange('iterations', 'it_val');
bindRange('solver_time', 'time_val');
bindRange('coverage', 'cov_val');
bindRange('break_from_start', 'bstart_val');
bindRange('break_from_end', 'bend_val');

const form = document.getElementById('genForm');
const results = document.getElementById('results');
const agents = document.getElementById('agents');
const coverage = document.getElementById('coverage');
const demandImg = document.getElementById('demand_img');
const resultImg = document.getElementById('result_img');
const downloadLink = document.getElementById('download_link');
const excelLink = document.getElementById('excel_link');

if (form) {
  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const data = new FormData(form);
    const res = await fetch(form.action, {method: 'POST', body: data});
    const json = await res.json();
    if (json.metrics) {
      agents.textContent = `Agentes estimados: ${json.metrics.total_agents}`;
      coverage.textContent = `Cobertura: ${json.metrics.coverage_percentage.toFixed(1)}%`;
    }
    demandImg.src = 'data:image/png;base64,' + json.demand_image;
    resultImg.src = 'data:image/png;base64,' + json.coverage_image;
    excelLink.href = json.excel_url;
    downloadLink.style.display = 'block';
    results.style.display = 'block';
  });
}

