/* Helper to update the text next to a slider. */
function updateSliderValue(sliderId, labelId) {
  const slider = document.getElementById(sliderId);
  const label = document.getElementById(labelId);
  if (!slider || !label) return;
  label.textContent = slider.value;
  slider.addEventListener('input', () => {
    label.textContent = slider.value;
  });
}

/* Show/hide FT related checkboxes. */
function toggleFTOptions() {
  const ft = document.getElementById('ft');
  const box = document.getElementById('ft_opts');
  if (ft && box) box.style.display = ft.checked ? 'block' : 'none';
}

/* Show/hide PT related checkboxes. */
function togglePTOptions() {
  const pt = document.getElementById('pt');
  const box = document.getElementById('pt_opts');
  if (pt && box) box.style.display = pt.checked ? 'block' : 'none';
}

/* Toggle custom parameter sections based on profile selector. */
function toggleCustomParams() {
  const profile = document.getElementById('profile');
  const custom = document.getElementById('personalizado');
  const jean = document.getElementById('jean');
  if (!profile || !custom || !jean) return;
  const val = profile.value;
  custom.style.display = val === 'Personalizado' ? 'block' : 'none';
  jean.style.display = val === 'JEAN Personalizado' ? 'block' : 'none';
}

/* Format a 7x24 matrix as an HTML table. */
function formatDemandAnalysis(matrix) {
  if (!Array.isArray(matrix)) return '';
  const table = document.createElement('table');
  table.className = 'table table-sm table-bordered';
  matrix.forEach(row => {
    const tr = document.createElement('tr');
    row.forEach(cell => {
      const td = document.createElement('td');
      td.textContent = cell;
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  return table.outerHTML;
}

/* Render the results section using the JSON returned by the server. */
function displayResults(data) {
  let container = document.getElementById('results');
  if (!container) {
    container = document.createElement('div');
    container.id = 'results';
    document.getElementById('genForm').after(container);
  }
  container.innerHTML = '';

  if (data.metrics) {
    const m = data.metrics;
    container.insertAdjacentHTML('beforeend',
      `<h4>Resultados</h4>
       <p>Agentes estimados: ${m.total_agents}</p>
       <p>Cobertura: ${m.coverage_percentage.toFixed(1)}%</p>`);
  }

  if (data.demand_url) {
    const img = document.createElement('img');
    img.src = data.demand_url;
    img.className = 'img-fluid';
    container.appendChild(img);
  }

  if (data.image_url) {
    const img = document.createElement('img');
    img.src = data.image_url;
    img.className = 'img-fluid';
    container.appendChild(img);
  }

  if (data.diff_matrix) {
    container.insertAdjacentHTML('beforeend', formatDemandAnalysis(data.diff_matrix));
  }

  if (data.download_url) {
    container.insertAdjacentHTML('beforeend',
      `<p class="mt-2"><a class="btn btn-success" href="${data.download_url}">Descargar Excel</a></p>`);
  }
}

/* Event bindings */
updateSliderValue('iterations', 'it_val');
updateSliderValue('solver_time', 'time_val');
updateSliderValue('coverage', 'cov_val');
updateSliderValue('break_from_start', 'bstart_val');
updateSliderValue('break_from_end', 'bend_val');
toggleFTOptions();
togglePTOptions();
toggleCustomParams();

document.getElementById('profile').addEventListener('change', toggleCustomParams);
document.getElementById('ft').addEventListener('change', toggleFTOptions);
document.getElementById('pt').addEventListener('change', togglePTOptions);

/* Intercept form submission and send via fetch. */
const form = document.getElementById('genForm');
const progressContainer = document.getElementById('progress-container');
const progressBar = document.getElementById('progressBar');
if (form) {
  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    if (progressContainer && progressBar) {
      progressContainer.style.display = 'block';
      progressBar.style.width = '100%';
    }
    const data = new FormData(form);
    try {
      const res = await fetch('/generador', { method: 'POST', body: data });
      const json = await res.json();
      displayResults(json);
    } catch (err) {
      console.error(err);
    }
    if (progressContainer && progressBar) {
      progressBar.style.width = '0%';
      progressContainer.style.display = 'none';
    }
  });
}

