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

/* Render the results section using the JSON returned by the server. */
function displayResults(data) {
  const metrics = document.getElementById('metrics');
  if (data.metrics && metrics) {
    document.getElementById('m_agents').textContent = data.metrics.total_agents;
    document.getElementById('m_cov').textContent = data.metrics.coverage_percentage.toFixed(1) + '%';
    document.getElementById('m_over').textContent = data.metrics.overstaffing;
    document.getElementById('m_under').textContent = data.metrics.understaffing;
    metrics.style.display = 'flex';
  }

  if (data.heatmaps) {
    if (data.heatmaps.demand) {
      document.getElementById('hm-demand').src = 'data:image/png;base64,' + data.heatmaps.demand;
    }
    if (data.heatmaps.coverage) {
      document.getElementById('hm-coverage').src = 'data:image/png;base64,' + data.heatmaps.coverage;
    }
    if (data.heatmaps.difference) {
      document.getElementById('hm-diff').src = 'data:image/png;base64,' + data.heatmaps.difference;
    }
  }

  const dl = document.getElementById('downloadBtn');
  if (dl && data.download_url) {
    dl.href = data.download_url;
    dl.style.display = 'inline-block';
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
const progressContainer = document.getElementById('progress');
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
      const res = await fetch('/generador', {
        method: 'POST',
        body: data,
        headers: { 'Accept': 'application/json' }
      });
      let json;
      try {
        json = await res.json();
      } catch (err) {
        console.error('JSON parse error:', err);
        alert('Error: ' + err);
        return;
      }
      if (!res.ok) {
        console.error('Server error:', json);
        alert(json.error || 'Error');
        return;
      }
      displayResults(json);
    } catch (err) {
      console.error(err);
      alert('Error: ' + err);
    }
    if (progressContainer && progressBar) {
      progressBar.style.width = '0%';
      progressContainer.style.display = 'none';
    }
  });
}

