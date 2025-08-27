document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('genForm');
  const spinner = document.getElementById('spinner');
  const slowMsg = document.getElementById('slowMsg');
  const btnExcel = document.getElementById('btnExcel');
  const btnCharts = document.getElementById('btnCharts');
  let generateCharts = false;
  let slowTimer;
  let controller;
  let job_id;

  if (!form || !spinner || !slowMsg || !btnExcel || !btnCharts) return;

  form.addEventListener('submit', async (e) => {
    generateCharts = e.submitter === btnCharts;
    if (!form.checkValidity()) {
      e.preventDefault();
      e.stopPropagation();
      form.classList.add('was-validated');
      return;
    }

    e.preventDefault();

    btnExcel.disabled = true;
    btnCharts.disabled = true;
    spinner.classList.remove('d-none');
    slowTimer = setTimeout(() => slowMsg.classList.remove('d-none'), 10000);

    job_id = crypto.randomUUID();

    const data = new FormData(form);
    data.append('generate_charts', generateCharts ? 'true' : 'false');
    data.append('job_id', job_id);
    controller = new AbortController();

    try {
      const resp = await fetch(form.action, {
        method: form.method,
        body: data,
        headers: { 'Accept': 'application/json' },
        signal: controller.signal
      });
      const info = await resp.json();
      job_id = info.job_id || job_id;
      if (!job_id) throw new Error('sin id');
      pollStatus(job_id);
    } catch (err) {
      alert('La generación falló. Por favor inténtalo nuevamente.');
      resetUI();
    }
  });

  window.addEventListener('pagehide', () => {
    if (controller) {
      controller.abort();
      navigator.sendBeacon('/cancel', JSON.stringify({ job_id }));
      resetUI();
    }
  });

  async function pollStatus(job_id) {
    try {
      const resp = await fetch(`/generador/status/${job_id}`);
      const data = await resp.json();
      if (data.status === 'finished') {
        window.location.href = '/resultados';
      }
      if (data.status === 'error') {
        throw new Error('error');
      }
    } catch (err) {
      alert('La generación falló. Por favor inténtalo nuevamente.');
      resetUI();
      return;
    }
    setTimeout(() => pollStatus(job_id), 3000);
  }

  function resetUI() {
    spinner.classList.add('d-none');
    slowMsg.classList.add('d-none');
    btnExcel.disabled = false;
    btnCharts.disabled = false;
    clearTimeout(slowTimer);
    controller = null;
    job_id = null;
  }
});
