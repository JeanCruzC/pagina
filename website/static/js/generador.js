document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('genForm');
  const spinner = document.getElementById('spinner');
  const slowMsg = document.getElementById('slowMsg');
  const btnExcel = document.getElementById('btnExcel');
  const btnCharts = document.getElementById('btnCharts');
  let generateCharts = false;
  let slowTimer;
  let controller;
  let jobId;

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

    jobId = crypto.randomUUID();

    const data = new FormData(form);
    data.append('generate_charts', generateCharts ? 'true' : 'false');
    data.append('job_id', jobId);
    controller = new AbortController();

    try {
      const resp = await fetch(form.action, {
        method: form.method,
        body: data,
        headers: { 'Accept': 'application/json' },
        signal: controller.signal
      });
      const info = await resp.json();
      jobId = info.job_id || jobId;
      if (!jobId) throw new Error('sin id');
      pollStatus(jobId);
    } catch (err) {
      alert('La generación falló. Por favor inténtalo nuevamente.');
      resetUI();
    }
  });

  window.addEventListener('pagehide', () => {
    if (controller) {
      controller.abort();
      navigator.sendBeacon('/cancel', JSON.stringify({ job_id: jobId }));
      resetUI();
    }
  });

  async function pollStatus(jobId) {
    try {
      const resp = await fetch(`/generador/status/${jobId}`);
      const data = await resp.json();
      if (data.status === 'finished') {
        window.location.href = '/resultados';
        return;
      }
      if (data.status === 'error') {
        throw new Error('error');
      }
    } catch (err) {
      alert('La generación falló. Por favor inténtalo nuevamente.');
      resetUI();
      return;
    }
    setTimeout(() => pollStatus(jobId), 3000);
  }

  function resetUI() {
    spinner.classList.add('d-none');
    slowMsg.classList.add('d-none');
    btnExcel.disabled = false;
    btnCharts.disabled = false;
    clearTimeout(slowTimer);
  }
});
