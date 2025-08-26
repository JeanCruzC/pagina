document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('genForm');
  const spinner = document.getElementById('spinner');
  const slowMsg = document.getElementById('slowMsg');
  const btnExcel = document.getElementById('btnExcel');
  const btnCharts = document.getElementById('btnCharts');
  let generateCharts = false;
  let slowTimer;

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

    const data = new FormData(form);
    data.append('generate_charts', generateCharts ? 'true' : 'false');

    try {
      const resp = await fetch(form.action, {
        method: form.method,
        body: data,
        headers: { 'Accept': 'application/json' }
      });
      const info = await resp.json();
      const jobId = info.job_id;
      if (!jobId) throw new Error('sin id');
      pollStatus(jobId);
    } catch (err) {
      alert('La generación falló. Por favor inténtalo nuevamente.');
      spinner.classList.add('d-none');
      slowMsg.classList.add('d-none');
      btnExcel.disabled = false;
      btnCharts.disabled = false;
      clearTimeout(slowTimer);
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
      spinner.classList.add('d-none');
      slowMsg.classList.add('d-none');
      btnExcel.disabled = false;
      btnCharts.disabled = false;
      clearTimeout(slowTimer);
      return;
    }
    setTimeout(() => pollStatus(jobId), 3000);
  }
});
