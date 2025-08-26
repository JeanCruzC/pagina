document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('genForm');
  const spinner = document.getElementById('spinner');
  const slowMsg = document.getElementById('slowMsg');
  const btnExcel = document.getElementById('btnExcel');
  const btnCharts = document.getElementById('btnCharts');
  const btnCancel = document.getElementById('btnCancel');
  let generateCharts = false;
  let slowTimer;
  let controller;

  if (!form || !spinner || !slowMsg || !btnExcel || !btnCharts || !btnCancel) return;

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
    btnCancel.classList.remove('d-none');
    btnCancel.disabled = false;
    spinner.classList.remove('d-none');
    slowTimer = setTimeout(() => slowMsg.classList.remove('d-none'), 10000);

    const data = new FormData(form);
    data.append('generate_charts', generateCharts ? 'true' : 'false');

    controller = new AbortController();
    btnCancel.addEventListener('click', () => {
      controller.abort();
    }, { once: true });

    try {
      const resp = await fetch(form.action, {
        method: form.method,
        body: data,
        signal: controller.signal,
        headers: { 'Accept': 'application/json' }
      });
      if (resp.ok) {
        await resp.json();
        window.location.href = '/resultados';
        return;
      }
      throw new Error('fetch-failed');
    } catch (err) {
      if (err.name !== 'AbortError') {
        alert('La generación falló. Por favor inténtalo nuevamente.');
      }
    } finally {
      spinner.classList.add('d-none');
      slowMsg.classList.add('d-none');
      btnExcel.disabled = false;
      btnCharts.disabled = false;
      btnCancel.classList.add('d-none');
      clearTimeout(slowTimer);
    }
  });
});
