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
        body: data
      });
      if (resp.redirected) {
        window.location.href = resp.url;
        return;
      }
      throw new Error('non-redirect');
    } catch (err) {
      alert('La generación falló. Por favor inténtalo nuevamente.');
      spinner.classList.add('d-none');
      slowMsg.classList.add('d-none');
      btnExcel.disabled = false;
      btnCharts.disabled = false;
      clearTimeout(slowTimer);
    }
  });
});
