document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('genForm');
  const spinner = document.getElementById('spinner');
  const slowMsg = document.getElementById('slowMsg');
  const btnExcel = document.getElementById('btnExcel');
  const btnCharts = document.getElementById('btnCharts');
  let generateCharts = false;

  if (!form || !spinner || !slowMsg || !btnExcel || !btnCharts) return;

  form.addEventListener('submit', (e) => {
    generateCharts = e.submitter === btnCharts;
    if (!form.checkValidity()) {
      e.preventDefault();
      e.stopPropagation();
      form.classList.add('was-validated');
      return;
    }
    btnExcel.disabled = true;
    btnCharts.disabled = true;
    spinner.classList.remove('d-none');
    setTimeout(() => slowMsg.classList.remove('d-none'), 10000);
  });

  form.addEventListener('formdata', (e) => {
    e.formData.append('generate_charts', generateCharts ? 'true' : 'false');
  });
});
