document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('genForm');
  const spinner = document.getElementById('spinner');
  const slowMsg = document.getElementById('slowMsg');
  const btnExcel = document.getElementById('btnExcel');
  const btnCharts = document.getElementById('btnCharts');
  const progressBar = document.getElementById('progressBar');
  const bar = progressBar ? progressBar.querySelector('.progress-bar') : null;
  let generateCharts = false;
  let intervalId = null;
  let jobId = null;

  if (!form || !spinner || !slowMsg || !btnExcel || !btnCharts || !progressBar || !bar) return;

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
    progressBar.classList.remove('d-none');
    bar.style.width = '0%';

    jobId = self.crypto?.randomUUID ? self.crypto.randomUUID() : Date.now().toString();
    const hidden = document.createElement('input');
    hidden.type = 'hidden';
    hidden.name = 'job_id';
    hidden.value = jobId;
    form.appendChild(hidden);

    intervalId = setInterval(() => {
      fetch(`/progress/${jobId}`)
        .then((r) => r.json())
        .then((data) => {
          const percent = data.percent || 0;
          bar.style.width = `${percent}%`;
          bar.setAttribute('aria-valuenow', percent);
          if (percent >= 100) {
            clearInterval(intervalId);
            spinner.classList.add('d-none');
          }
        })
        .catch(() => {});
    }, 1000);
    setTimeout(() => slowMsg.classList.remove('d-none'), 10000);
  });

  form.addEventListener('formdata', (e) => {
    e.formData.append('generate_charts', generateCharts ? 'true' : 'false');
  });
});
