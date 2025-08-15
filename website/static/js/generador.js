document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('genForm');
  const spinner = document.getElementById('spinner');
  const slowMsg = document.getElementById('slowMsg');
  const btnExcel = document.getElementById('btnExcel');

  if (!form || !spinner || !slowMsg || !btnExcel) return;

  form.addEventListener('submit', (e) => {
    if (!form.checkValidity()) {
      e.preventDefault();
      e.stopPropagation();
      form.classList.add('was-validated');
      return;
    }
    btnExcel.disabled = true;
    spinner.classList.remove('d-none');
    setTimeout(() => slowMsg.classList.remove('d-none'), 10000);
  });
});
