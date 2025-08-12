document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('genForm');
  const spinner = document.getElementById('spinner');
  const slowMsg = document.getElementById('slowMsg');
  const btnGenerar = document.getElementById('btnGenerar');

  if (!form || !spinner || !slowMsg || !btnGenerar) return;

  form.addEventListener('submit', (e) => {
    if (!form.checkValidity()) {
      e.preventDefault();
      e.stopPropagation();
      form.classList.add('was-validated');
      return;
    }
    btnGenerar.disabled = true;
    spinner.classList.remove('d-none');
    setTimeout(() => slowMsg.classList.remove('d-none'), 10000);
  });
});
