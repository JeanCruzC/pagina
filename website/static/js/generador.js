document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('genForm');
  const spinner = document.getElementById('spinner');
  const slowMsg = document.getElementById('slowMsg');
  const btn = document.getElementById('btnSubmit');

  if (!form || !spinner || !slowMsg || !btn) return;

  form.addEventListener('submit', (e) => {
    if (!form.checkValidity()) {
      e.preventDefault();
      e.stopPropagation();
      form.classList.add('was-validated');
      return;
    }
    btn.disabled = true;
    spinner.classList.remove('d-none');
    setTimeout(() => slowMsg.classList.remove('d-none'), 10000);
  });
});
