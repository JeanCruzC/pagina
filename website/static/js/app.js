document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('generatorForm');
  const progress = document.getElementById('progressContainer');
  const toastEl = document.getElementById('timeoutToast');
  const themeToggle = document.getElementById('themeToggle');
  const htmlEl = document.documentElement;
  const TIMEOUT_MS = 30000;
  let toast;

  if (toastEl) {
    toast = new bootstrap.Toast(toastEl);
  }

  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const current = htmlEl.getAttribute('data-bs-theme') || 'light';
      const next = current === 'light' ? 'dark' : 'light';
      htmlEl.setAttribute('data-bs-theme', next);
      const icon = themeToggle.querySelector('i');
      if (icon) {
        icon.classList.toggle('bi-moon', next === 'light');
        icon.classList.toggle('bi-sun', next === 'dark');
      }
    });
  }

  if (form) {
    form.addEventListener('submit', () => {
      const btn = document.getElementById('generateBtn');
      const spinner = btn.querySelector('.spinner-border');
      const btnText = btn.querySelector('.btn-text');
      const btnIcon = btn.querySelector('.bi');

      btn.disabled = true;
      btnText.classList.add('d-none');
      btnIcon.classList.add('d-none');
      spinner.classList.remove('d-none');

      if (progress) {
        progress.classList.remove('d-none');
      }

      if (toast) {
        setTimeout(() => toast.show(), TIMEOUT_MS);
      }
    });
  }
});
