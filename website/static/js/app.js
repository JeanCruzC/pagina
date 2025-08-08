document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('generatorForm');
  const progress = document.getElementById('progressContainer');
  const toastEl = document.getElementById('timeoutToast');
  const themeToggle = document.getElementById('themeToggle');
  const htmlEl = document.documentElement;
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
      const btn = form.querySelector('button[type="submit"]');
      const spinner = btn ? btn.querySelector('.spinner-border') : null;
      const btnText = btn ? btn.querySelector('.btn-text') : null;
      const btnIcon = btn ? btn.querySelector('.bi') : null;

      if (btn) {
        btn.disabled = true;
      }
      if (btnText) {
        btnText.classList.add('d-none');
      }
      if (btnIcon) {
        btnIcon.classList.add('d-none');
      }
      if (spinner) {
        spinner.classList.remove('d-none');
      }

      if (progress) {
        progress.classList.remove('d-none');
      }

      if (toast) {
        setTimeout(() => toast.show(), 30000);
      }
    });
  }
});
