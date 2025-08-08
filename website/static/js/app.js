document.addEventListener('DOMContentLoaded', () => {
  const htmlEl = document.documentElement;
  const themeToggle = document.getElementById('themeToggle');
  const storedTheme = localStorage.getItem('theme') || 'light';
  htmlEl.setAttribute('data-bs-theme', storedTheme);

  if (themeToggle) {
    const icon = themeToggle.querySelector('i');
    if (icon) {
      icon.classList.toggle('bi-moon', storedTheme === 'light');
      icon.classList.toggle('bi-sun', storedTheme === 'dark');
    }
    themeToggle.addEventListener('click', () => {
      const current = htmlEl.getAttribute('data-bs-theme') === 'dark' ? 'dark' : 'light';
      const next = current === 'light' ? 'dark' : 'light';
      htmlEl.setAttribute('data-bs-theme', next);
      if (icon) {
        icon.classList.toggle('bi-moon', next === 'light');
        icon.classList.toggle('bi-sun', next === 'dark');
      }
      localStorage.setItem('theme', next);
    });
  }

  const form = document.getElementById('genForm');
  if (form) {
    form.addEventListener('submit', () => {
      const btn = document.getElementById('btnGenerar');
      const spinner = document.getElementById('spinner');
      const progress = document.getElementById('progressContainer');
      if (btn && spinner) {
        btn.disabled = true;
        spinner.classList.remove('d-none');
      }
      if (progress) {
        progress.classList.remove('d-none');
      }
      setTimeout(() => {
        const toastEl = document.getElementById('warningToast');
        if (toastEl) {
          const toast = new bootstrap.Toast(toastEl);
          toast.show();
        }
      }, 240000);
    });
  }
});

