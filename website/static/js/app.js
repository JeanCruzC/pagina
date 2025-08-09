document.addEventListener('DOMContentLoaded', () => {
  const htmlEl = document.documentElement;
  const body = document.body;
  const themeToggle = document.getElementById('themeToggle');
  const storedTheme = localStorage.getItem('theme');

  const applyTheme = (theme) => {
    const isDark = theme === 'dark';
    htmlEl.setAttribute('data-bs-theme', isDark ? 'dark' : 'light');
    body.classList.toggle('dark-mode', isDark);
    return isDark;
  };

  const initIsDark = applyTheme(storedTheme || 'light');

  if (themeToggle) {
    const icon = themeToggle.querySelector('i');
    const updateIcon = (isDark) => {
      if (!icon) return;
      icon.classList.toggle('bi-moon', !isDark);
      icon.classList.toggle('bi-sun', isDark);
    };
    updateIcon(initIsDark);
    themeToggle.addEventListener('click', () => {
      const isDark = body.classList.toggle('dark-mode');
      htmlEl.setAttribute('data-bs-theme', isDark ? 'dark' : 'light');
      updateIcon(isDark);
      localStorage.setItem('theme', isDark ? 'dark' : 'light');
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

