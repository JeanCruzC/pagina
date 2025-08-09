// Tema Bootstrap usando data-bs-theme en <html>
(function () {
  const root = document.documentElement;
  const btn  = document.getElementById('themeToggle');
  const saved = localStorage.getItem('theme') || 'light';

  function apply(theme){
    root.setAttribute('data-bs-theme', theme);
    localStorage.setItem('theme', theme);
    // icono
    if(btn){
      btn.innerHTML = theme === 'dark'
        ? '<i class="bi bi-sun" aria-hidden="true"></i>'
        : '<i class="bi bi-moon" aria-hidden="true"></i>';
    }
  }
  apply(saved);

  if(btn){
    btn.addEventListener('click', () => {
      const next = (root.getAttribute('data-bs-theme') === 'dark') ? 'light' : 'dark';
      apply(next);
    });
  }
})();

document.addEventListener('DOMContentLoaded', () => {
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

