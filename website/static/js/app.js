/* global bootstrap */
function initApp() {
  // Initialize Bootstrap toasts
  document.querySelectorAll('.toast').forEach(el => bootstrap.Toast.getOrCreateInstance(el));

  // Theme toggle between light and dark
  const toggle = document.getElementById('themeToggle');
  if (toggle) {
    toggle.addEventListener('click', () => {
      const root = document.documentElement;
      const current = root.getAttribute('data-bs-theme') === 'dark' ? 'dark' : 'light';
      root.setAttribute('data-bs-theme', current === 'light' ? 'dark' : 'light');
    });
  }

  // Handle submit button spinner and timeout toast
  const form = document.getElementById('genForm');
  const btn = form ? form.querySelector('button[type="submit"]') : null;
  let timeoutId = null;

  if (btn && !btn.querySelector('.btn-text')) {
    const textSpan = document.createElement('span');
    textSpan.className = 'btn-text';
    textSpan.textContent = btn.textContent.trim();
    const spinner = document.createElement('span');
    spinner.className = 'spinner-border spinner-border-sm d-none';
    spinner.setAttribute('role', 'status');
    spinner.setAttribute('aria-hidden', 'true');
    btn.textContent = '';
    btn.append(textSpan, spinner);
  }

  document.addEventListener('generator:loading', e => {
    if (!btn) return;
    const textSpan = btn.querySelector('.btn-text');
    const spinner = btn.querySelector('.spinner-border');

    if (e.detail) {
      if (textSpan) textSpan.classList.add('d-none');
      if (spinner) spinner.classList.remove('d-none');
      btn.disabled = true;

      const toastEl = document.getElementById('timeoutToast');
      if (toastEl) {
        const delay = parseInt(toastEl.dataset.timeout || '8000', 10);
        timeoutId = setTimeout(() => {
          const toast = bootstrap.Toast.getOrCreateInstance(toastEl);
          toast.show();
        }, delay);
      }
    } else {
      if (textSpan) textSpan.classList.remove('d-none');
      if (spinner) spinner.classList.add('d-none');
      btn.disabled = false;
      if (timeoutId) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
    }
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
