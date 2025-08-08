document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('generatorForm');
  const progress = document.getElementById('progressContainer');
  const toastEl = document.getElementById('timeoutToast');
  let toast;

  if (toastEl) {
    toast = new bootstrap.Toast(toastEl);
  }

  if (form) {
    form.addEventListener('submit', () => {
      if (progress) {
        progress.classList.remove('d-none');
      }

      if (toast) {
        setTimeout(() => toast.show(), 30000);
      }
    });
  }
});

