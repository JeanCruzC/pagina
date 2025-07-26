const profile = document.getElementById('profile');
const personalizado = document.getElementById('personalizado');
const jean = document.getElementById('jean');

function updateVisibility() {
  const val = profile.value;
  personalizado.style.display = val === 'Personalizado' ? 'block' : 'none';
  jean.style.display = val === 'JEAN Personalizado' ? 'block' : 'none';
}
profile.addEventListener('change', updateVisibility);
updateVisibility();

function bindRange(id, target) {
  const el = document.getElementById(id);
  const span = document.getElementById(target);
  if (el && span) {
    span.textContent = el.value;
    el.addEventListener('input', () => span.textContent = el.value);
  }
}

bindRange('iterations', 'it_val');
bindRange('solver_time', 'time_val');
bindRange('coverage', 'cov_val');
bindRange('break_from_start', 'bstart_val');
bindRange('break_from_end', 'bend_val');

