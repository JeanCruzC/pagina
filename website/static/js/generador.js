document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('genForm');
  const spinner = document.getElementById('spinner');
  const slowMsg = document.getElementById('slowMsg');
  const btnExcel = document.getElementById('btnExcel');
  const btnCharts = document.getElementById('btnCharts');
  const profileSelect = document.getElementById('profile');
  let generateCharts = false;
  let slowTimer;
  let controller;
  let job_id;
  let navigatingToResults = false;

  if (!form || !spinner || !slowMsg || !btnExcel || !btnCharts) return;

  // Configuraciones de perfiles (sincronizadas con PERFILES_OPTIMIZACION.md)
  const profileConfigs = {
    'Equilibrado (Recomendado)': {
      agent_limit_factor: 12, excess_penalty: 2.0, peak_bonus: 1.5, critical_bonus: 2.0,
      solver_time: 300, coverage: 98, iterations: 30
    },
    'Conservador': {
      agent_limit_factor: 30, excess_penalty: 0.5, peak_bonus: 1.0, critical_bonus: 1.2,
      solver_time: 240, coverage: 98, iterations: 25
    },
    'Agresivo': {
      agent_limit_factor: 15, excess_penalty: 0.05, peak_bonus: 1.5, critical_bonus: 2.0,
      solver_time: 180, coverage: 95, iterations: 20
    },
    'Máxima Cobertura': {
      agent_limit_factor: 7, excess_penalty: 0.005, peak_bonus: 3.0, critical_bonus: 4.0,
      solver_time: 400, coverage: 99.5, iterations: 35
    },
    'Mínimo Costo': {
      agent_limit_factor: 35, excess_penalty: 0.8, peak_bonus: 0.8, critical_bonus: 1.0,
      solver_time: 200, coverage: 90, iterations: 20
    },
    '100% Cobertura Eficiente': {
      agent_limit_factor: 6, excess_penalty: 0.01, peak_bonus: 3.5, critical_bonus: 4.5,
      solver_time: 500, coverage: 100, iterations: 40
    },
    '100% Cobertura Total': {
      agent_limit_factor: 5, excess_penalty: 0.001, peak_bonus: 4.0, critical_bonus: 5.0,
      solver_time: 600, coverage: 100, iterations: 45
    },
    'Cobertura Perfecta': {
      agent_limit_factor: 8, excess_penalty: 0.01, peak_bonus: 3.0, critical_bonus: 4.0,
      solver_time: 450, coverage: 99.8, iterations: 35
    },
    '100% Exacto': {
      agent_limit_factor: 6, excess_penalty: 0.005, peak_bonus: 4.0, critical_bonus: 5.0,
      solver_time: 600, coverage: 100, iterations: 50
    },
    'JEAN': {
      agent_limit_factor: 30, excess_penalty: 5.0, peak_bonus: 2.0, critical_bonus: 2.5,
      solver_time: 240, coverage: 98, iterations: 30
    },
    'JEAN Personalizado': {
      agent_limit_factor: 30, excess_penalty: 5.0, peak_bonus: 2.0, critical_bonus: 2.5,
      solver_time: 240, coverage: 98, iterations: 30
    },
    'Personalizado': {
      agent_limit_factor: 25, excess_penalty: 0.5, peak_bonus: 1.5, critical_bonus: 2.0,
      solver_time: 300, coverage: 98, iterations: 30
    },
    'Aprendizaje Adaptativo': {
      agent_limit_factor: 8, excess_penalty: 0.01, peak_bonus: 3.0, critical_bonus: 4.0,
      solver_time: 400, coverage: 99, iterations: 40
    }
  };

  // Aplicar configuración de perfil
  function applyProfileConfig(profileName) {
    const config = profileConfigs[profileName];
    if (!config) {
      console.warn(`Configuración no encontrada para perfil: ${profileName}`);
      return;
    }

    console.log(`Aplicando configuración para perfil: ${profileName}`, config);

    // Actualizar campos del formulario con animación visual
    const fields = {
      'agent_limit_factor': config.agent_limit_factor,
      'excess_penalty': config.excess_penalty,
      'peak_bonus': config.peak_bonus,
      'critical_bonus': config.critical_bonus,
      'solver_time': config.solver_time,
      'coverage': config.coverage,
      'iterations': config.iterations
    };

    Object.entries(fields).forEach(([name, value]) => {
      const input = form.querySelector(`[name="${name}"]`);
      if (input) {
        // Animación visual para mostrar que el campo se actualizó
        input.style.backgroundColor = '#e3f2fd';
        input.value = value;
        
        // Remover el highlight después de un momento
        setTimeout(() => {
          input.style.backgroundColor = '';
        }, 1000);
      } else {
        console.warn(`Campo no encontrado: ${name}`);
      }
    });

    // Mostrar notificación de que se aplicaron los parámetros
    showProfileNotification(profileName);
  }

  // Mostrar notificación de cambio de perfil
  function showProfileNotification(profileName) {
    // Crear notificación temporal
    const notification = document.createElement('div');
    notification.className = 'alert alert-info alert-dismissible fade show';
    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.zIndex = '9999';
    notification.style.maxWidth = '400px';
    notification.innerHTML = `
      <strong>Parámetros actualizados</strong><br>
      Se aplicaron los parámetros del perfil: <em>${profileName}</em>
      <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remover después de 3 segundos
    setTimeout(() => {
      if (notification.parentNode) {
        notification.remove();
      }
    }, 3000);
  }

  // Event listener para cambio de perfil
  if (profileSelect) {
    profileSelect.addEventListener('change', (e) => {
      applyProfileConfig(e.target.value);
    });
    // Aplicar configuración inicial
    applyProfileConfig(profileSelect.value);
  }

  form.addEventListener('submit', async (e) => {
    generateCharts = e.submitter === btnCharts;
    const exportFiles = e.submitter === btnExcel; // solo generar archivos cuando se pide Excel
    if (!form.checkValidity()) {
      e.preventDefault();
      e.stopPropagation();
      form.classList.add('was-validated');
      return;
    }

    e.preventDefault();

    btnExcel.disabled = true;
    btnCharts.disabled = true;
    spinner.classList.remove('d-none');
    slowTimer = setTimeout(() => slowMsg.classList.remove('d-none'), 10000);

    job_id = crypto.randomUUID();

    const data = new FormData(form);
    data.append('generate_charts', generateCharts ? 'true' : 'false');
    data.append('export_files', exportFiles ? 'true' : 'false');
    data.append('job_id', job_id);
    controller = new AbortController();

    try {
      const resp = await fetch(form.action, {
        method: form.method,
        body: data,
        headers: { 'Accept': 'application/json' },
        signal: controller.signal
      });
      const info = await resp.json();
      job_id = info.job_id || job_id;
      if (info.status === 'error' || !job_id) {
        throw new Error(info.error || 'sin id');
      }
      // UX alineada al legacy/Streamlit: navegar a la página de resultados
      // inmediatamente y dejar que ésta haga auto-refresh hasta que haya datos.
      navigatingToResults = true;
      window.location.assign(`/resultados/${job_id}`);
      // No seguimos con el polling aquí: la página de resultados se encargará
      // del auto-refresh. Esto evita 'Failed to fetch' durante la navegación.
    } catch (err) {
      alert(err.message || 'La generación falló. Por favor inténtalo nuevamente.');
      resetUI();
    }
  });

  window.addEventListener('pagehide', () => {
    // No cancelar si estamos navegando deliberadamente a Resultados
    if (controller && !navigatingToResults) {
      controller.abort();
      navigator.sendBeacon(
        '/cancel',
        new Blob([JSON.stringify({ job_id })], { type: 'application/json' })
      );
      resetUI();
    }
  });

  async function pollStatus(job_id) {
    try {
      const resp = await fetch(`/generador/status/${job_id}`);
      const data = await resp.json();
      
      // Mostrar progreso si está disponible
      if (data.progress) {
        updateProgressDisplay(data.progress);
      }
      
      if (data.status === 'finished') {
        const next = data.redirect || `/resultados/${job_id}`;
        window.location.assign(next);
        return;
      }
      if (data.status === 'error') {
        alert(data.error || 'La generación falló. Por favor inténtalo nuevamente.');
        resetUI();
        return;
      }
    } catch (err) {
      alert(err.message || 'La generación falló. Por favor inténtalo nuevamente.');
      resetUI();
      return;
    }
    setTimeout(() => pollStatus(job_id), 3000);
  }

  function updateProgressDisplay(progress) {
    let progressText = 'Procesando';
    const parts = [];
    
    // Construir texto de progreso más detallado
    if (progress.stage) {
      parts.push(progress.stage);
    }
    
    if (progress.pulp_status) {
      parts.push(`PuLP: ${progress.pulp_status}`);
    }
    
    if (progress.greedy_iteration || progress.greedy_status) {
      const greedyInfo = progress.greedy_iteration || progress.greedy_status;
      parts.push(`Greedy: ${greedyInfo}`);
    }
    
    if (progress.jean_iteration) {
      parts.push(`JEAN: ${progress.jean_iteration}`);
    }
    
    if (parts.length > 0) {
      progressText += '... ' + parts.join(' | ');
    } else {
      progressText += '…';
    }
    
    spinner.innerHTML = progressText;
    
    // Agregar indicador visual de actividad
    if (!spinner.querySelector('.spinner-border')) {
      const spinnerIcon = document.createElement('span');
      spinnerIcon.className = 'spinner-border spinner-border-sm me-2';
      spinnerIcon.setAttribute('role', 'status');
      spinner.insertBefore(spinnerIcon, spinner.firstChild);
    }
  }

  function resetUI() {
    spinner.classList.add('d-none');
    spinner.innerHTML = 'Procesando…'; // Reset texto
    
    // Remover spinner icon si existe
    const spinnerIcon = spinner.querySelector('.spinner-border');
    if (spinnerIcon) {
      spinnerIcon.remove();
    }
    
    slowMsg.classList.add('d-none');
    btnExcel.disabled = false;
    btnCharts.disabled = false;
    clearTimeout(slowTimer);
    controller = null;
    job_id = null;
  }
});
