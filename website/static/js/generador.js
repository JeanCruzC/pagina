/* Helper to update the text next to a slider. */
function updateSliderValue(sliderId, labelId) {
  const slider = document.getElementById(sliderId);
  const label = document.getElementById(labelId);
  if (!slider || !label) return;
  label.textContent = slider.value;
  slider.addEventListener('input', () => {
    label.textContent = slider.value;
  });
}

/* Show/hide FT related checkboxes. */
function toggleFTOptions() {
  const ft = document.getElementById('ft');
  const box = document.getElementById('ft_opts');
  if (ft && box) box.style.display = ft.checked ? 'block' : 'none';
}

/* Show/hide PT related checkboxes. */
function togglePTOptions() {
  const pt = document.getElementById('pt');
  const box = document.getElementById('pt_opts');
  if (pt && box) box.style.display = pt.checked ? 'block' : 'none';
}

/* Toggle custom parameter sections based on profile selector. */
function toggleCustomParams() {
  const profile = document.getElementById('profile');
  const custom = document.getElementById('personalizado');
  const jean = document.getElementById('jean');
  if (!profile || !custom || !jean) return;
  const val = profile.value;
  custom.style.display = val === 'Personalizado' ? 'block' : 'none';
  jean.style.display = val === 'JEAN Personalizado' ? 'block' : 'none';
}

/* Render the results section using the JSON returned by the server. */
function displayResults(data) {
  console.log('📊 Mostrando resultados:', data);
  
  const metrics = document.getElementById('metrics');
  if (data.metrics && metrics) {
    const totalAgents = document.getElementById('m_agents');
    const coverage = document.getElementById('m_cov');
    const overstaffing = document.getElementById('m_over');
    const understaffing = document.getElementById('m_under');
    
    if (totalAgents) totalAgents.textContent = data.metrics.total_agents;
    if (coverage) coverage.textContent = data.metrics.coverage_percentage.toFixed(1) + '%';
    if (overstaffing) overstaffing.textContent = data.metrics.overstaffing;
    if (understaffing) understaffing.textContent = data.metrics.understaffing;
    
    metrics.style.display = 'flex';
  }

  if (data.heatmaps) {
    const demandImg = document.getElementById('hm-demand');
    const coverageImg = document.getElementById('hm-coverage');
    const diffImg = document.getElementById('hm-diff');
    
    if (data.heatmaps.demand && demandImg) {
      demandImg.src = 'data:image/png;base64,' + data.heatmaps.demand;
    }
    if (data.heatmaps.coverage && coverageImg) {
      coverageImg.src = 'data:image/png;base64,' + data.heatmaps.coverage;
    }
    if (data.heatmaps.difference && diffImg) {
      diffImg.src = 'data:image/png;base64,' + data.heatmaps.difference;
    }
  }

  const dl = document.getElementById('downloadBtn');
  if (dl && data.download_url) {
    dl.href = data.download_url;
    dl.style.display = 'inline-block';
  }
  
  alert('✅ ¡Optimización completada exitosamente!');
}

/* Show error message */
function showError(message) {
  console.error('❌ Error:', message);
  alert('❌ Error: ' + message);
}

/* Show loading state */
function showLoading(show) {
  const progressContainer = document.getElementById('progress');
  const progressBar = document.getElementById('progressBar');
  const generateBtn = document.querySelector('button[type="submit"]');
  
  if (progressContainer && progressBar) {
    if (show) {
      progressContainer.style.display = 'block';
      progressBar.style.width = '100%';
    } else {
      progressBar.style.width = '0%';
      progressContainer.style.display = 'none';
    }
  }
  
  if (generateBtn) {
    generateBtn.disabled = show;
    generateBtn.textContent = show ? 'Procesando...' : 'Generar';
  }
}

/* Event bindings - Wait for DOM to load */
function initGenerator() {
  console.log('🚀 Inicializando Generador de Horarios...');
  
  // Verificar que todos los elementos existen
  const form = document.getElementById('genForm');
  const submitBtn = document.querySelector('button[type="submit"]');
  
  if (!form) {
    console.error('❌ No se encontró el formulario con ID "genForm"');
    return;
  }
  
  if (!submitBtn) {
    console.error('❌ No se encontró el botón de envío');
    return;
  }
  
  // Initialize sliders
  updateSliderValue('iterations', 'it_val');
  updateSliderValue('solver_time', 'time_val');
  updateSliderValue('coverage', 'cov_val');
  updateSliderValue('break_from_start', 'bstart_val');
  updateSliderValue('break_from_end', 'bend_val');
  
  // Initialize toggles
  toggleFTOptions();
  togglePTOptions();
  toggleCustomParams();

  // Add event listeners
  const profileSelect = document.getElementById('profile');
  const ftCheckbox = document.getElementById('ft');
  const ptCheckbox = document.getElementById('pt');
  
  if (profileSelect) {
    profileSelect.addEventListener('change', toggleCustomParams);
  }
  if (ftCheckbox) {
    ftCheckbox.addEventListener('change', toggleFTOptions);
  }
  if (ptCheckbox) {
    ptCheckbox.addEventListener('change', togglePTOptions);
  }

  // Agregar evento al formulario
  form.addEventListener('submit', async function(ev) {
    ev.preventDefault();
    ev.stopPropagation();
    console.log('🚀 Formulario enviado');
    
    // Verificar archivo
    const fileInput = document.querySelector('input[type="file"][name="excel"]');
    if (!fileInput || !fileInput.files || !fileInput.files[0]) {
      showError('Por favor selecciona un archivo Excel');
      return;
    }
    
    console.log('📁 Archivo válido:', fileInput.files[0].name);
    showLoading(true);

    const formData = new FormData(form);
    console.log('🔍 Verificando FormData...');
    for (const [key, value] of formData.entries()) {
      console.log(`${key}:`, value instanceof File ? value.name : value);
    }

    try {
      console.log('📡 Enviando petición al servidor...');
      const response = await Promise.race([
        fetch('/generador', {
          method: 'POST',
          body: formData,
          headers: { 'Accept': 'application/json' }
        }),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Timeout')), 30000)
        )
      ]);
      
      console.log('✅ Respuesta recibida:', response.status);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Error del servidor');
      }
      
      const data = await response.json();
      console.log('📊 Datos recibidos exitosamente');

      try {
        displayResults(data);
      } catch (err) {
        console.error('❌ Error mostrando resultados:', err);
        showError('Error: ' + err.message);
      }

    } catch (error) {
      console.error('❌ Error completo:', error);
      if (error.stack) {
        console.error('❌ Stack trace:', error.stack);
      }
      showError('Error de conexión: ' + error.message);
    } finally {
      showLoading(false);
    }
  });
  
  console.log('✅ Generador de Horarios inicializado correctamente');
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initGenerator);
} else {
  initGenerator();
}
