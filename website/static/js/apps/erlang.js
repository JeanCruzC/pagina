(function(){
  const dataEl = document.getElementById('erlang-data');
  const modeSelect = document.getElementById('erlang-mode');
  const patienceField = document.getElementById('patience-field');

  if(modeSelect && patienceField){
    const togglePatience = ()=>{
      patienceField.style.display = modeSelect.value === 'erlang_x' ? '' : 'none';
    };
    togglePatience();
    modeSelect.addEventListener('change', togglePatience);
  }

  if(!dataEl) return;
  let result = {};
  try { result = JSON.parse(dataEl.textContent); } catch(e) { return; }

  // Dimension bar
  const bar = document.getElementById('dimension-bar');
  if(bar){
    const actual = result.agents || 0;
    const rec = result.recommended || 0;
    const maxVal = Math.max(actual, rec, 1);
    bar.innerHTML = `<div class="progress" style="height:30px">`+
      `<div class="progress-bar bg-danger" role="progressbar" style="width:${(actual/maxVal)*100}%">Actual (${actual})</div>`+
      `<div class="progress-bar bg-success" role="progressbar" style="width:${(rec/maxVal)*100}%">Recomendado (${rec})</div>`+
      `</div>`;
  }

  // Sensitivity chart
  const canvas = document.getElementById('sensitivity-chart');
  if(canvas && result.sensitivity){
    const s = result.sensitivity;
    new Chart(canvas, {
      type: 'line',
      data: {
        labels: s.agents,
        datasets: [
          {
            label: 'SL %',
            data: (s.sl || []).map(v => v*100),
            borderColor: 'blue',
            yAxisID: 'y',
            tension: 0.3
          },
          {
            label: 'ASA (s)',
            data: s.asa || [],
            borderColor: 'red',
            yAxisID: 'y1',
            tension: 0.3
          }
        ]
      },
      options: {
        responsive: true,
        interaction: { mode: 'index', intersect: false },
        stacked: false,
        plugins: {
          annotation: {
            annotations: {
              rec: {
                type: 'line',
                xMin: result.recommended,
                xMax: result.recommended,
                borderColor: 'orange',
                borderDash: [6,6],
                label: {
                  content: 'Óptimo',
                  enabled: true,
                  position: 'start'
                }
              }
            }
          }
        },
        scales: {
          y: { type: 'linear', position: 'left', min:0, max:100 },
          y1: { type: 'linear', position: 'right', grid: { drawOnChartArea: false } }
        }
      }
    });
  }

  function download(ext){
    if(!result.sensitivity) return;
    const s = result.sensitivity;
    const rows = [['agents','sl','asa']];
    for(let i=0; i < (s.agents||[]).length; i++){
      rows.push([s.agents[i], s.sl[i], s.asa[i]]);
    }
    const csv = rows.map(r=>r.join(',')).join('\n');
    const blob = new Blob([csv], {type: ext==='csv' ? 'text/csv' : 'application/vnd.ms-excel'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `erlang.${ext}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  const csvBtn = document.getElementById('download-csv');
  if(csvBtn) csvBtn.addEventListener('click', ()=>download('csv'));
  const xlsBtn = document.getElementById('download-excel');
  if(xlsBtn) xlsBtn.addEventListener('click', ()=>download('xlsx'));
})();
