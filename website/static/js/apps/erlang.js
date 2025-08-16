let dimensionChart = null;
let sensitivityChart = null;

function renderErlangCharts() {
  const data = window.ERLANG_RESULT;
  if (!data) return;

  const dimCanvas = document.getElementById('dimensionChart');
  if (dimCanvas && data.dimension_bar) {
    if (dimensionChart) dimensionChart.destroy();
    const bar = data.dimension_bar;
    dimensionChart = new Chart(dimCanvas, {
      type: 'bar',
      data: {
        labels: ['Actual', 'Recomendado'],
        datasets: [
          {
            data: [bar.actual, bar.recomendado],
            backgroundColor: ['#0d6efd', '#198754'],
          },
        ],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { min: bar.min, max: bar.max },
        },
      },
    });
  }

  const sensCanvas = document.getElementById('sensitivityChart');
  if (sensCanvas && data.sensitivity) {
    if (sensitivityChart) sensitivityChart.destroy();
    const sens = data.sensitivity;
    const recommended = data.agents_recommended;
    const verticalLine = {
      id: 'recommendedLine',
      afterDraw(chart) {
        if (recommended === undefined) return;
        const xScale = chart.scales.x;
        const x = xScale.getPixelForValue(recommended);
        const ctx = chart.ctx;
        ctx.save();
        ctx.strokeStyle = 'red';
        ctx.beginPath();
        ctx.moveTo(x, chart.chartArea.top);
        ctx.lineTo(x, chart.chartArea.bottom);
        ctx.stroke();
        ctx.restore();
      },
    };
    sensitivityChart = new Chart(sensCanvas, {
      type: 'line',
      data: {
        labels: sens.agents,
        datasets: [
          {
            label: 'SL (%)',
            data: sens.sl,
            borderColor: '#198754',
            yAxisID: 'y1',
            tension: 0.1,
          },
          {
            label: 'ASA (s)',
            data: sens.asa,
            borderColor: '#0d6efd',
            yAxisID: 'y2',
            tension: 0.1,
          },
        ],
      },
      options: {
        responsive: true,
        scales: {
          y1: { beginAtZero: true, max: 100, position: 'left' },
          y2: { beginAtZero: true, position: 'right' },
        },
      },
      plugins: [verticalLine],
    });
  }
}

document.addEventListener('DOMContentLoaded', renderErlangCharts);

