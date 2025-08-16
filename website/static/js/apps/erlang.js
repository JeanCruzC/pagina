// Plotly charts for Erlang page
window.addEventListener('DOMContentLoaded', () => {
  const dimEl = document.getElementById('erlang-dimension-bar');
  if (dimEl && dimEl.dataset.bar && window.Plotly) {
    try {
      const data = JSON.parse(dimEl.dataset.bar);
      const trace = {
        type: 'bar',
        x: ['Actual', 'Recomendado'],
        y: [data.current, data.recommended],
        marker: {
          color: ['#0d6efd', '#198754'],
        },
      };
      const layout = {
        margin: { t: 10, r: 10, b: 40, l: 40 },
        yaxis: { title: 'Agentes' },
      };
      Plotly.react(dimEl, [trace], layout, { responsive: true });
    } catch (e) {
      console.error('Failed to render dimension bar', e);
    }
  }

  const sensEl = document.getElementById('erlang-sensitivity');
  if (sensEl && sensEl.dataset.figure && window.Plotly) {
    try {
      const fig = JSON.parse(sensEl.dataset.figure);
      Plotly.react(sensEl, fig.data, fig.layout || {}, { responsive: true });
    } catch (e) {
      console.error('Failed to render sensitivity chart', e);
    }
  }
});
