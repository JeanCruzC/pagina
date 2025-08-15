/**
 * Utilities to mount Plotly figures.
 * Usage: mountPlot('chartId', figureJson)
 */
window.mountPlot = function (containerId, figure) {
  if (!window.Plotly) return;
  const el = document.getElementById(containerId);
  if (!el || !figure) return;
  Plotly.react(el, figure.data, figure.layout || {}, {responsive: true});
};
