const chartTooltip = {
  backgroundColor: 'rgba(15, 23, 42, 0.9)',
  borderColor: 'rgba(255,255,255,0.12)',
  borderWidth: 1,
  titleColor: '#e2e8f0',
  bodyColor: '#cbd5e1',
  padding: 10,
  displayColors: false,
};

const axisCommon = {
  grid: { color: 'rgba(148, 163, 184, 0.18)', drawTicks: false },
  ticks: { color: '#94a3b8', font: { family: 'Inter', size: 12 } },
};

const chartAxisX = { ...axisCommon, border: { display: false } };
const chartAxisY = { ...axisCommon, beginAtZero: true, border: { display: false } };
