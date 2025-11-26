const games = [
  {
    id: 1,
    home: 'Flamengo',
    away: 'Palmeiras',
    start: '19:00',
    tournament: 'Brasileirão Série A',
    probability: 68,
    odds: { home: 1.72, draw: 3.4, away: 4.8 },
    suggestedOdd: 1.72,
    model: 'Combo Neural',
    score: '2 - 1',
    status: 'Em breve',
    roi: '+12.4%',
  },
  {
    id: 2,
    home: 'Chelsea',
    away: 'Liverpool',
    start: '16:30',
    tournament: 'Premier League',
    probability: 58,
    odds: { home: 2.2, draw: 3.7, away: 3.1 },
    suggestedOdd: 2.2,
    model: 'Expected Goals',
    score: '1 - 1',
    status: 'Ao vivo',
    roi: '+7.8%',
  },
  {
    id: 3,
    home: 'Barcelona',
    away: 'Real Madrid',
    start: '17:00',
    tournament: 'La Liga',
    probability: 62,
    odds: { home: 2.05, draw: 3.8, away: 3.5 },
    suggestedOdd: 2.05,
    model: 'Monte Carlo',
    score: '2 - 1',
    status: 'Programado',
    roi: '+10.1%',
  },
  {
    id: 4,
    home: 'Benfica',
    away: 'Porto',
    start: '15:00',
    tournament: 'Liga Portugal',
    probability: 64,
    odds: { home: 1.9, draw: 3.5, away: 3.9 },
    suggestedOdd: 1.9,
    model: 'Combo Neural',
    score: '2 - 0',
    status: 'Ao vivo',
    roi: '+6.2%',
  },
];

const highlightRule = (game) => game.probability > 60 && game.suggestedOdd > 1.2;

function renderGameCards() {
  const grid = document.getElementById('games-grid');
  grid.innerHTML = '';

  games.forEach((game) => {
    const isHighlighted = highlightRule(game);
    const card = document.createElement('article');
    card.className = `data-card p-5 flex flex-col gap-4 ${isHighlighted ? 'neon-highlight' : ''}`;
    card.innerHTML = `
      <header class="flex justify-between items-start">
        <div>
          <p class="text-xs uppercase tracking-[0.08em] text-slate-500 dark:text-slate-400">${game.tournament}</p>
          <div class="flex items-center gap-2">
            <h3 class="text-lg font-semibold">${game.home} <span class="text-slate-400">vs</span> ${game.away}</h3>
            ${isHighlighted ? '<span class="badge text-slate-900 dark:text-slate-900" style="background: var(--neon); border-color: var(--neon);">Sugestão Guru</span>' : ''}
          </div>
          <p class="text-sm text-slate-500 dark:text-slate-400">${game.start} · ${game.model}</p>
        </div>
        <div class="flex items-center gap-2">
          <span class="chip text-xs">Prob. ${game.probability}%</span>
          <span class="chip text-xs">Odd ${game.suggestedOdd}</span>
        </div>
      </header>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div class="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/70 border border-slate-100 dark:border-slate-700">
          <p class="text-xs text-slate-500">Placar previsto</p>
          <p class="text-xl font-semibold">${game.score}</p>
        </div>
        <div class="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/70 border border-slate-100 dark:border-slate-700">
          <p class="text-xs text-slate-500">Odd Casa</p>
          <p class="text-xl font-semibold font-mono">${game.odds.home}</p>
        </div>
        <div class="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/70 border border-slate-100 dark:border-slate-700">
          <p class="text-xs text-slate-500">Odd Empate</p>
          <p class="text-xl font-semibold font-mono">${game.odds.draw}</p>
        </div>
        <div class="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/70 border border-slate-100 dark:border-slate-700">
          <p class="text-xs text-slate-500">Odd Visitante</p>
          <p class="text-xl font-semibold font-mono">${game.odds.away}</p>
        </div>
      </div>
      <footer class="flex flex-wrap items-center gap-2">
        <span class="badge">${game.status}</span>
        <span class="badge">ROI ${game.roi}</span>
        <button class="btn-ghost text-sm">Detalhes</button>
      </footer>
    `;
    grid.appendChild(card);
  });
}

function renderRecommendationsTable() {
  const tbody = document.getElementById('table-body');
  tbody.innerHTML = '';
  games.forEach((game) => {
    const row = document.createElement('tr');
    row.className = `border-b border-slate-100 dark:border-slate-800 ${highlightRule(game) ? 'neon-highlight' : ''}`;
    row.innerHTML = `
      <td class="py-3">${game.tournament}</td>
      <td class="py-3 font-semibold">${game.home} x ${game.away}</td>
      <td class="py-3 text-center">${game.probability}%</td>
      <td class="py-3 text-center font-mono">${game.suggestedOdd}</td>
      <td class="py-3 text-center">${game.status}</td>
      <td class="py-3 text-right font-semibold">${game.roi}</td>
    `;
    tbody.appendChild(row);
  });
}

function setupModeToggle() {
  const toggle = document.getElementById('mode-toggle');
  const html = document.documentElement;
  const setMode = (dark) => {
    html.classList.toggle('dark', dark);
    toggle.innerText = dark ? 'Light Mode' : 'Dark Mode';
  };
  setMode(window.matchMedia('(prefers-color-scheme: dark)').matches);
  toggle.addEventListener('click', () => setMode(!html.classList.contains('dark')));
}

function setupCharts() {
  const ctxRoi = document.getElementById('roi-chart');
  const ctxGoals = document.getElementById('goals-chart');
  const ctxGauge = document.getElementById('gauge-chart');

  const lineGrad = ctxRoi.getContext('2d').createLinearGradient(0, 0, 0, 400);
  lineGrad.addColorStop(0, 'rgba(96, 165, 250, 0.35)');
  lineGrad.addColorStop(1, 'rgba(96, 165, 250, 0)');

  new Chart(ctxRoi, {
    type: 'line',
    data: {
      labels: ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom'],
      datasets: [
        {
          label: 'ROI semanal',
          data: [8, 12, 10, 16, 14, 18, 22],
          fill: true,
          backgroundColor: lineGrad,
          borderColor: '#60a5fa',
          tension: 0.42,
          pointRadius: 4,
          pointBackgroundColor: '#22d3ee',
        },
      ],
    },
    options: {
      plugins: { legend: { display: false }, tooltip: chartTooltip },
      scales: { x: chartAxisX, y: chartAxisY },
    },
  });

  new Chart(ctxGoals, {
    type: 'bar',
    data: {
      labels: ['0-1', '2', '3', '4+'],
      datasets: [
        {
          label: 'Gols previstos',
          data: [12, 30, 24, 14],
          backgroundColor: ['#0ea5e9', '#22d3ee', '#60a5fa', '#a855f7'],
          borderRadius: 10,
        },
      ],
    },
    options: {
      plugins: { legend: { display: false }, tooltip: chartTooltip },
      scales: { x: chartAxisX, y: chartAxisY },
    },
  });

  new Chart(ctxGauge, {
    type: 'doughnut',
    data: {
      labels: ['Confiança do modelo', ''],
      datasets: [
        {
          data: [86, 14],
          backgroundColor: ['#bfff3b', 'rgba(226,232,240,0.1)'],
          borderWidth: 0,
          cutout: '75%',
          rotation: -90,
          circumference: 180,
        },
      ],
    },
    options: {
      plugins: {
        legend: { display: false },
        tooltip: chartTooltip,
      },
    },
  });
}

document.addEventListener('DOMContentLoaded', () => {
  renderGameCards();
  renderRecommendationsTable();
  setupModeToggle();
  setupCharts();
});
