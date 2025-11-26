import React from 'react';
import GameCard from './components/GameCard';
import ProbabilityCard from './components/ProbabilityCard';
import FiltersBar from './components/FiltersBar';
import StatsTable from './components/StatsTable';

const mockGames = [
  { id: 1, home: 'Flamengo', away: 'Palmeiras', prob: 68, odd: 1.72, model: 'Combo Neural', score: '2 - 1' },
  { id: 2, home: 'Barcelona', away: 'Real Madrid', prob: 62, odd: 2.05, model: 'Monte Carlo', score: '2 - 1' },
];

const highlightRule = (g) => g.prob > 60 && g.odd > 1.2;

export default function App() {
  return (
    <div className="min-h-screen bg-[var(--bg)] text-[var(--text)]">
      <header className="sticky top-0 z-10 backdrop-blur-xl bg-[color-mix(in_srgb,var(--bg)_90%,transparent)] border-b border-[var(--stroke)]">
        <div className="max-w-6xl mx-auto flex items-center justify-between py-4 px-6">
          <div className="flex items-center gap-2">
            <div className="w-9 h-9 rounded-2xl bg-gradient-to-br from-sky-500 to-emerald-300 shadow-lg" />
            <span className="font-semibold tracking-tight text-lg">Placar Guru</span>
          </div>
          <FiltersBar />
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8 space-y-6">
        <section className="grid gap-4 md:grid-cols-2">
          {mockGames.map((game) => (
            <GameCard key={game.id} game={game} highlight={highlightRule(game)} />
          ))}
        </section>

        <section className="grid gap-4 md:grid-cols-3">
          <ProbabilityCard title="BTTS" value="64%" trend="+6%" />
          <ProbabilityCard title="Over 2.5" value="58%" trend="+3%" />
          <ProbabilityCard title="Odd média" value="1.92" trend="+0.05" />
        </section>

        <StatsTable games={mockGames} highlightRule={highlightRule} />
      </main>
    </div>
  );
}
