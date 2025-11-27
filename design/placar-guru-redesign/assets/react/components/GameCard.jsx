import React from 'react';

export default function GameCard({ game, highlight }) {
  return (
    <article className={`data-card p-5 flex flex-col gap-3 ${highlight ? 'neon-highlight' : ''}`}>
      <header className="flex items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.08em] text-slate-500">{game.model}</p>
          <h3 className="text-lg font-semibold">{game.home} x {game.away}</h3>
        </div>
        {highlight && (
          <span className="badge" style={{ background: 'var(--neon)', borderColor: 'var(--neon)', color: '#0f172a' }}>
            Sugestão Guru
          </span>
        )}
      </header>
      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 rounded-xl border border-[var(--stroke)] bg-[color-mix(in_srgb,var(--panel)_85%,transparent)]">
          <p className="text-xs text-slate-500">Probabilidade</p>
          <p className="text-xl font-semibold">{game.prob}%</p>
        </div>
        <div className="p-3 rounded-xl border border-[var(--stroke)] bg-[color-mix(in_srgb,var(--panel)_85%,transparent)]">
          <p className="text-xs text-slate-500">Odd</p>
          <p className="text-xl font-semibold font-mono">{game.odd}</p>
        </div>
      </div>
      <footer className="flex items-center justify-between">
        <span className="text-sm text-slate-500">Placar previsto {game.score}</span>
        <button className="btn-ghost text-sm">Detalhes</button>
      </footer>
    </article>
  );
}
