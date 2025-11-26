import React from 'react';

export default function StatsTable({ games, highlightRule }) {
  return (
    <div className="data-card p-4">
      <header className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-semibold tracking-tight">Recomendações</h4>
        <span className="badge">Destacadas: prob > 60% & odd > 1.20</span>
      </header>
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="text-left text-slate-500">
            <tr>
              <th className="py-2">Jogo</th>
              <th className="py-2">Prob</th>
              <th className="py-2">Odd</th>
              <th className="py-2 text-right">Placar</th>
            </tr>
          </thead>
          <tbody className="text-[var(--text)]">
            {games.map((g) => (
              <tr
                key={g.id}
                className={`${highlightRule(g) ? 'neon-highlight' : ''} border-b border-[var(--stroke)] last:border-0`}
              >
                <td className="py-3 font-medium">{g.home} x {g.away}</td>
                <td className="py-3">{g.prob}%</td>
                <td className="py-3 font-mono">{g.odd}</td>
                <td className="py-3 text-right text-slate-500">{g.score}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
