import React from 'react';

const filters = ['Hoje', 'Ao vivo', 'Modelos', 'Campeonatos', 'Odds 1.20+'];

export default function FiltersBar() {
  return (
    <div className="flex items-center gap-2 overflow-x-auto">
      {filters.map((f) => (
        <button key={f} className="tab-pill text-sm whitespace-nowrap">
          {f}
        </button>
      ))}
    </div>
  );
}
