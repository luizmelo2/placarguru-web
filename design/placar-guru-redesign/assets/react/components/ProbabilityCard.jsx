import React from 'react';

export default function ProbabilityCard({ title, value, trend }) {
  return (
    <article className="data-card p-4 flex items-center justify-between">
      <div>
        <p className="text-xs uppercase tracking-[0.08em] text-slate-500">{title}</p>
        <p className="text-2xl font-semibold">{value}</p>
      </div>
      <span className="badge text-emerald-500 border-emerald-200 bg-emerald-50">{trend}</span>
    </article>
  );
}
