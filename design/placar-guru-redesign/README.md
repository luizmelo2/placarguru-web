# Redesign Placar Guru — UI/UX Blueprint

## Visão geral
Redesign premium inspirado em Notion, Linear, Figma, Stripe Dashboard e estética "futebol + data science". Estrutura mobile-first com glassmorphism moderado, toques de neubrutalism em realces e cards densos em dados.

## Paleta de cores
- **Light**: background `#0f172a0d` (azul-marinho translúcido), superfícies `#ffffff`, linhas `#e5e7eb`, texto primário `#0f172a`, secundário `#475569`, acentos `#2563eb` e `#22d3ee`, highlights premium `#bfff3b`.
- **Dark**: background `#0b1224`, superfícies `#0f172a`, vidro `rgba(255,255,255,0.04)`, texto primário `#e2e8f0`, secundário `#94a3b8`, acento principal `#60a5fa`, acento neon para recomendações `#bfff3b`, linhas `#1f2937`.
- **Estados**: sucesso `#10b981`, alerta `#f59e0b`, erro `#ef4444`, informação `#38bdf8`.

## Tipografia
- **Google Fonts**: `Inter` (UI, números), `Space Grotesk` (títulos, badges de modelo), `Roboto Mono` (odds, métricas de ROI).
- Hierarquia: H1 28-32 semibold, H2 22-24 semibold, H3 18-20 medium, corpo 14-16 regular, overline 12 uppercase tracking-wide.

## Estilos de componentes
- **Cards**: cantos 18px, borda 1px `var(--stroke)`, sombra suave `0 20px 60px rgba(0,0,0,0.12)`, fundo glass (dark) ou sólido (light). Cabeçalho com ícones minimalistas (Lucide), rodapé com chips de métricas. Hover: leve translateY(-2px) + borda acentuada.
- **Tabelas**: grid zebra sutil, header sticky, linhas com animação de entrada `slide-fade`. Células de odds usam `Roboto Mono`. Badges de status e mercado com cores consistentes.
- **Gráficos**: ApexCharts/Chart.js com gradientes verticais, cantos arredondados, tooltips com fundo vidro. Gauges semicírculo para confiança, linha de tendência com pontos luminosos.
- **Botões**: primary preenchido com gradiente `#2563eb → #22d3ee`, texto branco, raio 14px, sombra interna sutil. Secondary outline com vidro. Filtros como pills horizontais com ícones.
- **Filtros**: barra fixa no topo do conteúdo com chips; seleção múltipla usa estado preenchido. Dropdowns com cantos 14px e blur.
- **Micro animações**: hover de cards (translate + glow), skeleton shimmer, loaders circulares com gradiente, transição de modo claro/escuro 260ms ease-in-out, expansão de linhas de tabela ao focar, gráficos com easing `easeInOutCubic` e atraso escalonado.
- **Destaque automático de apostas**: se `probabilidade > 60%` **e** `odd > 1.20` → cartão e linha recebem contorno neon `#bfff3b`, badge "Sugestão Guru" e pulso suave.

## Telas principais
- **Home**: hero com headline, CTA duplo (Explorar previsões / Ver ROI), cards de highlights do dia, carrossel de campeonatos, bloco "Como calculamos" com passos e chips de modelo.
- **Dashboard**: KPIs (ROI, acurácia, yield), gráfico de tendência, barras de distribuição de gols, gauge de confiança do modelo, tabela de recomendações. Sidebar compacta com filtros de data/modelo/torneio.
- **Lista de jogos**: lista densa (mobile) e grade (desktop) com card de jogo, odds, probabilidade, placar previsto, chips de status e botão de detalhes. Filtros horizontais fixos.
- **Página do jogo**: header com equipes + horário, abas (Resumo, Probabilidades, Estatísticas, Histórico), gráfico BTTS, mapa de momentum, tabela de últimos 10 jogos, modal de análise detalhada com insights textuais.
- **Configurações**: preferências de modelo, intensidade do destaque, escolha de modo claro/escuro, unidade de odds, idioma, export (CSV/PDF). Switches com feedback imediato.

## Estrutura de pastas sugerida
```
design/placar-guru-redesign/
  README.md            # guia de design
  index.html           # protótipo estático pronto
  assets/
    theme.css          # tokens e estilos customizados
    main.js            # interações, mock de dados, dark mode
    chart-theme.js     # tema para Chart.js/ApexCharts
    react/             # componentes JSX opcionais
      App.jsx
      components/
        GameCard.jsx
        ProbabilityCard.jsx
        FiltersBar.jsx
        StatsTable.jsx
```

## Mockups textuais (high-level)
- **Home**: `[Navbar minimal] [Hero: "Previsões precisas em tempo real" + CTA] [Cards de destaque: jogo + prob + odd + badge Sugestão Guru] [Painel ROI/Winrate mini] [Lista "Top campeonatos"] [Footer com segurança e dados].`
- **Dashboard**: `[Header + filtros] [Row KPIs (ROI, Acurácia, BTTS acerto)] [Linha de tendência ROI] [Barras de gols previstos] [Tabela Recomendações com highlight neon] [Cards de modelo + confiança gauge].`
- **Lista de jogos**: `[Filtros horizontais scroll] [Cards responsivos: escudo, times, horário, odds H/D/A, prob total, badge Sugestão] [Toggle list/grid] [Botão detalhes].`
- **Página do jogo**: `[Header com escudos grandes + chips de campeonato] [Resumo de prob/odd + placar previsto] [Tabs] {Resumo: bullets; Probabilidades: gráfico stacked; Estatísticas: tabela responsiva; Histórico: timeline com badges de resultado}.`
- **Configurações**: `[Cards de preferências] [Switch dark/light] [Slider intensidade neon] [Select modelo padrão] [Botões de export e reset].`

## Sugestões de evolução
- Conectar API de probabilidades em tempo real e adicionar notificação de variação de odds.
- Criar modo "Value Bets" com heatmap de ROI histórico por campeonato.
- Salvar presets de filtros por usuário e compartilhamento via link curto.
- Adicionar onboarding interativo com tooltips contextuais.
- Integrar revisão de confiança do modelo por mercado (ex: BTTS vs 1x2).

## Racional de design
- Paleta escura sofisticada com acento neon comunica tecnologia e precisão sem cansar a leitura.
- Glassmorphism moderado traz futurismo; neubrutalism só em highlights para manter clareza.
- Tipografia neutra (Inter) + geométrica (Space Grotesk) equilibra seriedade e inovação; mono para dados reforça confiabilidade.
- Microinterações e skeletons melhoram percepção de velocidade e refinamento.
