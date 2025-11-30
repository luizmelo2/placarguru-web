# UX Review — Rodada 8

Contexto: revisão após consolidação de estados compartilhados e ajustes de header/hero compactos. O objetivo é reduzir redundâncias, modularizar componentes e garantir acessibilidade/consistência entre Python e CSS.

## 1) Hierarquia e mensagens
- **Header único e enxuto**: o topo mistura toggle de tema, breadcrumbs/estado e resumos em múltiplas linhas. Avalie mover o toggle para a área de ações (lado direito) e manter somente brand + breadcrumb + chips de status numa linha e o resumo (jogos/highlights/acurácia) em outra, evitando repetição de textos.
- **Mensagens de status unificadas**: há anúncios de tema e avisos de exportação/viewport espalhados. Centralize `aria-live` em um único contêiner `status` no header para não haver leitores de tela lendo mensagens duplicadas.
- **KPIs contextuais**: quando não houver dados, use uma única linha de empty-state no topo em vez de cards vazios; quando houver, alinhe KPIs/contagens ao lado do título para reduzir scroll inicial.

## 2) Responsividade e breakpoints
- **Breakpoint consistente**: Python define `MOBILE_BREAKPOINT=1024` e o CSS também; valide que listeners JS (viewport) e media queries compartilham o mesmo valor e use `rem`/`clamp` para tipografia em <768px para evitar zoom involuntário.
- **Toolbar mobile compacta**: unificar chips, quick search e seleção de torneios em uma única barra colapsável acima da lista no mobile, evitando dois blocos sequenciais (atalhos + filtros). Priorize espaço para tabela/lista.
- **Stickiness seletiva**: manter sticky no header, mas desativar stickiness de colunas/listras em mobile para evitar travamento horizontal; só aplicar `position: sticky` em desktop.

## 3) Fluxo de filtros e estado
- **Fonte única de verdade**: hoje `FilterState` grava em `pg_filters_cache` e também persiste filtros principais; remova qualquer uso remanescente de caches paralelos (ex.: variáveis soltas em sessão) e mantenha helper único para ler/escrever.
- **Reset previsível**: mantenha o botão de limpar filtros sempre visível próximo aos chips de contagem; aplique confirmação leve apenas quando houver mais de um filtro ativo e reponha defaults conforme aba (agendados/finalizados).
- **Busca e torneio sincronizados**: garanta que o valor de busca rápido reflita imediatamente no sidebar e vice-versa; idem para torneios selecionados, evitando drift entre mobile e desktop.

## 4) Tabelas, densidade e presets
- **Presets enxutos**: alinhe `TABLE_COLUMN_PRESETS` com o DataFrame real, removendo colunas não exibidas/derivadas quando o preset não usa (ex.: odds redundantes). Isso reduz custo de processamento e ordenações inúteis.
- **Densidade por contexto**: mantenha zebra e padding generoso em “comfortable”, mas reduza somente tipografia no modo “compact”; evite alterar altura de linha a ponto de quebrar ícones/badges.
- **Legenda e tooltips**: deixe a legenda de highlights fixa acima da tabela e simplifique tooltips para mensagens curtas; evite tooltip em colunas puramente textuais (home/away) para melhorar leitura em touch.

## 5) Gráficos e cartões de desempenho
- **Painel unificado**: agrupe barras, linha de tendência e métricas em um único container com heading e descrição curtos; remova textos repetitivos como “interativo” ou instruções de ordenação que já são padrão do Streamlit.
- **Fallbacks explícitos**: quando não houver dados ou a aba não suportar gráfico, exiba placeholder amigável (ex.: “Nenhum jogo filtrado neste período”) em vez de deixar cards vazios.
- **Paleta acessível**: revise `chart_tokens` para contrastes mínimos 4.5:1 em dark/light; use cores distintas para série principal vs. benchmarks para leitores daltônicos.

## 6) Acessibilidade e foco
- **Foco visível consistente**: reutilizar um helper de chip/botão com outline definido (ex.: `--focus-ring`) para tabs, chips e toggles; remover sombras pesadas no dark mode quando focado.
- **Ordem de navegação**: confirmar que a navegação por tab segue tema → filtros → lista → gráficos, sem saltar para HTML escondido; evitar múltiplos elementos com `role="status"` próximos.
- **Texto alternativo**: garantir que ícones de highlights, exportação e viewport tenham `aria-label` curto e distinto (ex.: “Sugestão Guru habilitada”).

## 7) Modularização e manutenção
- **Componentes de header/chips**: extrair o markup do header e chips de status para `ui_components.py` com props (tema, viewport label, counts). Reduz `app.py` e facilita reutilizar em futuras páginas.
- **Helpers reutilizáveis**: criar utilitário `render_chip(texto, tom="ghost")` + `render_badge(status)` e usá-los tanto na topbar quanto na tabela, evitando HTML repetido e inconsistências.
- **Documentação de rodada**: adicionar no `design/` um índice curto indicando status (pendente/feito) de cada recomendação das rodadas anteriores para acompanhar dívida de design.
