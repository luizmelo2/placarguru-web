# UX Review — Rodada 7

Contexto: Avaliação contínua do Placar Guru após a consolidação de estados compartilhados, presets de colunas e cabeçalho compacto. O foco aqui é refinar hierarquia, acessibilidade e simplificação da lógica/UI para reduzir complexidade percebida e custo de manutenção.

## 1) Hierarquia e narrativa
- **Unificar header/hero**: hoje há dois blocos consecutivos (pg-header e pg-hero) com mensagens similares. Sugiro fundir em uma faixa única com: título + breadcrumb, resumo de filtros (chips), KPIs essenciais (jogos, acurácia, sugestões) e export CTA; evitar repetição de textos e chips duplicados.
- **KPIs contextuais**: mover o bloco de KPIs para dentro do topo quando houver dados; quando vazio, exibir mensagem de estado único (ex.: “Aplique filtros para habilitar PDF”) evitando cards vazios.
- **Reduzir ruído de chips**: chips de status, viewport, tema e resumo aparecem em múltiplos lugares. Concentre-os em uma linha e torne o restante texto corrido (ex.: “Última atualização 12/08 18:20”).

## 2) Responsividade e breakpoints
- **Colunas fixas e sticky**: revisar sticky apenas em desktop e remover no mobile para evitar travamento lateral. Garanta largura mínima para colunas-chave (home/away, previsão, odds) antes de aplicar stickiness.
- **Grid de KPIs adaptável**: usar `st.columns` com regras 1/2/4 colunas conforme breakpoint (<=640px, <=1024px, >1024px) para evitar overflow e deslocamento vertical.
- **Toolbar mobile simplificada**: substituir “atalhos rápidos + chips” por um único bloco com dropdown de torneio, range quick-picks e busca, economizando espaço vertical.

## 3) Fluxo de filtros e busca
- **Estado único para filtros**: hoje `pg_filters_cache` + `FilterState` coexistem; elimine duplicidade mantendo apenas `FilterState` e salvando em cache persistente ao atualizar (evita drift entre sidebar e mobile).
- **Reset claro e reversível**: adicionar botão “Limpar filtros” com confirmação leve e mensagem ARIA; mostrar contagem de filtros ativos ao lado para reduzir dúvida sobre estado.
- **Default previsível**: inicializar recorte de datas conforme tab (ex.: Finalizados = últimos 3 dias, Agendados = próximos 3 dias) e refletir no placeholder do datepicker; persistir escolha do usuário sem sobrescrever manualmente quando volta à aba.

## 4) Tabelas e legibilidade
- **Presets coesos**: alinhar `TABLE_COLUMN_PRESETS` com os usos no `st.data_editor`; remover colunas não exibidas do pipeline (ex.: odds_D se não renderizada). Evita cálculos e masks desnecessárias.
- **Densidade e zebrado**: aplicar zebra e maior padding horizontal no modo “comfortable”; no modo “compact”, reduzir tipografia mas manter altura mínima para evitar toques acidentais em mobile.
- **Legenda e tooltips**: manter legenda de highlights, mas padronizar tooltip curto (ex.: “Sugestão Guru: Prob ≥60% e Odd >1.20”). Esconder tooltips redundantes em colunas puramente textuais.

## 5) Cartões de desempenho e gráficos
- **Painel único**: consolidar cards de acerto, gráfico de barras e linha de tendência em um container com heading único e descrições curtas; reduzir quantidade de textos auxiliares repetidos (“Interativo”, “Ordene por coluna”).
- **Fallbacks claros**: quando `metrics_df` vazio, mostrar placeholder amigável (“Sem jogos finalizados neste recorte”) em vez de manter cartão vazio.
- **Acessibilidade nos gráficos**: garantir cores com contraste ≥4.5:1 entre barras e fundo; revisar paleta `chart_tokens` para tons mais distintos no modo escuro.

## 6) Acessibilidade e foco
- **Ordem de leitura**: verificar que componentes críticos (toggle de tema, filtros, tabela) seguem ordem lógica no DOM. Evitar inserir HTML custom acima do toggle se não for necessário.
- **Foco visível**: reforçar `:focus-visible` em chips e botões com outline consistente (mesma cor usada no tema) e remover sombras dispersivas no modo escuro.
- **ARIA concisa**: mensagens ARIA no topo devem ser únicas; evitar múltiplos elementos com `role="status"` para o mesmo contexto (ex.: exportação pronta).

## 7) Modularização e manutenção
- **Separar layout de estado**: mover HTML/CSS grandes do header/hero para funções em `ui_components.py`, aceitando dados calculados (counts, labels), para reduzir ruído em `app.py`.
- **Helpers para chips**: criar utilitário `render_chip(text, tone="ghost", aria_label=None)` e reutilizar, evitando strings HTML repetidas.
- **Documentação curta**: adicionar README rápido em `design/` explicando como cada rodada foi aplicada e status (pendente / implementado) para acompanhar a evolução.
