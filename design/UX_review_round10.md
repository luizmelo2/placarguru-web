# UX Review — Rodada 10

Contexto: revisão após consolidação dos presets e toolbar móvel compartilhada. Objetivo: eliminar redundâncias remanescentes, simplificar estados e garantir modularização consistente nos componentes de header, filtros e tabelas.

## 1) Hierarquia e mensagens
- **Header enxuto**: manter apenas título + chips de resumo (jogos, destaques, acurácia) na primeira linha; mover ações (exportar, tema) para o alinhamento à direita e evitar segunda barra.
- **Estados explícitos**: reutilizar o `aria-live` único para export/busca/viewport; mensagens curtas como "Exportação pronta" ou "Nada encontrado" na busca.
- **Feedback de filtros**: exibir contagem de filtros ativos como badge discreto na barra e repetir a legenda de destaques somente quando houver dados carregados.

## 2) Responsividade e breakpoints
- **Breakpoints alinhados**: usar o mesmo breakpoint móvel em CSS e Python (`state.py`) para stickiness, densidade e colunas renderizadas.
- **Toolbar móvel unificada**: consolidar busca, chips de torneios e botão de limpar em um container colapsável; remover tooltips redundantes em mobile.
- **Tipografia com clamp**: aplicar `clamp()` nos headings e chips para reduzir jumps entre 768–1024px; evitar truncamento agressivo em tablets.

## 3) Fluxo de filtros e estado
- **Defaults centralizados**: consumir presets/odds/filtros diretamente de `state.py`, sem reatribuir defaults nos componentes.
- **Reset previsível**: botão de limpar sempre reseta filtros + densidade para o preset padrão sem modal; badge de contagem deve zerar junto.
- **Busca sincronizada**: campo único controlado para sidebar e quick search com debounce ~250ms; evitar eco em URL ou múltiplos handlers.

## 4) Tabelas e presets
- **Presets fidedignos**: garantir que `TABLE_COLUMN_PRESETS` só inclua colunas renderizadas; remover campos ocultos ou calculados não exibidos.
- **Densidade coerente**: variar somente tipografia/badges entre "confortável" e "compacto"; paddings e altura de linha constantes para evitar desalinhamento.
- **Legenda enxuta**: posicionar legenda de destaques acima da tabela com ícone/cor curtos; evitar tooltips duplicados em células textuais.

## 5) Gráficos e módulos auxiliares
- **Cards agrupados**: envolver gráficos/KPIs em container único com heading + descrição curta; remover instruções repetidas.
- **Fallback amigável**: mensagens claras para datasets vazios ou filtros sem dados (ex.: "Nenhum jogo disponível neste recorte").
- **Paleta acessível**: revisar cores dos gráficos para contraste 4.5:1 em claro/escuro; preferir variação de saturação ou padrão para múltiplas séries.

## 6) Acessibilidade e interação
- **Foco consistente**: reutilizar token de `focus-ring` para tabs, chips, toggles e botões; evitar sombras intensas no dark mode.
- **Ordem de leitura**: tabulação seguindo tema → filtros → resultados → gráficos; não duplicar `role="status"` no header e na tabela.
- **Labels curtos**: `aria-label` sucinto para destaques/viewport/export (ex.: "Viewport móvel ativo"), sem repetir texto visível.

## 7) Modularização e limpeza
- **Componentes reutilizáveis**: extrair chips/badges e legendas em helpers de `ui_components.py`, evitando HTML duplicado entre header e tabela.
- **Helpers de filtro**: concentrar presets, densidade e reset/clear em funções de `state.py` já consumidas pelo app; remover lógica paralela.
- **Documentação contínua**: registrar o status desta rodada em `design/UX_index.md` e listar pendências claras para próxima entrega.
