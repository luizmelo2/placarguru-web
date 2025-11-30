# UX Review — Rodada 9

Contexto: revisão pós-unificação do cabeçalho e alinhamento de presets/tabelas. Foco em simplificar fluxo de filtros, remover redundâncias e modularizar markup compartilhado.

## 1) Hierarquia e mensagens
- **Título + resumo compacto**: manter apenas título da página + chips de status/contagem na primeira linha; mover métricas secundárias (acurácia, destaques) para uma linha abaixo quando houver dados, evitando duplicar copy entre header e tabela.
- **Estados claros**: substituir textos genéricos de sucesso/erro por mensagens curtas e específicas (ex.: "Exportação pronta" / "Falha ao exportar"), sempre via único `aria-live`.
- **Sem banners duplicados**: garantir que viewport/theme announcements não reapareçam em barras móveis; preferir rótulos embutidos nos chips.

## 2) Responsividade e breakpoints
- **Toolbar única no mobile**: consolidar busca rápida, chips de torneios e botão de limpar em um único bloco colapsável; remover segunda linha de atalhos quando não houver filtros ativos.
- **Tipografia escalável**: aplicar `clamp()` nos títulos e chips para evitar jumps entre 768–1024px, mantendo legibilidade sem quebra de linha.
- **Stickiness seletiva**: desativar sticky de colunas/zebras em mobile e tablets; manter apenas header/tab bar sticky para minimizar jitter em scroll horizontal.

## 3) Fluxo de filtros e estado
- **Defaults declarativos**: centralizar defaults de filtros em `state.py` (ex.: torneios e janela de datas) e consumir via helper, evitando declarar valores padrão em componentes individuais.
- **Reset previsível**: expor o botão de limpar próximo aos chips ativos com contador; evitar confirmação modal e restaurar sempre o preset padrão, incluindo densidade.
- **Busca sincronizada**: usar único campo controlado para sidebar e quick search; debounce de 250ms é suficiente para evitar ecos em URL/state.

## 4) Tabelas, densidade e presets
- **Presets consistentes**: alinhar `TABLE_COLUMN_PRESETS` com colunas realmente renderizadas, removendo campos calculados não exibidos em determinados modos (ex.: odds repetidas).
- **Densidade clara**: manter paddings iguais e variar apenas tipografia/badges entre "confortável" e "compacto"; evitar trocar altura de linha para não desalinharem ícones.
- **Legenda mínima**: colocar legenda de destaques sempre acima da tabela com ícones/cores curtos; remover tooltips redundantes em colunas textuais.

## 5) Gráficos e módulos auxiliares
- **Cards unificados**: agrupar gráficos e KPIs em um único container com heading e descrição curta; remover instruções repetidas de interação.
- **Fallbacks amigáveis**: exibir mensagens claras quando o dataset filtrado não suportar gráfico/estatística, ao invés de cards vazios.
- **Tokens de cor acessíveis**: revisar paleta de gráficos para contraste 4.5:1 em light/dark e diferenciar séries com padrões ou saturação variável.

## 6) Acessibilidade e interação
- **Foco reaproveitado**: reutilizar um token de `focus-ring` para tabs, chips, toggles e botões; remover box-shadows intensos no dark mode.
- **Ordem de leitura**: garantir que `tabindex` siga: tema → filtros → resultados → gráficos; evitar múltiplos elementos com `role="status"` na mesma região.
- **Labels curtos**: `aria-label` sucinto para highlights/viewport/export (ex.: "Viewport móvel ativo"), sem repetir o texto visível.

## 7) Modularização e limpeza
- **Componentes reutilizáveis**: extrair chips/badges do header e da tabela para helpers em `ui_components.py` (ex.: `render_chip`, `render_status_badge`), reduzindo HTML duplicado.
- **Helpers de filtro**: mover lógica de presets, densidade e clear/reset para funções dedicadas em `state.py` para que app e componentes compartilhem o mesmo contrato.
- **Documentação contínua**: atualizar `design/UX_index.md` com o status da rodada e pendências visíveis para as próximas entregas.
