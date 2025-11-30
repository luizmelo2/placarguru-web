# Revisão de UI/UX — Rodada 3

## 1. Identidade visual e hierarquia
- A topbar continua com gradiente/glow fortes e sem navegação ou CTA visível; em telas pequenas ela só mostra logo e subtítulo, desperdiçando o espaço sticky. Considerar uma barra mais compacta com breadcrumbs, botão de exportação e estado de filtros ativos para justificar o destaque visual.【F:styles.py†L90-L188】【F:app.py†L96-L179】
- A hero ainda usa badge textual para o tema enquanto o toggle já existe acima, gerando redundância e ocupando área nobre. Trocar o badge por um atalho ou resumo de filtros aplicados tornaria o header mais funcional.【F:app.py†L77-L179】【F:app.py†L411-L446】
- Há muitos blocos com sombras/gradientes sobrepostos (topbar, hero, cards de tabela e charts). Consolidar intensidade de sombra e reduzir radial overlays melhora legibilidade, sobretudo no dark mode.【F:styles.py†L90-L236】【F:styles.py†L229-L240】

## 2. Responsividade e layout
- A detecção de viewport segue gravando `vw` na query string a cada resize, mesmo com throttle. Em ambientes com links compartilhados, isso continua poluindo a URL e cria dependência do navegador para restaurar o estado. Avalie mover o valor para `st.session_state` via `components.html` ou WebRTC storage em vez de `replaceState`.【F:app.py†L128-L178】
- A toolbar móvel de filtros rápidos é estática e não exibe o estado atual (ex.: torneio já aplicado no sidebar). Mostrar o valor selecionado e sincronizar com `session_state` evita divergências entre o atalho e o filtro real.【F:app.py†L264-L317】【F:ui_components.py†L156-L199】
- O layout da hero + download gera salto visual: o botão de PDF fica logo abaixo mesmo quando não há dados (apenas desabilitado). Ocultar o bloco inteiro quando `curr_df` estiver vazio ou movê-lo para um card lateral reduz scroll inicial em mobile.【F:app.py†L360-L460】

## 3. Filtros e fluxo de uso
- O sidebar continua habilitado por padrão e duplica filtros principais já renderizados no corpo, forçando o usuário a percorrer duas regiões. Tornar o menu colapsado por default e mover apenas controles avançados (odds, presets) para lá simplifica o fluxo.【F:app.py†L35-L75】【F:app.py†L255-L317】【F:ui_components.py†L156-L199】
- Multiselects ainda carregam todas as equipes por padrão no desktop, criando listas longas e sem indicação de seleção atual. Introduzir chips-resumo e limitar altura dos dropdowns melhora a percepção do recorte ativo.【F:ui_components.py†L81-L134】
- O controle de list view fica fora do contexto dos filtros e se repete em mensagens (“Visão lista”/“Visão tabela”). Move-lo para a barra de status ou para a hero com toggle visível manteria consistência e reduziria texto redundante.【F:app.py†L264-L533】

## 4. Apresentação de dados
- A tabela “glassy” segue usando `st.dataframe` com scroll interno; em telas estreitas as colunas de odds e status ficam truncadas e o ícone de destaque Guru é apenas uma estrela textual. Considerar `st.data_editor` com colunas essenciais, coloração por status e ícone visual (ex.: chip/emoji lateral).【F:ui_components.py†L22-L75】【F:app.py†L491-L533】
- A métrica “Sugestão Guru” aparece no KPI e nas linhas, mas sem tooltip explicando a regra (prob ≥ 60% e odd > 1.20). Incluir help ou legenda próxima à tabela evita inferências erradas do usuário.【F:app.py†L430-L443】【F:ui_components.py†L52-L75】
- O bloco de ocultar lista de jogos finalizados ocupa espaço mesmo quando não há gráficos exibidos. Esconder o card quando `hide_games` está ativo ou agrupar o toggle com a seção de gráficos deixaria a página mais enxuta.【F:app.py†L512-L733】

## 5. Tema, feedback e acessibilidade
- O toggle de tema não informa ao leitor de tela qual modo está ativo além do texto do checkbox; adicionar `aria-live` ou etiqueta dinâmica na topbar ajuda navegação assistiva. Além disso, falta variação de foco em chips/abas no dark mode para manter contraste suficiente.【F:app.py†L77-L123】【F:styles.py†L142-L188】
- A logo SVG segue sem `title/desc` e o container usa apenas `aria-label`, o que pode ser ignorado por alguns leitores. Incluir `role="img"` + `<title>` dentro do SVG melhora acessibilidade sem alterar o layout.【F:app.py†L96-L117】
- Várias ações importantes (download, ocultar lista, exportar gráfico) não têm tooltips ou mensagens de erro contextualizadas; quando o PDF falha ou está desabilitado, o usuário vê apenas um botão apagado. Substituir por `st.container` com estado visível e dica de “aplique filtros para habilitar” reduz frustração.【F:app.py†L450-L460】【F:app.py†L516-L540】

## 6. Recomendação de próximos passos
- Centralizar tokens em um arquivo CSS externo (ou módulo) e remover inline HTML/CSS nos `st.markdown` para facilitar lint e experimentos de tema.【F:app.py†L96-L178】【F:styles.py†L10-L240】
- Adicionar testes de screenshot (desktop/mobile, dark/light) e validar contraste mínimo de foco/bordas para chips, tabs e badges.【F:styles.py†L145-L188】【F:styles.py†L219-L240】
- Criar variantes compactas para hero/KPIs e para tabelas no mobile, priorizando status, horários e sugestão Guru antes de odds secundárias.【F:app.py†L411-L533】【F:ui_components.py†L22-L107】
