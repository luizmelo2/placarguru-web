# Revisão de Design (Rodada 5)

Avaliação considerando as implementações atuais (header compacto, quick filters sincronizados e tabela com highlights). Recomendações priorizadas em seis frentes.

## 1) Hierarquia e narrativa
- Consolidar a identidade (logo, breadcrumbs e chips) numa barra em coluna única no mobile, escondendo o placeholder vazio do `topbar_placeholder` e reusando o mesmo container em vez de grids separados. Ajuda a reduzir ruído visual e garante foco na tabela.
- Títulos e KPIs: expor um resumo de acerto por mercado/logo antes dos filtros, evitando que usuários percorram a página sem contexto. O bloco pode ser reaproveitado via função dedicada na `ui_components` para manter consistência entre desktop e mobile.

## 2) Layout responsivo
- Adotar breakpoints coerentes entre CSS e lógica Python (ex.: `modo_mobile` em `app.py` usa `<1100px`; CSS troca para 1 coluna em `900px`). Harmonizar para evitar flutuação de estados e chips saltando na troca de viewport.
- Nos filtros laterais, priorizar toggles/datas acima do fold e mover seletor de torneio para um drawer adicional no mobile, reduzindo scroll e repetição das chips de atalho.

## 3) Fluxo de busca e filtros
- Unificar o estado de pesquisa (`search_query`) e dos filtros rápidos em um dataclass ou dict único armazenado no `session_state`, permitindo inicialização e reset em um só lugar. Reduz código duplicado entre `filtros_ui` e a barra rápida.
- Incluir um botão "Limpar" contextual visível quando houver filtros ativos, evitando múltiplos cliques em chips e garantindo previsibilidade.

## 4) Dados e tabelas
- Padronizar as colunas ocultadas no mobile em um mapeamento único (ex.: dicionário de `colunas_por_viewport`) e reutilizar tanto no `st.data_editor` quanto no CSS, facilitando ajustes e A/B tests.
- Destacar a densidade da tabela: oferecer opção de compactação (linha alta vs. densa) para usuários avançados, controlada via toggle na barra superior e propagada para a função `render_glassy_table`.

## 5) Tema, acessibilidade e microinterações
- Anunciar mudanças de tema apenas quando o toggle for usado (hoje a mensagem aria-live é emitida a cada render). Mover o anúncio para dentro do callback do toggle evita ruído de leitores de tela.
- Ajustar foco visível e `aria-label` do toggle de modo escuro e dos chips de exportação/viewport para descrever ação e estado (ex.: "Alternar modo escuro: ativo").
- Revisar contrastes dos estados ghost/secondary no dark mode para ultrapassar 4.5:1, especialmente em chips (`.pg-chip.ghost`) e tabs (`.pg-tab`).

## 6) Refatoração e modularização
- Extrair o CSS longo de `styles.inject_custom_css` para arquivos estáticos ou usar templates menores por bloco (header, filtros, tabela). Isso facilita cache, leitura e edição isolada de tokens.
- Criar um módulo dedicado a estado/constantes (ex.: `state.py`) com funções `get_filters_state()` e `reset_filters()`, alinhando `app.py` e `ui_components.py` e removendo `st.session_state.setdefault` dispersos.
- Encapsular a sincronização de viewport (`detect_viewport_width`) num componente Streamlit custom ou wrapper reutilizável (incluir debounce configurável e listener de orientation change), habilitando testes unitários e reuso em outras páginas.
