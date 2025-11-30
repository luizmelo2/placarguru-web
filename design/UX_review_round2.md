# Revisão de UI/UX — Rodada 2

## 1. Identidade visual e hierarquia
- O topo e a hero usam gradientes fortes e glass no logotipo/barra, mas não há aplicação consistente nos demais componentes (tabelas, badges, filtros). Sugiro reduzir os overlays radiais na topbar e reutilizar o mesmo gradiente apenas em ícones ou CTAs para equilibrar foco nos dados.【F:styles.py†L73-L169】【F:app.py†L86-L112】
- Faltam tokens para tipografia e espaçamentos; os tamanhos de fonte dos KPIs e chips são hardcoded no CSS. Centralizar pesos/tamanhos em variáveis (ex.: `--font-xs`, `--font-sm`, `--radius-md`) permitiria ajustes rápidos e um toggle de acessibilidade (fonte grande).【F:styles.py†L54-L169】
- As badges de tema e chips informativos usam borda `--stroke` e texto padrão, mas não há estados de foco/teclado. Adicionar `outline`/`box-shadow` em `:focus-visible` melhora acessibilidade sem alterar o layout.【F:styles.py†L149-L169】

## 2. Responsividade e modos de visualização
- A detecção de viewport grava `vw` na query string a cada resize, forçando recálculo do layout. Isso pode gerar loops de rerender e poluir históricos compartilhados. Recomendo usar `st_javascript` ou um componente de sessão para sincronizar largura via `SessionState`/`on_event` sem mexer na URL.【F:app.py†L118-L160】
- Mesmo com `modo_mobile` automático, a lista ainda depende de um checkbox manual para desktop. Unificar a regra (ex.: `use_list_view = modo_mobile or user_pref`) e persistir a preferência em `st.session_state` evita saltos visuais ao redimensionar.【F:app.py†L244-L245】
- A barra fixa e a subhead ocupam altura considerável; em telas <960px faltam controles de filtros rápidos. Uma toolbar compacta (torneio, período, busca) poderia aparecer abaixo da topbar somente no mobile, reduzindo scroll até o sidebar.【F:styles.py†L73-L200】【F:ui_components.py†L156-L200】

## 3. Filtros e formulários
- O bloco de filtros é estilizado (`pg-filter-shell`), mas o sidebar padrão do Streamlit continua aberto por padrão, gerando redundância com os controles na página. Avalie mover filtros críticos para o topo da seção central e deixar o sidebar apenas para downloads/ajustes avançados.【F:styles.py†L171-L200】【F:app.py†L237-L245】
- Multiselects para equipes e torneios carregam todos os itens por padrão no desktop; isso dificulta entender o recorte atual. Exibir chips/resumo de seleção e limites de altura nos dropdowns melhora legibilidade e evita scroll longo.【F:ui_components.py†L81-L107】【F:ui_components.py†L156-L200】
- O filtro de período tem botões rápidos e um `date_input` no mesmo expander. Adicionar feedback textual (intervalo ativo) e presets alinhados com o dashboard (ex.: “Últimos finalizados”) cria coerência com a lógica de recorte automático na listagem.【F:ui_components.py†L109-L134】【F:app.py†L366-L390】

## 4. Apresentação de dados e cartões
- A hero ocupa largura total com título, meta e quatro KPIs sem rótulos de status/recorte; quando a lista está vazia, ela permanece visível e cria espaço morto. Condicionar a hero ao `curr_df` não vazio e adicionar subtítulo com filtros aplicados melhora contexto. Também considere transformar os KPIs em `st.metric` ou componentes reutilizáveis para manter responsividade nativa.【F:app.py†L287-L365】
- A tabela “glassy” depende de `st.dataframe`, que adiciona scroll interno e títulos duplicados. Para listas mais curtas (mobile), `st.data_editor` ou HTML simples renderizado com `st.markdown` evita barras internas e permite colunas mais densas.【F:ui_components.py†L22-L49】
- Destaques “Sugestão Guru” são calculados no backend, mas não há sinalização visual na tabela/lista além do counter no KPI. Adicionar uma coluna com ícone ou cor de linha para linhas destacadas melhora descobribilidade.【F:app.py†L310-L359】【F:ui_components.py†L52-L75】

## 5. Temas, contraste e estado
- Tokens de cor já existem, mas sombras, bordas e blur são repetidos em vários blocos (`topbar`, `filter`, `table-card`). Centralizar `--shadow-card`, `--shadow-strong`, `--blur-bg` e reaproveitar reduz inconsistências entre dark/light e facilita atender WCAG de contraste de borda.【F:styles.py†L13-L200】
- Não há mecanismo de alternar tema além do estado inicial; o badge “Tema: Dark/Light” é informativo. Inserir um switch real na topbar e persistir via `st.session_state` permitiria testes de contraste e alinharia UI com a indicação textual.【F:app.py†L333-L338】【F:styles.py†L73-L169】
- O patch opcional de header/sidebar não aplica fallback visual no dark mode (background transparente). Se habilitado, sombras e bordas podem sobrepor o glass da topbar. Limitar o patch a contextos detectados (por exemplo, tema claro + layout wide) e revisar `z-index` evita conflitos.【F:app.py†L42-L75】【F:styles.py†L73-L169】

## 6. Acessibilidade e microinterações
- Falta texto alternativo na logo (usa `aria-hidden`) e não há foco navegável nos chips/tabs. Incluir `aria-label` e `role="button"` nos controles customizados aumenta suporte a teclado/leitores.【F:app.py†L86-L112】【F:styles.py†L127-L169】
- Os botões de download/exportação aparecem apenas em alguns estados (agendados). Tornar a ação de exportar sempre visível, mas desabilitada com tooltip quando sem dados, reduz confusão sobre onde salvar relatórios.【F:app.py†L396-L407】
- O script JS não trata debounces longos; em dispositivos lentos pode travar a rolagem durante resize. Encapsular o listener em `requestAnimationFrame` ou limitar a frequência evita jank na UI.【F:app.py†L118-L160】

## 7. Próximos passos recomendados
- Criar um mini design system (tokens + componentes básicos) e mover o CSS inline para um arquivo versionado, permitindo lint/validação de acessibilidade.
- Adicionar testes de screenshot (desktop/mobile, dark/light) para validar regressões visuais ao alterar os tokens.
- Introduzir modo compacto para tabelas/lista no mobile com colunas essenciais e CTA de detalhe por linha.
