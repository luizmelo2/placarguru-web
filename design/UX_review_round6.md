# Revisão de Design (Rodada 6)

Avaliação considerando o estado atual (estado unificado, tabela com densidade e header compacto). Itens priorizados por impacto em clareza, consistência e manutenibilidade.

## 1) Hierarquia e narrativa
- Reforçar a hierarquia do topo adicionando um resumo rápido de desempenho (acertos recentes/mercado) abaixo dos breadcrumbs, reutilizando um mesmo componente para desktop e mobile. Isso contextualiza antes dos filtros e reduz a sensação de vazio quando não há export pronto.
- Nos chips de status (modo, export, filtro), exibir ícone + label curto e consolidar feedback de sucesso/erro de exportação no mesmo container para evitar “pulos” no layout.

## 2) Layout responsivo
- Harmonizar o breakpoint móvel entre CSS e Python em 1024px (único valor) e documentar no `styles.py` e `state.py`. Reduz estados intermediários e evita re-render duplo no data editor.
- Em mobile, mover ações secundárias (PDF/CSV e densidade) para um menu sheet ou linha de ícones comprimidos, mantendo apenas busca e filtros como ações primárias visíveis.

## 3) Fluxo de busca e filtros
- Unificar o valor de busca e chips rápidos em um único campo do `session_state` (ex.: `filter_state.search`). Evita divergência entre barra lateral e toolbar móvel após resets parciais.
- Adicionar indicador de filtros ativos diretamente no botão “Limpar” (ex.: badge numérico) para facilitar descoberta em telas densas.
- Persistir as seleções principais (torneio/mercado) entre sessões via `st.cache_data` ou arquivo local leve, com opção de reset. Ajuda usuários recorrentes sem reintroduzir query params.

## 4) Dados e tabelas
- Consolidar presets de colunas por viewport/densidade em um único dict exportado de `state.py` e usado tanto pelo `st.data_editor` quanto por qualquer lógica de transformação. Diminui divergência entre esconder coluna e remover dado.
- Adicionar microcopy de legenda para os highlights Guru diretamente na tabela (tooltip ou linha abaixo do título) e um modo compacto sem hover-depender no mobile para acessibilidade.
- Considerar “sticky” para colunas chave (data/mandante/visitante) no desktop para manter contexto durante scroll horizontal.

## 5) Tema, acessibilidade e microinterações
- Restringir o anúncio de tema (aria-live) apenas quando o toggle é acionado pelo usuário e incluir o estado atual no label ("Alternar tema — claro"). Evita anúncios repetitivos a cada rerender.
- Revisar contraste das variantes ghost/secondary no dark mode para atingir 4.5:1, especialmente em chips e tabs. Documentar tokens de cor de superfície/linha em `styles.py`.
- Incluir foco visível consistente em todos os chips/botões de filtro e no data editor (célula ativa), alinhando espessura/radius ao restante dos tokens.

## 6) Refatoração e modularização
- Extrair CSS de componentes grandes (header, toolbar, tabela) para strings dedicadas ou arquivos estáticos em `styles/`, mantendo apenas injeções curtas em `inject_custom_css`. Facilita testes visuais e troca de temas.
- Criar uma função utilitária para registrar e limpar listeners de viewport (debounce + orientation) evitando duplicação do script em múltiplos pontos do app.
- Padronizar helpers de construção de chips/badges em `ui_components.py` (ex.: `render_status_chip`, `render_action_chip`) para reduzir repetição e manter iconografia coerente entre desktop e mobile.
