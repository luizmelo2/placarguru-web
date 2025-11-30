# Revisão de UI/UX — Placar Guru

## 1. Identidade visual e proposta
- O topo usa um logotipo em gradiente e um hero fixo, mas não há orientação de marca para cores secundárias, tipografia ou uso de ícones. Recomendo criar um mini design system (paleta, espaçamentos, componentes base) para manter consistência entre a página principal e os relatórios/exportações. Além disso, considere reduzir o brilho do gradiente da logo e do fundo da topbar para evitar competir com os KPIs e tabelas.【F:styles.py†L13-L155】【F:app.py†L82-L126】
- A experiência alterna entre tons neon (verde/azul) e cards “glassmorphism” claros e escuros. Unificar a hierarquia cromática (por exemplo, manter primário em azul, secundário em turquesa e destacar alertas/estados com cores neutras) deixará o visual mais limpo e legível.【F:styles.py†L13-L155】

## 2. Layout e modernidade
- O layout já adota grade ampla, cards de KPI e sticky topbar, mas depende de um toggle manual “📱 Mobile” para liberar a lista ou a grade. Isso cria fricção: recomendo detectar largura via CSS/JS ou `st.columns` responsivos e eliminar o controle manual, exibindo automaticamente a visão de lista quando a largura for menor que ~960px.【F:app.py†L114-L132】【F:app.py†L199-L204】
- A topbar usa um padrão de três colunas, mas fica visualmente carregada em telas menores. Avalie mover ações secundárias (ex.: chips informativos) para um drawer/toolbar compacta no mobile e usar uma “headline section” mais leve (logo pequena + título + botão de tema) para dar respiro aos filtros logo abaixo.【F:styles.py†L63-L159】【F:app.py†L82-L126】
- Os cards de KPI e hero ocupam boa parte da dobra inicial, empurrando filtros e tabelas para baixo. Uma versão condensada (KPI em linha com ícones e números grandes, sem subtítulo longo) ajuda a priorizar lista e gráficos. Considere usar tooltips em vez de subtextos fixos para economizar altura.【F:app.py†L234-L278】

## 3. Bugs e oportunidades de simplificação
- A aplicação força visibilidade do header e do menu lateral com CSS customizado; em algumas versões do Streamlit isso pode gerar z-index conflitante com elementos fixos ou esconder mensagens de erro. Só injete esse patch quando detectar a necessidade (feature flag) e teste em dark/light para evitar sobreposição de sombras.【F:app.py†L35-L71】【F:styles.py†L63-L159】
- O default do modo mobile está como `True`, o que pode manter usuários de desktop presos à lista mesmo em telas largas. Sugiro usar `st.form_state` ou cookies para lembrar a última escolha e iniciar em desktop quando a largura reportada pelo navegador for grande.【F:app.py†L114-L132】
- Há repetição de cores e sombras diretamente no CSS, o que dificulta ajustes de contraste. Centralize tokens (cores, raios, sombras) em variáveis CSS já declaradas e referencie-as nos componentes; isso facilita passar em avaliações de acessibilidade e permite um toggle de alto contraste no futuro.【F:styles.py†L13-L155】【F:styles.py†L161-L200】

## 4. Próximos passos sugeridos
- Criar uma barra de ações flutuante no mobile com filtros essenciais (campeonato, intervalo de datas, busca por time) e botões de exportação, evitando scroll longo até o sidebar.
- Adicionar indicadores de status diretamente nas listas (ex.: bolinha verde/laranja/vermelha) e melhorar affordance dos botões de ocultar/exibir jogos finalizados.
- Implementar testes visuais simples (screenshots) para validar consistência entre modo claro/escuro e mobile/desktop a cada release.
