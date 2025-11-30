# Revisão de UI/UX — Rodada 4

## 1. Identidade visual e hierarquia
- A topbar continua ocupando grande área com três colunas de chips e breadcrumbs repetidos; em telas menores ela vira uma grade alta antes do conteúdo principal. Simplificar para duas faixas (brand + status) e mover estado de exportação/filtros para tooltips reduziria distração visual.【F:app.py†L407-L449】【F:styles.py†L90-L184】
- Hero e breadcrumb duplicam o mesmo resumo de filtros, e os chips "Tema"/"Exportação" aparecem tanto na topbar quanto na hero/metadados. Consolidar o resumo em um único local e priorizar CTA (exportar/ajustar filtros) melhora a hierarquia primária.【F:app.py†L449-L499】
- Os efeitos glassy/gradientes da logo, hero e cartões persistem fortes no dark mode, gerando baixa distinção entre fundo e cartões. Diminuir sombras e usar contornos mais finos no tema escuro aumentaria contraste real de texto e bordas.【F:styles.py†L100-L179】【F:styles.py†L215-L240】

## 2. Responsividade e layout
- O indicador "Visual automático" ocupa uma linha completa logo após o toggle de tema, empurrando o conteúdo para baixo em viewports médios. Substituir por um chip compacto próximo ao título ou dentro da topbar evita espaçamento vertical extra.【F:app.py†L77-L149】
- A toolbar móvel replica apenas torneio/período, mas não espelha modelos/odds ou favoritos; além disso, o texto "Ativos" não é clicável para reabrir o sidebar. Considerar chips interativos que abrem o filtro correspondente e um atalho para colapsar/expandir o menu lateral melhora fluxo mobile.【F:app.py†L248-L313】【F:ui_components.py†L218-L239】
- A seção de ocultar lista de finalizados fica acima dos próprios dados e dos KPIs, criando scroll adicional para chegar às métricas. Mover o toggle para a hero ou para a barra de status evita um bloco inteiro repetido em cada renderização.【F:app.py†L551-L602】

## 3. Filtros e fluxo de uso
- Mesmo com sidebar colapsado por padrão, os multiselects de equipes e sugestões seguem longos e sem pré-filtros. Adicionar chips-resumo no corpo principal e limitar altura dos dropdowns com rolagem ajudaria a percepção de seleção ativa.【F:ui_components.py†L94-L151】
- O atalho de busca rápida (`q_team`) é independente do texto de busca do sidebar, podendo divergir; um único estado compartilhado evitaria resultados diferentes entre mobile e desktop.【F:app.py†L248-L313】【F:ui_components.py†L99-L114】
- O controle de "Visão lista" permanece no corpo e na mensagem da seção de finalizados, mas não aparece na barra principal; manter o toggle junto ao status/contador na topbar deixaria o fluxo mais consistente.【F:app.py†L248-L259】【F:app.py†L551-L602】

## 4. Apresentação de dados
- A tabela `st.data_editor` usa apenas ícone e help para Sugestão Guru; faltam cores/legenda visual para linhas destacadas, o que reduziria esforço de escaneamento. Adicionar zebra, badges coloridos para status e uma legenda fixa abaixo do card tornaria o destaque mais claro.【F:ui_components.py†L22-L68】
- Colunas de odds e probabilidades ainda aparecem mesmo quando vazias, e a largura automática mantém colunas de resultado previstas espremidas em mobile. Introduzir presets de colunas essenciais por viewport e truncar métricas secundárias melhoraria legibilidade.【F:app.py†L538-L599】
- O KPI "Sugestão Guru" repete a regra no subtítulo, mas a tabela mostra somente um emoji; alinhar o texto da regra no caption ou adicionar tooltip persistente no cabeçalho mantém contexto para quem chega direto via âncoras.【F:app.py†L449-L481】【F:ui_components.py†L44-L68】

## 5. Tema, feedback e acessibilidade
- O anúncio de tema usa apenas texto invisível (`pg-sr`), mas os chips do topo não recebem `aria-live` ou foco claro; incorporar `role="status"` nos chips ativos e estados de foco mais contrastantes ajudaria leitores de tela e navegação por teclado.【F:app.py†L77-L149】【F:styles.py†L161-L179】
- A logo SVG ganhou `<title>`/`<desc>`, porém o contorno e o glow continuam com pouco contraste no dark mode; fornecer uma variante monocromática ou alternar stroke/fill pelo tema evita perda de forma para daltônicos.【F:app.py†L407-L426】【F:styles.py†L109-L134】
- A mensagem de desativação do PDF aparece somente após o hero e depende do estado `export_disabled`; exibir um tooltip ou aviso inline no próprio botão (estado disabled) oferece feedback imediato sem percorrer a página.【F:app.py†L488-L499】

## 6. Próximos passos sugeridos
- Consolidar topbar + hero em um header responsivo com CTA e resumo únicos, eliminando a duplicidade de chips e breadcrumbs em diferentes áreas.【F:app.py†L407-L485】
- Criar variantes compactas para toolbar mobile e painel de filtros, com chips clicáveis que reflitam todos os filtros principais e acionem o sidebar quando tocados.【F:app.py†L248-L313】【F:styles.py†L215-L240】
- Introduzir esquema de estados para tabelas (vazias, carregando, destaque Guru) com legenda fixa e colunas responsivas para odds/probabilidades, priorizando status e horários em telas estreitas.【F:ui_components.py†L22-L68】【F:app.py†L538-L599】
