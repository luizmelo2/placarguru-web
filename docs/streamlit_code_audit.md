# Auditoria técnica do dashboard Streamlit

Este documento resume pontos críticos encontrados na revisão técnica do código da aplicação Streamlit.

## Achados críticos

1. **Variável indefinida em filtros de campeonatos** (`tourn_opts`) pode quebrar em runtime no clique de "Selecionar Todos".
2. **Leitura HTTP com `verify=False`** desabilita validação TLS e reduz confiabilidade/segurança.
3. **Injeção de scripts externos via CDN dentro de `components.html`** aumenta risco operacional e de supply-chain.

## Achados de alta prioridade

- Filtro de data inclui linhas com data `NaT` mesmo quando o usuário define intervalo.
- Seleção de status "Finalizados" pode exportar dataset diferente do exibido devido ao recorte automático de 3 dias aplicado após `curr_df`.
- Vários `apply(axis=1)` em toda a base degradam performance com aumento de volume.

## Recomendações

- Corrigir erro de variável e cobrir com teste de regressão.
- **Manter `verify=False` (requisito atual)**, mas compensar com controles: timeout/retry, validação de checksum/assinatura do arquivo baixado, allowlist de domínio e monitoramento de integridade.
- Separar pipeline: ingestão/validação/transformação/renderização.
- Introduzir validações de schema e tratamento defensivo para colunas opcionais.
- Reduzir HTML `unsafe_allow_html` e scripts inline quando possível.
