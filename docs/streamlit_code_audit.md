# Auditoria técnica do dashboard Streamlit

## Resumo da rodada atual

- A base evoluiu em **performance** (vetorização de cálculos de acerto), **arquitetura** (serviços para filtros/insights) e **testabilidade** (novos testes unitários).
- Nesta rodada foi adicionada a análise solicitada de **ranking de performance por mercado x modelo**, com tabela dedicada por mercado e ordenação decrescente por acerto.
- O recorte de dados segue limitado para janela recente (2 meses por configuração de `months_back`) para reduzir custo de carregamento.

## Achados já endereçados

1. Bug de variável indefinida em seleção de campeonatos (`tourn_opts`).
2. Ajustes de consistência entre recorte exibido e exportado em finalizados.
3. Remoção/redução de blocos pesados de HTML/JS inline com migração para componentes nativos (Altair/Streamlit).
4. Redução relevante de `apply(axis=1)` em análises centrais via `analysis_service.compute_hit_columns`.
5. Ranking de modelo por mercado com volume avaliado (`Total de Jogos Avaliados`) e ordenação de acurácia.

## Pontos que ainda merecem implementação/fortalecimento

### 1) Persistência cross-session como requisito explícito de produto

- Hoje há persistência local best-effort para filtros, mas falta política clara de produto (escopo por usuário, ambiente compartilhado, expiração, versionamento de schema e governança de migrações).
- Próximo passo: definir contrato de persistência (arquivo local x backend/DB) com TTL e estratégia de fallback.

### 2) Cobertura de testes para UI crítica e fluxo de filtros

- Há boa cobertura unitária de serviços, mas ainda faltam testes de integração dos fluxos principais de dashboard (filtros + render + export em recortes específicos).
- Próximo passo: adicionar smoke/integration tests para cenários de borda (dataset vazio, colunas ausentes, datas inválidas, torneios mistos).

### 3) Segurança operacional mantendo `verify=False` (requisito)

- `verify=False` deve permanecer por requisito atual.
- Mitigações necessárias em produção: allowlist rígida, checksum obrigatório, observabilidade de download e alerta de alteração inesperada do arquivo de origem.

## Recomendações práticas de continuidade

1. Definir e documentar estratégia oficial de persistência de filtros (cross-session).
2. Aumentar testes de regressão orientados a dados reais de produção.
3. Instrumentar métricas de performance (tempo de carga, tempo de filtros, tamanho do dataframe pós-ingestão).
4. Revisar periodicamente colunas obrigatórias/opcionais com validação de schema antes da camada de visualização.
