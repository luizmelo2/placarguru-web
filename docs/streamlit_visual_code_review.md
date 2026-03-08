# Auditoria técnica do código Streamlit (visualização de previsões)

## Escopo analisado
- `app.py`
- `ui_components.py`
- `dashboard_service.py`
- `state.py`
- `analysis.py`
- `analysis_service.py`
- `utils.py`

---

## A) Resumo executivo

A base está funcional e tem bons sinais de maturidade (organização por módulos, tipagem parcial, utilitários de avaliação vetorizada e cobertura de testes existente). Porém, há **riscos relevantes de confiabilidade** na camada de visualização/estado: existe bug crítico de variável não inicializada quando o recorte retorna vazio, distorção silenciosa de KPIs quando o filtro Guru está ativo e alguns pontos de UX/estado que geram comportamento inconsistente. Em segurança, há uso extensivo de `unsafe_allow_html=True` e download HTTP com `verify=False` que merece endurecimento.

**Pontos fortes**:
- separação inicial entre serviços (`dashboard_service`, `analysis_service`, `ui_components`);
- validações básicas de colunas em vários pontos;
- cálculo vetorizado de acurácia em `compute_hit_columns`.

**Principais riscos**:
- quebra em runtime por variáveis fora de escopo;
- métricas/relatórios podendo sair incorretos em filtros específicos;
- possível XSS/HTML injection via dados externos;
- recomputações pesadas a cada interação no Streamlit.

---

## B) Problemas encontrados

### 1) `curr_df/curr_label/export_disabled` podem ficar indefinidos
- **Severidade:** crítica
- **Onde:** `app.py`, bloco de exportação após `if df_filtered.empty`.
- **Por que é problema:** quando `df_filtered.empty == True`, o código só mostra warning e segue para `generate_pdf_report(curr_df)`, mas `curr_df`/`curr_label`/`export_disabled` não foram definidos.
- **Impacto real:** exceção em runtime ao aplicar filtros sem resultados.
- **Como corrigir:** inicializar valores padrão antes do `if df_filtered.empty` ou mover totalmente a área de exportação para dentro do branch onde as variáveis existem.

### 2) Filtro `guru_only` altera o conteúdo do DataFrame e distorce KPIs
- **Severidade:** alta
- **Onde:** `dashboard_service.py`, `apply_dashboard_filters`.
- **Por que é problema:** com `guru_only=True`, o código zera (`pd.NA`) sugestões de mercados não destacados dentro dos jogos filtrados.
- **Impacto real:** métricas de acerto, tabelas, cards e PDF podem ser calculados com dados amputados, produzindo resultado silenciosamente incorreto.
- **Como corrigir:** não sobrescrever colunas de previsão; aplicar o filtro apenas no subconjunto de partidas e manter valores originais.

### 3) Possível `UnboundLocalError` no fluxo de exportação em recorte vazio
- **Severidade:** crítica
- **Onde:** `app.py`, mesma área do item 1.
- **Por que é problema:** além de `curr_df`, o botão usa `curr_label` e `export_disabled` também não inicializados no branch vazio.
- **Impacto real:** app quebra justamente no caso de borda mais comum (filtro sem resultados).
- **Como corrigir:** inicializar as três variáveis antes do bloco condicional.

### 4) Construção de score final em cards pode quebrar com `result_away` nulo
- **Severidade:** alta
- **Onde:** `ui_components.py`, `_prepare_display_data`.
- **Por que é problema:** `final_score` testa `result_home` apenas; se `result_home` existe e `result_away` é NaN, `int(result_away)` levanta erro.
- **Impacto real:** renderização de cards pode falhar em partidas com dado parcial.
- **Como corrigir:** validar ambos (`result_home` e `result_away`) antes da conversão para `int`.

### 5) Contagem de filtros ativos é incorreta para “selecionado tudo”
- **Severidade:** média
- **Onde:** `state.py` (`FilterState.active_count`) e duplicação de lógica em `app.py`.
- **Por que é problema:** `active_count` soma filtro de torneio/modelo só por lista não vazia, mesmo quando todos os valores estão selecionados.
- **Impacto real:** UX enganosa (“filtros ativos” inflado), decisões erradas de usuário.
- **Como corrigir:** comparar com cardinalidade total de opções (como já é parcialmente feito em `app.py`) e centralizar cálculo em um único lugar.

### 6) `st.session_state["pg_table_density"]` é resetado a cada rerun
- **Severidade:** média
- **Onde:** `app.py`.
- **Por que é problema:** estado é reatribuído incondicionalmente para `DEFAULT_TABLE_DENSITY` em toda interação.
- **Impacto real:** preferência do usuário nunca persiste; comportamento parece “quebrado”.
- **Como corrigir:** usar `setdefault` + controle explícito via widget/ações.

### 7) Variáveis/imports mortos e lógica incompleta
- **Severidade:** baixa
- **Onde:** `app.py` (`etag`, `action_col`, `real_code`, `render_chip` importado sem uso), tema (`pg_dark_mode` inicializado mas ignorado com `dark_mode=False`).
- **Por que é problema:** ruído, confusão arquitetural e risco de manutenção.
- **Impacto real:** dívida técnica, leitura mais difícil, bugs por suposições erradas.
- **Como corrigir:** remover código morto e reconectar tema a `session_state`.

### 8) `str.contains` com regex implícito no filtro por time
- **Severidade:** média
- **Onde:** `dashboard_service.py`, `build_filter_mask`.
- **Por que é problema:** texto de busca vira regex; caracteres especiais (`+`, `(`, `[`) podem causar match inesperado ou `re.error`.
- **Impacto real:** filtro retorna jogos errados ou quebra com entradas específicas.
- **Como corrigir:** usar `regex=False` ou `re.escape(q)`.

### 9) Risco de XSS/HTML injection em blocos com `unsafe_allow_html=True`
- **Severidade:** alta
- **Onde:** principalmente `ui_components.py` (`display_list_view`, `render_app_header`, detalhes HTML).
- **Por que é problema:** parte dos textos vem de dados externos (times/modelo/campeonato) e entra em HTML customizado.
- **Impacto real:** risco de script/style injection em ambiente com dados não confiáveis.
- **Como corrigir:** escapar sistematicamente (`html.escape`) em campos textuais antes de interpolar HTML; restringir uso de HTML bruto.

### 10) Download da release com TLS sem verificação
- **Severidade:** alta
- **Onde:** `utils.py`, `fetch_release_file`.
- **Por que é problema:** `requests.get(..., verify=False)` reduz segurança de transporte.
- **Impacto real:** risco MITM e ingestão de arquivo adulterado.
- **Como corrigir:** habilitar verificação TLS (`verify=True`) e manter checksum/allowlist como defesa adicional.

### 11) Reprocessamento redundante de métricas em cada rerun
- **Severidade:** média
- **Onde:** `app.py` + `analysis.py`.
- **Por que é problema:** para o mesmo `df_fin`, são feitos múltiplos cálculos derivados (`calculate_kpis`, `prepare_accuracy_chart_data`, `get_best_model_by_market`, `build_model_ranking_by_market`), vários chamando `compute_hit_columns` de novo.
- **Impacto real:** latência maior, UI menos responsiva.
- **Como corrigir:** calcular `df_eval = compute_hit_columns(df_fin)` uma vez e injetar nos serviços; considerar `st.cache_data` para saídas pesadas.

### 12) Persistência de filtros em arquivo local sem isolamento de usuário
- **Severidade:** possível problema (média)
- **Onde:** `state.py` (`~/.placarguru_filters.json`).
- **Por que é problema:** em deploy multiusuário, filtro de um usuário pode “vazar” para outro.
- **Impacto real:** comportamento inconsistente e risco de privacidade operacional.
- **Como corrigir:** persistir por sessão/usuário (chave por `st.session_state` + auth id) ou desabilitar persistência em servidor compartilhado.

---

## C) Melhorias recomendadas

1. **Confiabilidade primeiro:** tratar fluxo sem dados como caminho principal (estado default + guard clauses).
2. **Separação de camadas:** consolidar pipeline em: ingestão -> normalização -> feature/eval -> apresentação.
3. **`DashboardViewModel`:** criar objeto único com `df_filtered`, `df_ag`, `df_fin`, `df_eval`, KPIs e rankings cacheados.
4. **Estado de filtros unificado:** remover cálculo duplicado de filtros ativos; usar apenas `FilterState` com metadados de opções.
5. **Sanitização de HTML:** helper central de escape para campos textuais.
6. **Performance:** memoizar transformações pesadas por hash de `df_fin` + filtros.
7. **UX de filtros:** agrupar filtros primários em `st.form` para reduzir reruns a cada clique.
8. **Exportação robusta:** gerar PDF sob demanda com spinner + fallback de erro amigável.
9. **Validações defensivas:** checar intervalos de probabilidade (0..1), duplicidade de jogos por chave composta e coerência odds/prob.
10. **Observabilidade:** adicionar logs estruturados para recorte aplicado, volume de jogos e tempo por etapa.

---

## D) Refatorações sugeridas

### D.1 Estruturar o fluxo principal
```python
# app.py (ideia)
state = load_dashboard_state(df)
view = build_view_model(df, state)   # aplica filtros + separa agendados/finalizados + eval
render_header(view)
render_filters_feedback(view)
render_main_table_or_cards(view)
render_kpis_and_charts(view)
render_export(view)
```

### D.2 Evitar recomputar hits
```python
# analysis_facade.py
@st.cache_data(show_spinner=False)
def compute_all_views(df_fin: pd.DataFrame) -> dict:
    df_eval = compute_hit_columns(df_fin)
    return {
        "kpis_multi": calculate_kpis_from_eval(df_eval, multi_model=True),
        "kpis_overall": calculate_kpis_from_eval(df_eval, multi_model=False),
        "daily": prepare_accuracy_from_eval(df_eval),
        "best_model": best_model_from_eval(df_eval),
        "ranking": ranking_from_eval(df_eval),
    }
```

### D.3 Corrigir exportação sem dados
```python
curr_df = pd.DataFrame()
curr_label = "recorte"
export_disabled = True

if not df_filtered.empty:
    ...
    curr_df = df_ag if status_view.startswith("🗓️") else df_fin
    curr_label = "Agendados" if status_view.startswith("🗓️") else "Finalizados"
    export_disabled = curr_df.empty
```

### D.4 Filtro de busca sem regex acidental
```python
home_contains = df["home"].astype(str).str.contains(q, case=False, na=False, regex=False)
away_contains = df["away"].astype(str).str.contains(q, case=False, na=False, regex=False)
```

---

## E) Código corrigido (trechos)

### E.1 Não mutilar colunas ao aplicar `guru_only`
```python
# dashboard_service.py
if params.guru_only:
    mask &= guru_flag_all

# manter previsões originais; apenas anotar flags
df_filtered = df.loc[mask].assign(
    guru_highlight_scope=guru_scope_all.loc[mask],
    guru_highlight=guru_flag_all.loc[mask],
)
```

### E.2 Corrigir `final_score` defensivamente
```python
rh = row.get("result_home")
ra = row.get("result_away")
final_score = "—"
if pd.notna(rh) and pd.notna(ra):
    final_score = f"{int(rh)}-{int(ra)}"
```

### E.3 Persistência de densidade
```python
st.session_state.setdefault("pg_table_density", DEFAULT_TABLE_DENSITY)
table_density = st.session_state["pg_table_density"]
```

---

## Lista priorizada do que corrigir primeiro

1. Corrigir fluxo de variáveis não inicializadas no export (quebra em runtime).
2. Remover mutação de colunas em `guru_only` (erro silencioso de métricas).
3. Sanear HTML interpolado com `unsafe_allow_html=True`.
4. Corrigir `final_score` quando placar parcial/NaN.
5. Ajustar filtro textual para `regex=False`.
6. Eliminar recomputação de `compute_hit_columns` em cadeia.
7. Corrigir contagem de filtros ativos e feedback visual.
8. Remover reset de `pg_table_density` e outros estados que “não pegam”.
9. Endurecer `fetch_release_file` (TLS verify + monitoramento).
10. Limpar variáveis/imports mortos e reduzir acoplamento do `app.py`.

---

## Plano de refatoração em etapas

### Etapa 1 — Estabilização (rápida)
- Guard clauses de “sem dados”.
- Fix de `curr_df/curr_label/export_disabled`.
- Fix de `final_score` + `regex=False`.

### Etapa 2 — Integridade dos dados
- Remover mutação em `guru_only`.
- Adicionar validações de coerência (probabilidade, odds, duplicados por jogo/modelo).

### Etapa 3 — Performance
- Criar camada cacheada para `df_eval` e derivados.
- Reduzir recomputações em gráficos/tabelas.

### Etapa 4 — UX/arquitetura
- Introduzir `DashboardViewModel`.
- Organizar filtros em `st.form` e feedback consistente de estado ativo.

### Etapa 5 — Segurança e produção
- Sanitização sistemática de HTML.
- Revisar `verify=False` e políticas de secrets/inputs.

---

## Top 10 melhorias de maior impacto (versão resumida)

1. Corrigir crash de export quando filtro retorna vazio.
2. Parar de zerar previsões com `guru_only`.
3. Sanitizar todas as strings interpoladas em HTML.
4. Validar placar final com ambos os lados não nulos.
5. Trocar `contains` para busca literal (`regex=False`).
6. Calcular `compute_hit_columns` uma única vez por recorte.
7. Centralizar cálculo de filtros ativos.
8. Preservar estado de densidade/toggles sem reset em rerun.
9. Endurecer download da release (TLS + checksum obrigatório em produção).
10. Quebrar `app.py` em funções pequenas por responsabilidade.
