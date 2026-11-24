# AvianScale CV

## What This Is

AvianScale CV é um sistema de visão computacional de baixo custo para estimativa não invasiva de peso vivo individual e médio de frangos de corte, combinando técnicas clássicas e interpretáveis de Processamento Digital de Imagens (PDI) com regressão tabular calibrada por parâmetros ópticos EXIF. Voltado a produtores integrados e agroindústrias avícolas, a **v1.0 MVP** entrega uma aplicação web interativa local single-origin para pitch acadêmico/produto, rigorosamente validada sobre toda a base limpa do dataset público *Broiler weights of Cobb Avian* (`1.738` imagens, `160` pesos únicos, `188` sessões), expondo em `< 100 ms` o passo a passo visual do PDI ("visão raio-X"), estimativa de peso com calibração óptica EXIF, separação interativa de aves encostadas via *Watershed*, dashboard científico de auditoria e roadmap arquitetural v2.

## Core Value

Estimar o peso vivo individual de frangos de corte de forma precisa, rigorosamente auditada e 100% interpretável usando apenas Processamento Digital de Imagens clássico calibrado por óptica EXIF — sem deep learning caixa-preta e com zero vazamento de dados do visor LCD da balança — exposto visualmente etapa por etapa em uma demonstração web local de alta credibilidade.

## Business Context

- **Customer**: Produtores integrados de frango de corte e agroindústrias avícolas (contexto do Paraná, maior produtor brasileiro, onde a pesagem manual por amostragem gera estresse animal, lentidão e baixa frequência de dados).
- **Revenue model**: MVP local de validação para pitch acadêmico/produto (evoluindo no roadmap conceitual v2 para monitoramento contínuo com câmera fixa vertical no teto do aviário).
- **Success metric**: Regressão validada em 5-fold `StratifiedGroupKFold` por `session_id` com ablação comprovando ganho da calibração óptica EXIF (`C2_physics_normalized_strict`: `MAE = 158,11 g`, `RMSE = 264,02 g`, `R² = 0,5582`, `R² por sessão = 0,6905`, desvio contrafactual `0,0%`), inferência local `< 500 ms` (atingido `56,1–79,7 ms` em `/api/predict`) e MVP concluído em `2026-11-24` (4 dias antes do freeze de `2026-11-28`, pronto para pitch em `2026-11-30`).
- **Strategy notes**: Especificação em [PRD.md](PRD.md), proposta acadêmica em [proposta_pdi.md.pdf](docs/proposta_pdi.md.pdf), auditoria forense completa em [DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md](docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md), síntese de ingestão em [.planning/intel/SYNTHESIS.md](.planning/intel/SYNTHESIS.md) e histórico de entregas em [.planning/MILESTONES.md](.planning/MILESTONES.md).

## Requirements

### Validated

- ✓ **Ingestor Forense Determinístico (REQ-01 / DATA-01):** Varre recursivamente `data/raw/Broiler weights of Cobb Avian/` aplicando todas as regras da Auditoria §9 (`1.738` limpas, `15` quarentena, `160` pesos únicos, `188` sessões, `367 → 267 g`, CP866/árabe/EXIF/Xiaomi `39321`) persistindo Parquets tipados e `models/dataset_audit_summary.json` — v1.0 (Phase 1)
- ✓ **Split Anti-Leakage Reprodutível (REQ-02 / DATA-02):** Agrupamento por `session_id` (gap `> 600 s`) e 5-fold `StratifiedGroupKFold` estratificado (`<500 g`, `500–1000 g`, `>1000 g`) com `0` sobreposição e `[4, 5, 5, 5, 5]` sessões pesadas por fold — v1.0 (Phase 1)
- ✓ **Pré-processamento Clássico EXIF-First em OpenCV (REQ-03 / PDI-01):** Loader único Pillow (`ImageOps.exif_transpose()`, `0` chamadas `cv2.imread`/`cv2.imdecode` no repo), downscale preservando aspecto (`working_long_edge = 768 px`, `PIL.Image.draft()`), Bilateral/Mediana e CLAHE no canal `L` (CIE Lab) sob `PDIConfig` congelado (`hash = 43a3cfaa3208019e`) — v1.0 (Phase 2)
- ✓ **Segmentação Anti-Leakage da Ave (REQ-04 / PDI-02):** Classificação de cena (`bucket_dorsal`: `377` vs. `scale_or_litter`: `1.361`), supressão invariante a rotação do LCD/prato SF-400 (`786` acionamentos, `0` falsos positivos no balde), segmentação clássica nos 3 cenários com apenas `4/1.738` (`0,23%`) de fallback (`0/377` no balde, incl. `0/59` aves `>1.000 g` branco-sobre-branco) — v1.0 (Phase 2)
- ✓ **Descritores Morfométricos, Cromáticos e Calibração Óptica EXIF (REQ-05 / PDI-03):** `49` features estritamente dentro de `bird_mask == 255` (invariância comprovada `0,0`-diff em perturbação externa), calibração `area_optical_norm = area_px / zoom_multiple²` validada em `127` sessões dual-lens (`W/UW = 0,9638` em pares de balde, sem dupla contagem `f·zoom` e sem dimensões brutas), extração em lote (`--workers 4`) materializada em `data/processed/features_clean.parquet` (`1.738 × 60`) em `46,53 s` e `models/pdi_calibration_audit.json` — v1.0 (Phase 2)
- ✓ **Artefatos Visuais Intermediários "Raio-X" (REQ-06 / PDI-04):** 4 etapas a `640 px` (3× JPEG Q=82 + máscara PNG 1 canal, `p50 = 140,1 KB`, `max = 166,8 KB`) com contorno e elipse/eixos orientados corretamente, latência ponta a ponta em imagens nativas `4080×1840` de `p50 = 74,4 ms`, `p95 = 88,3 ms` (`< 500 ms`) — v1.0 (Phase 2)
- ✓ **Módulo Demonstrativo Watershed Multi-Aves (REQ-07 / PDI-05):** Pipeline clássico isolado (pré-supressão cromática de comedouro vermelho + supressão LCD + EDT exata + `h_maxima` recortado por bbox + `skimage.segmentation.watershed`) com `9/13` contagens exatas e `12/13` em ±1 (`MAE = 0,38`, `p50 = 131,8 ms`) nas 13 imagens multi-aves (`417`, `757`, `823`, `1520`) e suporte a uploads arbitrários retornando `49` features por instância — v1.0 (Phase 2)
- ✓ **Treinamento e Comparação de Regressores Tabulares (REQ-08 / MOD-01):** Comparação 5-fold OOF (`0` sobreposição de `session_id`) de 4 famílias clássicas (`RidgeLog`, `HuberLog`, `RandomForestLog`, `LightGBMLog`) com wrappers pickláveis (`AllometricLogTransformer`, `LogDuanRegressor`), ponderação intra-fold sessão×faixa limitada (`≤ 5×`), diagnóstico de *Duan smearing* e higiene estrita de features (`FEATURE_COLUMNS` apenas, zero `scene_is_bucket` ou `BANNED_FEATURE_COLUMNS`) — v1.0 (Phase 3)
- ✓ **Relatório de Validação Rigoroso, Ablação Óptica 4-Vias e Auditoria Contrafactual (REQ-09 / MOD-02):** `models/evaluation_report.json` bit-determinístico e sem vazamento de caminhos (SHA-256 `b8d21b2d...`) provando vitória unânime de `C2_physics_normalized_strict` sobre `C1_uncalibrated_pdi` (`LightGBMLog`: `MAE = 158,11 g` [`-6,16 g`], `RMSE = 264,02 g` [`-11,17 g`], `BalMAPE = 35,11%` [`-1,08 pp`], `R² = 0,5582` [`+0,0381`], `R² por sessão = 0,6905`), com desvio contrafactual exato de `0,0%` em ambos os modos enquanto `C3` e `C4` são desqualificados por confounding de lente — v1.0 (Phase 3)
- ✓ **Serialização Dual-Bundle e Catálogo de Presets OOF (REQ-10 / MOD-03):** `models/weight_regressor.joblib` (git-ignored, `3,18 MiB`, `production_pipeline` + 5 `fold_pipelines` `{0..4}`, `PDIConfig` reconstruível + hash `43a3cfaa3208019e`, carga fria `75,2 ms`, inferência unitária `2,4 ms` com `0` warnings), mapeador etário `age_stage.py` (`~7–29 d` / `116–2.093 g`) e `models/presets_catalog.json` (git-tracked, SHA-256 `3a0c7e50...`, 6 presets OOF de ave única cobrindo os 5 folds + 4 presets Watershed referenciados apenas por `image_id` opaco) — v1.0 (Phase 3)
- ✓ **API Backend FastAPI Local (REQ-11 / WEB-01):** Pacote modular `backend/app/api/` expondo `/api/health`, `/api/presets`, `/api/presets/{preset_id}/thumbnail`, `/api/predict`, `/api/watershed-demo` e `/api/metrics` com validação Pydantic v2, `lifespan` com `cv2.setNumThreads(2)`, `run_in_threadpool`, guardrails ASVS L1 (`15 MiB` / `40 MP` / magic Pillow / allowlist não-symlink), roteamento OOF honesto (`fold_pipelines[fold]`), modo single-origin sobre `frontend/dist/` preservando `404`/`405` JSON em `/api/*` e latência live de `56,1–79,7 ms` (`< 500 ms`) — v1.0 (Phase 4)
- ✓ **Estúdio de Inferência & "Raio-X" do PDI (REQ-12 / WEB-02):** Frontend React 19 + TypeScript 7 + Vite 8 + Tailwind CSS v4 (`@theme` clinical-light + quadros `#0F172A`, `pt-BR`) com upload drag-and-drop (EXIF automático, aviso de EXIF ausente e override manual de lente/zoom), visualização 2×2 + lightbox das 4 etapas do PDI, card de peso estimado/erro/estágio Cobb Avian 48, tabela expansível dos 39 atributos mask-gated e simulador local de lote com média/desvio/CV% e confirmação inline em 2 etapas — v1.0 (Phase 4)
- ✓ **Galeria de Presets Curados para Pitch (REQ-13 / WEB-03):** 6 presets OOF em 1 clique cobrindo a curva de crescimento (`180 g`, `576 g`, `1.555 g`, balde branco-sobre-branco `1.986 g`) e comparador Dual-Lens ao vivo (`541 g` Fold 0: Wide `383,22 g` vs. Ultrawide `384,16 g`, delta `0,94 g`) exibindo Ground Truth vs. Estimado OOF, erro absoluto e percentual — v1.0 (Phase 4)
- ✓ **Aba Demonstrativa Watershed & Visão de Futuro (REQ-14 / WEB-04):** Aba interativa `Separação Watershed` com 4 presets multi-aves (`417`, `757`, `823`, `1520`) + upload customizado, sliders reconciliados com os limites ASVS V5 de `backend/app/pdi/config.py`, `AbortController`, aviso de EXIF ausente sempre visível, tabela de instâncias com swatches e pesos ilustrativos + banner permanente `group_sum_only`, e aba estática `Visão Futura (v2)` (`PROIBIÇÃO 5`, zero telemetria/backend v2) — v1.0 (Phase 4)
- ✓ **Painel de Métricas e Auditoria (REQ-15 / WEB-05):** Dashboard científico `Auditoria & Métricas` com 5 KPIs executivos + contexto `R² > 0,90`, gráfico Recharts v3 `ScatterChart` filtrável (`1.738` imagens / `188` sessões, `actual_g`, diagonal `y=x`), `BarChart` de ablação óptica (`C1`–`C4`, `C2 = 0,0%` vs. `C3`/`C4` desqualificados), Top-12 atributos, erros estratificados, matriz de 16 modelos e funil forense `1.753 → 1.738` reconciliando a alegação `"1.772 / 45 dias"` do Mendeley e o hiato `801–1.216 g` — v1.0 (Phase 4)

### Active

*Candidatos pós-pitch para o próximo milestone (`v1.1` Hardening / `v2.0` Barn Deployment — a formalizar em `/gsd-new-milestone`):*

- [ ] **HARD-01:** Auditoria offline automatizada de rótulos via OCR do visor SF-400 exclusivamente em pesquisa/curadoria para detectar eventuais erros de digitação remanescentes de 10 g / 100 g além de `367 → 267 g`.
- [ ] **HARD-02:** Calibração geométrica clássica `mm/pixel` pela borda interna do balde plástico branco (Hough/contorno) em imagens dorsais do Cenário B.
- [ ] **HARD-03:** Quality gate clássico de nitidez/*motion blur* (variância do Laplaciano) sinalizando frames borrados por obturador lento antes da predição.
- [ ] **HARD-04:** Validação diagnóstica *Leave-One-Date-Out* quantificando extrapolação temporal sobre o hiato de 8 dias (`2022-04-03 → 2022-04-11`).
- [ ] **HARD-05:** Upload em lote de pasta e exportação CSV/PDF do relatório de peso médio e uniformidade (`CV%`) do lote.
- [ ] **BARN-01 / BARN-02 / BARN-03:** Evolução v2 para câmera fixa vertical no teto ($Z_{const}$ e área real em $\text{cm}^2$), expansão de base para fase final comercial (`30–45 dias`, `2,1–3,2 kg`) e empacotamento edge CPU-only (Raspberry Pi / câmera industrial).

### Out of Scope

- **Redes neurais end-to-end / Deep Learning na inferência principal (CNNs, YOLO, ViTs, SAM)** — Estritamente proibido (`PROIBIÇÃO 1`); o objetivo central da avaliação é demonstrar domínio técnico em Processamento Digital de Imagens clássico e interpretável.
- **OCR ou leitura de pixels fora da máscara da ave (ex.: visor LCD da SF-400)** — Estritamente proibido (`PROIBIÇÃO 2`); configura *data leakage* visual que destrói a validade científica e operacional do modelo.
- **Split aleatório por imagem (`train_test_split`)** — Estritamente proibido (`PROIBIÇÃO 3`); rajadas de ~9 fotos/sessão da mesma ave inflam artificialmente as métricas.
- **Versionamento do dataset bruto/processado ou listagem recursiva das 1.753 imagens em prompts** — Estritamente proibido (`PROIBIÇÃO 4`); 3,54 GiB fora do Git e higiene de contexto.
- **Rastreamento multi-frame de vídeo (*object tracking*), densidade populacional do galpão e alertas sanitários de aves paradas** — Fora de escopo do código v1 (`PROIBIÇÃO 5`); pertencem exclusivamente ao roadmap conceitual apresentado no pitch.

## Context

### Current State (v1.0 MVP Shipped — 2026-11-24)

- **Codebase entregue:** `26.989` linhas de código (`18.615` Python em `backend/{app,scripts,tests}` + `8.374` TypeScript/TSX/CSS em `frontend/src`), com backend modular (`backend/app/{ingestion,pdi,modeling,api}`), CLIs reprodutíveis (`build_manifest.py`, `extract_features.py`, `train_and_evaluate.py`), artefatos auditados em `models/{dataset_audit_summary,pdi_calibration_audit,evaluation_report,presets_catalog}.json` e bundle compilado `frontend/dist/` servido em origem única pelo FastAPI (`uv run uvicorn backend.app.api.main:app --host 127.0.0.1 --port 8000 --workers 1`).
- **Qualidade verificada:** **`272/272`** testes pytest (backend + contrato/latência/segurança ASVS L1) + `ruff` limpo + **`26/26`** testes Vitest (frontend) + `tsc` (dual-project) + Biome v2 + E2E single-origin sobre `frontend/dist/` + auditoria cross-phase `v1.0-MILESTONE-AUDIT.md` `passed` (`15/15` requisitos, `24/24` integrações, `6/6` fluxos E2E, `5/5` `PROIBIÇÕES`).
- **Dívida técnica menor (v1.1):** (1) Promover 4 helpers `_`-prefixed compartilhados entre pacotes (`_assert_no_proibicao4_leaks`, `_resolve_safe_image_path`, `_WATERSHED_GROUP_SUM_DISCLAIMER`, `_BASIN_PALETTE_BGR`) para exports públicos `__all__`; (2) Decidir manter ou aposentar `load_oriented_rgb` (`backend/app/pdi/io_exif.py:230`), hoje coberto apenas em testes unitários pois o runtime padronizou em `load_working_rgb`; (3) Monitorar `DeprecationWarning` upstream de `scikit-image/morphology/grayreconstruct.py:216` sob NumPy 2.5.

### Dataset Auditado (Fonte Única de Verdade)

- **Dataset:** `data/raw/Broiler weights of Cobb Avian/` contém `1.753` imagens JPEG RGB (`3,54 GiB`, `0` corrompidas, `0` duplicatas exatas, `0` GPS) em `168` pastas-folha (`165` raiz + `3` aninhadas em nível 2: `171/173`, `190/191`, `190/193`) e `166` pesos brutos (`116–2.093 g`), capturadas em `9` datas entre `2022-03-24` e `2022-04-15` (`~7` a `~29` dias de idade, linhagem Cobb Avian 48) com Xiaomi Redmi 10 (`21061119AG`) e balança digital SF-400.
- **Armadilhas Críticas Resolvidas na v1.0:**
  - *Encoding e Estrutura:* `1340_` sanitizado; `1.635` nomes com dígitos árabes (`٠-٩`); `21` arquivos em `547/` e `562/` com mojibake UTF-8→CP866 revertidos; EXIF `DateTime` corrompido reconstruído por nome ou `mtime - 14401 s`; `81,86%` com `Orientation != 1` corrigidos via `ImageOps.exif_transpose()`.
  - *Curadoria / Quarentena:* `13` imagens multi-aves (`417`, `757`, `823`, `1520`) e `2` erros grosseiros (`1351`, `1465`) em quarentena (`15` imgs); `367` (`6` imgs) corrigida para `267 g`. Base limpa: **`1.738` imagens**, **`160` pesos únicos**, **`188` sessões**.
  - *Ambiguidade Óptica e Viés Lente×Peso:* `74,5%` Wide (`4,25 mm`, `4080×1840`) vs. `25,5%` Ultrawide (`1,65 mm`, `3264×1472`), mais `67` imagens com zoom digital (`0,60×`–`2,00×`) na tag Xiaomi `39321` (`0x9999`). Calibração física `area_px / zoom²` provada sem dupla contagem `f·zoom` e sem confounding de lente (`0,0%` desvio contrafactual).
  - *Desbalanceamento e Cenários:* `70%` das imagens têm peso `≤ 500 g` com lacuna em `801–1.216 g`, distribuídas em 4 cenários (SF-400 com LCD, balde branco dorsal, cama de palha oblíqua e multi-aves).

## Constraints

- **Classical PDI Only**: Proibido usar redes neurais/deep learning (CNNs/YOLO/ViTs) na segmentação ou regressão principal — exigência central de avaliação técnica em PDI clássico interpretável.
- **Anti-Leakage (Visual & Split)**: Exclusão obrigatória do visor LCD/prato SF-400 antes da extração de atributos e validação estritamente agrupada por `session_id` via 5-fold `StratifiedGroupKFold`.
- **Timeline**: Conclusão do código MVP até **2026-11-28** para apresentação de pitch em **2026-11-30** *(v1.0 entregue em 2026-11-24)*.
- **Performance**: Inferência ponta a ponta na API `/api/predict` abaixo de **500 ms** por imagem em hardware local (CPU) *(atingido `56,1–79,7 ms`)*.
- **Tech Stack**: Backend Python 3.12 (`uv`, `opencv-python-headless`, `scikit-image`, `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `lightgbm`, `pillow`, `fastapi`, `uvicorn`, `joblib`); Frontend React 19 + TypeScript 7 + Vite 8 + Tailwind CSS v4 + Recharts 3.
- **Data Hygiene**: `data/raw/`, `data/processed/`, `models/*.joblib`, `frontend/dist/`, `frontend/node_modules/` e imagens avulsas nunca entram no Git.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Usar exclusivamente PDI clássico (CLAHE Lab, segmentação multi-espaço Lab/HSV, morfologia, descritores geométricos/cromáticos, Watershed) + regressão tabular | Cumpre a `PROIBIÇÃO 1` do PRD e garante interpretabilidade total etapa por etapa ("visão raio-X") no pitch | ✓ Good (motor PDI validado na Fase 2 e 4 famílias tabulares auditadas na Fase 3: `LightGBMLog` `R² = 0,5582` / `0,6905` por sessão, `2,4 ms` CPU) |
| Calibração óptica explícita via EXIF (`focal_length_mm`, `zoom_multiple`, área normalizada) + estudo de ablação | Compensa a variação de até `6,6×` na área em pixels causada pela alternância entre lentes Wide (`4,25 mm`) e Ultrawide (`1,65 mm`) | ✓ Good (Fase 2 provou `area_px / zoom²` com `W/UW = 0,9638`; Fase 3 provou vitória unânime de `C2_physics_normalized_strict` sobre `C1` com `0,0%` desvio contrafactual e desqualificação de `C3`/`C4`) |
| Validação 5-fold `StratifiedGroupKFold` por `session_id` (gap > 10 min) com métricas estratificadas por faixa de peso, lente e cenário | Elimina *split leakage* entre rajadas da mesma ave e expõe comportamento real na cauda rara `>1.000 g` | ✓ Good (split congelado na Fase 1 e avaliação estratificada completa publicada em `models/evaluation_report.json` na Fase 3) |
| Quarentenar 15 imagens (`417`, `757`, `823`, `1520`, `1351`, `1465`) e corrigir `367` → `267 g`, reutilizando as 4 pastas multi-aves exclusivamente no módulo Watershed | Limpa ruído severo da regressão de ave única enquanto aproveita dados reais multi-aves para demonstrar separação de instâncias encostadas | ✓ Good (curadoria na Fase 1; Watershed na Fase 2 com `9/13` exato e `12/13` em ±1; 4 presets Watershed catalogados na Fase 3) |
| Congelar `PDIConfig` (`working_long_edge=768`, hash `43a3cfaa3208019e`) e `49` features estritamente em `bird_mask == 255` sem dimensões brutas | Garante paridade exata entre extração offline (`46,53 s`, `--workers 4`) e inferência online (`p95 = 88,3 ms`) com invariância `0,0`-diff fora da máscara (`PROIBIÇÃO 2`) | ✓ Good |
| Adotar contrafactual dual-mode (`focal_only` + `physical_lens_swap`) e catálogo de presets estritamente por `image_id` opaco (`models/presets_catalog.json`) | Detecta atalhos de lente reconstruídos por árvores em `C3`/`C4` (invisíveis ao flip focal isolado) e preserva `PROIBIÇÃO 4` / ASVS V4 com zero `.jpg`/`rel_path` no Git | ✓ Good (Fase 3: `C2 = 0,0%`, `C3`/`C4` flagrados e desqualificados; `0` substrings proibidas nos JSONs) |
| Arquitetura desacoplada FastAPI local + SPA React/TypeScript/Vite/Tailwind em 4 fases + modo single-origin sobre `frontend/dist/` (`_NonApiSpaRoute`) | Alinha diretamente ao prazo de 28/11, mantém latência `< 500 ms`, elimina fricção de CORS e oferece lançamento em 1 comando no dia do pitch | ✓ Good (Fase 4 entregue 4 dias antes do freeze: `56–80 ms` `/api/predict`, `0,94 g` Dual-Lens ao vivo, `272` pytest + `26` Vitest + E2E single-origin) |
| Manter v1 estritamente upload de imagem (sem vídeo) — resolução de `WARNING 1` (`proposta_pdi.md.pdf` vs. `PRD.md`) | Preserva `PROIBIÇÃO 5` e o prazo de 28/11; vídeo multi-frame permanece no roadmap conceitual | ✓ Good |
| Exibir peso estimado ilustrativo por ave + média da cena no módulo Watershed com disclaimer de soma de grupo — resolução de `WARNING 2` | Atende à proposta acadêmica (`proposta_pdi.md.pdf`) mantendo rigor científico sobre `417`/`757`/`823`/`1520` (que só têm rótulo de peso somado) | ✓ Good (Fase 4: banner permanente `group_sum_only`, 3 KPIs agregados, swatches sincronizados à Etapa 4 e sliders reconciliados ao `backend/app/pdi/config.py`) |
| Manter calibração EXIF dual-lens no v1 e câmera fixa vertical no teto no roadmap v2 (`BARN-01`) — resolução de `WARNING 3` | Alinha o v1 à realidade física das imagens handheld do dataset auditado enquanto preserva a arquitetura-alvo do galpão no painel `WEB-04` | ✓ Good |
| Travar validação v1 em `116–2.093 g` (~7–29 dias) e divulgar explicitamente a divergência dos "45 dias" do Mendeley no `WEB-05` — resolução de `WARNING 4` | Evita promessas impossíveis na faixa ausente `30–45 dias` (`BARN-02`) e reforça a transparência forense no pitch | ✓ Good |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-11-24 after v1.0 milestone*
