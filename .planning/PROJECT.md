# AvianScale CV

## What This Is

AvianScale CV é um sistema de visão computacional de baixo custo para estimativa não invasiva de peso vivo individual e médio de frangos de corte, combinando técnicas clássicas e interpretáveis de Processamento Digital de Imagens (PDI) com regressão tabular calibrada por parâmetros ópticos EXIF. Voltado a produtores integrados e agroindústrias avícolas, o MVP entrega uma aplicação web interativa local para pitch acadêmico/produto, validada sobre toda a base limpa do dataset público *Broiler weights of Cobb Avian* (1.738 imagens, 160 pesos únicos, 188 sessões), exibindo o passo a passo visual do PDI ("visão raio-X"), estimativa de peso e um módulo demonstrativo de separação de aves encostadas via *Watershed*.

## Core Value

Estimar o peso vivo individual de frangos de corte de forma precisa, rigorosamente auditada e 100% interpretável usando apenas Processamento Digital de Imagens clássico calibrado por óptica EXIF — sem deep learning caixa-preta e com zero vazamento de dados do visor LCD da balança — exposto visualmente etapa por etapa em uma demonstração web local de alta credibilidade.

## Business Context

- **Customer**: Produtores integrados de frango de corte e agroindústrias avícolas (contexto do Paraná, maior produtor brasileiro, onde a pesagem manual por amostragem gera estresse animal, lentidão e baixa frequência de dados).
- **Revenue model**: MVP local de validação para pitch acadêmico/produto (evoluindo no roadmap conceitual para monitoramento contínuo com câmera fixa no teto do aviário).
- **Success metric**: Regressão validada em 5-fold `StratifiedGroupKFold` por `session_id` com ablação comprovando ganho da calibração óptica EXIF, métricas estratificadas (MAE, RMSE, MAPE, $R^2$) por faixa de peso/lente/cenário, inferência local `< 500 ms` por imagem e MVP concluído até 2026-11-28 para pitch em 2026-11-30.
- **Strategy notes**: Especificação em [PRD.md](PRD.md), proposta acadêmica em [proposta_pdi.md.pdf](docs/proposta_pdi.md.pdf), auditoria forense completa em [DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md](docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md) e síntese de ingestão em [.planning/intel/SYNTHESIS.md](.planning/intel/SYNTHESIS.md).

## Requirements

### Validated

- ✓ **Ingestor Forense Determinístico (REQ-01 / DATA-01):** Varrer recursivamente `data/raw/Broiler weights of Cobb Avian/` aplicando todas as regras da Auditoria §9 (`1.738` limpas, `15` quarentena, `160` pesos únicos, `188` sessões, `367 → 267 g`, CP866/árabe/EXIF/Xiaomi `39321`) persistindo Parquets tipados e `models/dataset_audit_summary.json` — Phase 1
- ✓ **Split Anti-Leakage Reprodutível (REQ-02 / DATA-02):** Agrupamento por `session_id` (gap `> 600 s`) e 5-fold `StratifiedGroupKFold` estratificado (`<500 g`, `500–1000 g`, `>1000 g`) com `0` sobreposição e `[4, 5, 5, 5, 5]` sessões pesadas por fold — Phase 1
- ✓ **Pré-processamento Clássico EXIF-First em OpenCV (REQ-03 / PDI-01):** Loader único Pillow (`ImageOps.exif_transpose()`, `0` chamadas `cv2.imread`/`cv2.imdecode` no repo), downscale preservando aspecto (`working_long_edge = 768 px`, `PIL.Image.draft()`), Bilateral/Mediana e CLAHE no canal `L` (CIE Lab) sob `PDIConfig` congelado (`hash = 43a3cfaa3208019e`) — Phase 2
- ✓ **Segmentação Anti-Leakage da Ave (REQ-04 / PDI-02):** Classificação de cena (`bucket_dorsal`: `377` vs. `scale_or_litter`: `1.361`), supressão invariante a rotação do LCD/prato SF-400 (`786` acionamentos, `0` falsos positivos no balde), segmentação clássica nos 3 cenários com apenas `4/1.738` (`0,23%`) de fallback (`0/377` no balde, incl. `0/59` aves `>1.000 g` branco-sobre-branco) — Phase 2
- ✓ **Descritores Morfométricos, Cromáticos e Calibração Óptica EXIF (REQ-05 / PDI-03):** `49` features estritamente dentro de `bird_mask == 255` (invariância comprovada `0,0`-diff em perturbação externa), calibração `area_optical_norm = area_px / zoom_multiple²` validada em `127` sessões dual-lens (`W/UW = 0,9638` em pares de balde, sem dupla contagem `f·zoom` e sem dimensões brutas), extração em lote (`--workers 4`) materializada em `data/processed/features_clean.parquet` (`1.738 × 60`) em `46,53 s` e `models/pdi_calibration_audit.json` — Phase 2
- ✓ **Artefatos Visuais Intermediários "Raio-X" (REQ-06 / PDI-04):** 4 etapas a `640 px` (3× JPEG Q=82 + máscara PNG 1 canal, `p50 = 140,1 KB`, `max = 166,8 KB`) com contorno e elipse/eixos orientados corretamente, latência ponta a ponta em imagens nativas `4080×1840` de `p50 = 74,4 ms`, `p95 = 88,3 ms` (`< 500 ms`) — Phase 2
- ✓ **Módulo Demonstrativo Watershed Multi-Aves (REQ-07 / PDI-05):** Pipeline clássico isolado (pré-supressão cromática de comedouro vermelho + supressão LCD + EDT exata + `h_maxima` recortado por bbox + `skimage.segmentation.watershed`) com `9/13` contagens exatas e `12/13` em ±1 (`MAE = 0,38`, `p50 = 131,8 ms`) nas 13 imagens multi-aves (`417`, `757`, `823`, `1520`) e suporte a uploads arbitrários retornando `49` features por instância — Phase 2
- ✓ **Treinamento e Comparação de Regressores Tabulares (REQ-08 / MOD-01):** Comparação 5-fold OOF (`0` sobreposição de `session_id`) de 4 famílias clássicas (`RidgeLog`, `HuberLog`, `RandomForestLog`, `LightGBMLog`) com wrappers pickláveis (`AllometricLogTransformer`, `LogDuanRegressor`), ponderação intra-fold sessão×faixa limitada (`≤ 5×`), diagnóstico de *Duan smearing* e higiene estrita de features (`FEATURE_COLUMNS` apenas, zero `scene_is_bucket` ou `BANNED_FEATURE_COLUMNS`) — Phase 3
- ✓ **Relatório de Validação Rigoroso, Ablação Óptica 4-Vias e Auditoria Contrafactual (REQ-09 / MOD-02):** `models/evaluation_report.json` bit-determinístico e sem vazamento de caminhos (SHA-256 `b8d21b2d...`) provando vitória unânime de `C2_physics_normalized_strict` sobre `C1_uncalibrated_pdi` (`LightGBMLog`: `MAE = 158,11 g` [`-6,16 g`], `RMSE = 264,02 g` [`-11,17 g`], `BalMAPE = 35,11%` [`-1,08 pp`], `R² = 0,5582` [`+0,0381`], `R² por sessão = 0,6905`), com desvio contrafactual exato de `0,0%` em ambos os modos enquanto `C3` e `C4` são desqualificados por confounding de lente — Phase 3
- ✓ **Serialização Dual-Bundle e Catálogo de Presets OOF (REQ-10 / MOD-03):** `models/weight_regressor.joblib` (git-ignored, `3,18 MiB`, `production_pipeline` + 5 `fold_pipelines` `{0..4}`, `PDIConfig` reconstruível + hash `43a3cfaa3208019e`, carga fria `75,2 ms`, inferência unitária `2,4 ms` com `0` warnings), mapeador etário `age_stage.py` (`~7–29 d` / `116–2.093 g`) e `models/presets_catalog.json` (git-tracked, SHA-256 `3a0c7e50...`, 6 presets OOF de ave única cobrindo os 5 folds + 4 presets Watershed referenciados apenas por `image_id` opaco) — Phase 3

### Active

- [ ] **API Backend FastAPI Local (REQ-11 / WEB-01):** Endpoints REST `/api/predict`, `/api/watershed-demo`, `/api/presets`, `/api/metrics` com validação Pydantic e latência de inferência `< 500 ms` por imagem.
- [ ] **Estúdio de Inferência & "Raio-X" do PDI (REQ-12 / WEB-02):** Frontend React + TypeScript + Vite + Tailwind CSS permitindo upload (com leitura automática de EXIF ou override manual de lente/zoom), exibição lado a lado das 4 etapas do PDI, tabela de atributos extraídos, peso estimado (g), estágio etário estimado e peso médio acumulado do lote simulado.
- [ ] **Galeria de Presets Curados para Pitch (REQ-13 / WEB-03):** Amostras reais pré-selecionadas de validação cobrindo diferentes idades/pesos (pintinho ~180 g, intermediário ~550 g, frango ~1.600 g e caso difícil branco-no-balde) exibindo Peso Real (Ground Truth) vs. Peso Estimado e erro percentual ao vivo.
- [ ] **Aba Demonstrativa Watershed & Visão de Futuro (REQ-14 / WEB-04):** Tela interativa com as cenas multi-aves (`417`, `757`, `823`, `1520`) separadas via *Watershed* e painel arquitetural comparando o MVP atual com o roadmap de câmera fixa no teto do aviário.
- [ ] **Painel de Métricas e Auditoria (REQ-15 / WEB-05):** Dashboard com resultados de validação cruzada 5-fold, curva Peso Real vs. Predito, ablação óptica e impacto da limpeza forense dos dados.

### Out of Scope

- **Redes neurais end-to-end / Deep Learning na inferência principal (CNNs, YOLO, ViTs, SAM)** — Estritamente proibido (`PROIBIÇÃO 1`); o objetivo central da avaliação é demonstrar domínio técnico em Processamento Digital de Imagens clássico e interpretável.
- **OCR ou leitura de pixels fora da máscara da ave (ex.: visor LCD da SF-400)** — Estritamente proibido (`PROIBIÇÃO 2`); configura *data leakage* visual que destrói a validade científica e operacional do modelo.
- **Split aleatório por imagem (`train_test_split`)** — Estritamente proibido (`PROIBIÇÃO 3`); rajadas de ~9 fotos/sessão da mesma ave inflam artificialmente as métricas.
- **Versionamento do dataset bruto/processado ou listagem recursiva das 1.753 imagens em prompts** — Estritamente proibido (`PROIBIÇÃO 4`); 3,54 GiB fora do Git e higiene de contexto.
- **Rastreamento multi-frame de vídeo (*object tracking*), densidade populacional do galpão e alertas sanitários de aves paradas** — Fora de escopo do código v1 (`PROIBIÇÃO 5`); pertencem exclusivamente ao roadmap conceitual apresentado no pitch.

## Context

- **Dataset Auditado (Fonte Única de Verdade):** `data/raw/Broiler weights of Cobb Avian/` contém 1.753 imagens JPEG RGB (3,54 GiB, 0 corrompidas, 0 duplicatas exatas, 0 GPS) em 168 pastas-folha (165 raiz + 3 aninhadas em nível 2: `171/173`, `190/191`, `190/193`) e 166 pesos brutos (116–2.093 g), capturadas em 9 datas entre 2022-03-24 e 2022-04-15 (~7 a ~29 dias de idade, linhagem Cobb Avian 48) com Xiaomi Redmi 10 (`21061119AG`) e balança digital SF-400.
- **Armadilhas Críticas Mapeadas na Auditoria:**
  - *Encoding e Estrutura:* `1340_` tem underscore; 1.635 nomes usam dígitos árabes (`٠-٩`); 21 arquivos em `547/` e `562/` sofrem mojibake UTF-8→CP866; EXIF `DateTime` é `????:??:?? ??:??:??` em 100% (reconstruído por nome ou `mtime - 14401 s`); 81,86% têm `Orientation != 1` (transposição EXIF obrigatória).
  - *Curadoria / Quarentena:* 13 imagens multi-aves (`417`, `757`, `823`, `1520`) e 2 erros grosseiros (`1351`, `1465`) vão para quarentena (15 imgs); `367` (6 imgs) é corrigida para `267 g` (confirmado no LCD). Base limpa: **1.738 imagens**, **160 pesos únicos**, **188 sessões**.
  - *Ambiguidade Óptica e Viés Lente×Peso:* 74,5% Wide (`4,25 mm`, `4080×1840`) vs. 25,5% Ultrawide (`1,65 mm`, `3264×1472`), além de 67 imagens com zoom digital (`0,60×` a `2,00×`) na tag Xiaomi `39321` (`0x9999`). A troca de lente muda a área projetada em pixels em até `6,6×`, e acima de `1.000 g` a fração ultrawide salta de ~20% para 56–78%.
  - *Desbalanceamento e Cenários:* 70% das imagens têm peso ≤ 500 g e há forte lacuna entre `801 g` e `1.216 g`. Quatro cenários coexistem: (A) sobre a balança SF-400 com LCD visível, (B) vista dorsal em balde plástico branco (~353 imgs, baixo contraste branco-sobre-branco em aves >1 kg), (C) ave livre na cama de palha/ângulos oblíquos, (D) multi-aves (quarentena/Watershed). Obturador lento (`1/20–1/50 s`) provoca *motion blur* frequente.
- **Estrutura Inicial do Repositório:** Diretórios vazios `backend/`, `frontend/`, `models/`, `data/processed/`, com `data/raw/` ignorado no `.gitignore`, além de `PRD.md` e `docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md`.

## Constraints

- **Classical PDI Only**: Proibido usar redes neurais/deep learning (CNNs/YOLO/ViTs) na segmentação ou regressão principal — exigência central de avaliação técnica em PDI clássico interpretável.
- **Anti-Leakage (Visual & Split)**: Exclusão obrigatória do visor LCD/prato SF-400 antes da extração de atributos e validação estritamente agrupada por `session_id` via 5-fold `StratifiedGroupKFold`.
- **Timeline**: Conclusão do código MVP até **2026-11-28** para apresentação de pitch em **2026-11-30** (8 dias corridos).
- **Performance**: Inferência ponta a ponta na API `/api/predict` abaixo de **500 ms** por imagem em hardware local (CPU).
- **Tech Stack**: Backend Python 3.11+ (`opencv-python-headless`, `scikit-image`, `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `lightgbm`, `pillow`, `fastapi`, `uvicorn`, `joblib`); Frontend React + TypeScript + Vite + Tailwind CSS.
- **Data Hygiene**: `data/raw/`, `data/processed/`, `models/*.joblib` e imagens avulsas nunca entram no Git (já protegidos pelo `.gitignore`).

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Usar exclusivamente PDI clássico (CLAHE Lab, segmentação multi-espaço Lab/HSV, morfologia, descritores geométricos/cromáticos, Watershed) + regressão tabular | Cumpre a `PROIBIÇÃO 1` do PRD e garante interpretabilidade total etapa por etapa ("visão raio-X") no pitch | ✓ Good (motor PDI validado na Fase 2 e 4 famílias tabulares auditadas na Fase 3: `LightGBMLog` `R² = 0,5582` / `0,6905` por sessão, `2,4 ms` CPU) |
| Calibração óptica explícita via EXIF (`focal_length_mm`, `zoom_multiple`, área normalizada) + estudo de ablação | Compensa a variação de até `6,6×` na área em pixels causada pela alternância entre lentes Wide (`4,25 mm`) e Ultrawide (`1,65 mm`) | ✓ Good (Fase 2 provou `area_px / zoom²` com `W/UW = 0,9638`; Fase 3 provou vitória unânime de `C2_physics_normalized_strict` sobre `C1` com `0,0%` desvio contrafactual e desqualificação de `C3`/`C4`) |
| Validação 5-fold `StratifiedGroupKFold` por `session_id` (gap > 10 min) com métricas estratificadas por faixa de peso, lente e cenário | Elimina *split leakage* entre rajadas da mesma ave e expõe comportamento real na cauda rara `>1.000 g` | ✓ Good (split congelado na Fase 1 e avaliação estratificada completa publicada em `models/evaluation_report.json` na Fase 3) |
| Quarentenar 15 imagens (`417`, `757`, `823`, `1520`, `1351`, `1465`) e corrigir `367` → `267 g`, reutilizando as 4 pastas multi-aves exclusivamente no módulo Watershed | Limpa ruído severo da regressão de ave única enquanto aproveita dados reais multi-aves para demonstrar separação de instâncias encostadas | ✓ Good (curadoria na Fase 1; Watershed na Fase 2 com `9/13` exato e `12/13` em ±1; 4 presets Watershed catalogados na Fase 3) |
| Congelar `PDIConfig` (`working_long_edge=768`, hash `43a3cfaa3208019e`) e `49` features estritamente em `bird_mask == 255` sem dimensões brutas | Garante paridade exata entre extração offline (`46,53 s`, `--workers 4`) e inferência online (`p95 = 88,3 ms`) com invariância `0,0`-diff fora da máscara (`PROIBIÇÃO 2`) | ✓ Good |
| Adotar contrafactual dual-mode (`focal_only` + `physical_lens_swap`) e catálogo de presets estritamente por `image_id` opaco (`models/presets_catalog.json`) | Detecta atalhos de lente reconstruídos por árvores em `C3`/`C4` (invisíveis ao flip focal isolado) e preserva `PROIBIÇÃO 4` / ASVS V4 com zero `.jpg`/`rel_path` no Git | ✓ Good (Fase 3: `C2 = 0,0%`, `C3`/`C4` flagrados e desqualificados; `0` substrings proibidas nos JSONs) |
| Arquitetura desacoplada FastAPI local + SPA React/TypeScript/Vite/Tailwind em 4 fases (Coarse granularity) | Alinha diretamente ao prazo de 28/11, mantém latência `< 500 ms` e entrega UI polida e orientada a dados para o pitch | — Pending |
| Manter v1 estritamente upload de imagem (sem vídeo) — resolução de `WARNING 1` (`proposta_pdi.md.pdf` vs. `PRD.md`) | Preserva `PROIBIÇÃO 5` e o prazo de 28/11; vídeo multi-frame permanece no roadmap conceitual | ✓ Good |
| Exibir peso estimado ilustrativo por ave + média da cena no módulo Watershed com disclaimer de soma de grupo — resolução de `WARNING 2` | Atende à proposta acadêmica (`proposta_pdi.md.pdf`) mantendo rigor científico sobre `417`/`757`/`823`/`1520` (que só têm rótulo de peso somado) | — Pending |
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
*Last updated: 2026-11-23 after Phase 3*
