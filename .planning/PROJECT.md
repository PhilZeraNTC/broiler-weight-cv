# AvianScale CV

## What This Is

AvianScale CV é um sistema de visão computacional de baixo custo para estimativa não invasiva de peso vivo individual e médio de frangos de corte, combinando técnicas clássicas e interpretáveis de Processamento Digital de Imagens (PDI) com regressão tabular calibrada por parâmetros ópticos EXIF. Voltado a produtores integrados e agroindústrias avícolas, o MVP entrega uma aplicação web interativa local para pitch acadêmico/produto, validada sobre toda a base limpa do dataset público *Broiler weights of Cobb Avian* (1.738 imagens, 160 pesos únicos, 188 sessões), exibindo o passo a passo visual do PDI ("visão raio-X"), estimativa de peso e um módulo demonstrativo de separação de aves encostadas via *Watershed*.

## Core Value

Estimar o peso vivo individual de frangos de corte de forma precisa, rigorosamente auditada e 100% interpretável usando apenas Processamento Digital de Imagens clássico calibrado por óptica EXIF — sem deep learning caixa-preta e com zero vazamento de dados do visor LCD da balança — exposto visualmente etapa por etapa em uma demonstração web local de alta credibilidade.

## Business Context

- **Customer**: Produtores integrados de frango de corte e agroindústrias avícolas (contexto do Paraná, maior produtor brasileiro, onde a pesagem manual por amostragem gera estresse animal, lentidão e baixa frequência de dados).
- **Revenue model**: MVP local de validação para pitch acadêmico/produto (evoluindo no roadmap conceitual para monitoramento contínuo com câmera fixa no teto do aviário).
- **Success metric**: Regressão validada em 5-fold `StratifiedGroupKFold` por `session_id` com ablação comprovando ganho da calibração óptica EXIF, métricas estratificadas (MAE, RMSE, MAPE, $R^2$) por faixa de peso/lente/cenário, inferência local `< 500 ms` por imagem e MVP concluído até 2026-11-28 para pitch em 2026-11-30.
- **Strategy notes**: Especificação em [PRD.md](PRD.md) e auditoria forense completa do dataset em [DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md](docs/dataset/DOCUMENTACAO_DATASET_BROILER_COBB_AVIAN.md).

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] **Ingestor Forense Determinístico (REQ-01):** Varrer recursivamente `data/raw/Broiler weights of Cobb Avian/` aplicando todas as regras de curadoria da Seção 9 da auditoria (pastas aninhadas nível 2, sanitização `1340_`, reversão mojibake CP866, dígitos arábico-orientais, reconstrução de `capture_dt`, transposição EXIF, extração óptica Xiaomi tag `39321`, quarentena de 15 imagens e correção `367` → `267 g`), persistindo `data/processed/manifest_clean.parquet` (1.738 imgs) e `data/processed/manifest_quarantine.parquet` (15 imgs).
- [ ] **Split Anti-Leakage Reprodutível (REQ-02):** Agrupar capturas da mesma pasta com intervalo $\le 10$ min em `session_id` único e atribuir 5 folds via `StratifiedGroupKFold` estratificado por faixas de peso (`<500 g`, `500–1000 g`, `>1000 g`), salvando a coluna de fold no manifesto limpo.
- [ ] **Pré-processamento Clássico em OpenCV (REQ-03):** Pipeline configurável com redimensionamento preservando razão de aspecto (*letterbox* / fator de escala conhecido), suavização preservadora de bordas (Filtro Bilateral / Mediana) e normalização de iluminação local via CLAHE no canal `L` (CIE Lab).
- [ ] **Segmentação Anti-Leakage da Ave (REQ-04):** Isolar a silhueta da ave separando-a da cama de palha, balde plástico branco e removendo obrigatoriamente o visor LCD e prato da balança SF-400 via limiarização multi-espaço (CIE Lab / HSV / Otsu / adaptativa), morfologia matemática, componente conexo dominante e refinamento de contorno.
- [ ] **Extração de Descritores Morfométricos, Cromáticos e Ópticos (REQ-05):** Extrair exclusivamente da máscara segmentada da ave os atributos geométricos (área, perímetro, eixos maior/menor da elipse, excentricidade, área convexa, solidez, circularidade), cromáticos invariantes à escala (razão penugem amarela vs. plumagem branca, estatísticas Lab/HSV) e ópticos EXIF (`focal_length_mm`, `zoom_multiple`, área normalizada $A_{px} / (f_{mm} \cdot \text{zoom})^2$).
- [ ] **Artefatos Visuais Intermediários "Raio-X" (REQ-06):** Retornar 4 imagens codificadas de inspeção para cada inferência: (1) Original orientada, (2) Pré-processada/CLAHE, (3) Máscara binária limpa sem LCD, (4) Overlay final com contorno, elipse ajustada e eixos desenhados.
- [ ] **Módulo Demonstrativo Watershed Multi-Aves (REQ-07):** Pipeline dedicado de separação de instâncias tocantes com *Distance Transform* + máximos locais suavizados + *Marker-based Watershed*, aplicado às 4 pastas multi-aves (`417`, `757`, `823`, `1520`) ou uploads do usuário, retornando mapa de bacias/fronteiras, contagem e área individual por ave.
- [ ] **Treinamento e Comparação de Regressores Tabulares (REQ-08):** Treinar e comparar regressores interpretáveis/tabulares (Ridge/Huber com transformação logarítmica, Random Forest, LightGBM/Gradient Boosting) usando exclusivamente features do pipeline clássico de PDI + metadados ópticos.
- [ ] **Relatório de Validação Rigoroso e Ablação Óptica (REQ-09):** Avaliar todos os modelos nos 5 folds (`session_id` isolado) gerando `models/evaluation_report.json` com MAE (g), RMSE (g), MAPE (%) e $R^2$ global, por faixa de peso (`<500 g`, `500–1000 g`, `>1000 g`), por lente (`4,25 mm` vs `1,65 mm`) e por cenário (balde dorsal vs. balança/cama), comparando com vs. sem calibração óptica EXIF.
- [ ] **Serialização para Produção (REQ-10):** Exportar o melhor pipeline treinado em `models/weight_regressor.joblib` com metadados de versão e intervalos de erro esperado por faixa de peso.
- [ ] **API Backend FastAPI Local (REQ-11):** Endpoints REST `/api/predict`, `/api/watershed-demo`, `/api/presets`, `/api/metrics` com validação Pydantic e latência de inferência `< 500 ms` por imagem.
- [ ] **Estúdio de Inferência & "Raio-X" do PDI (REQ-12):** Frontend React + TypeScript + Vite + Tailwind CSS permitindo upload (com leitura automática de EXIF ou override manual de lente/zoom), exibição lado a lado das 4 etapas do PDI, tabela de atributos extraídos, peso estimado (g), estágio etário estimado e peso médio acumulado do lote simulado.
- [ ] **Galeria de Presets Curados para Pitch (REQ-13):** Amostras reais pré-selecionadas de validação cobrindo diferentes idades/pesos (pintinho ~180 g, intermediário ~550 g, frango ~1.600 g e caso difícil branco-no-balde) exibindo Peso Real (Ground Truth) vs. Peso Estimado e erro percentual ao vivo.
- [ ] **Aba Demonstrativa Watershed & Visão de Futuro (REQ-14):** Tela interativa com as cenas multi-aves (`417`, `757`, `823`, `1520`) separadas via *Watershed* e painel arquitetural comparando o MVP atual com o roadmap de câmera fixa no teto do aviário.
- [ ] **Painel de Métricas e Auditoria (REQ-15):** Dashboard com resultados de validação cruzada 5-fold, curva Peso Real vs. Predito, ablação óptica e impacto da limpeza forense dos dados.

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
| Usar exclusivamente PDI clássico (CLAHE Lab, segmentação multi-espaço Lab/HSV, morfologia, descritores geométricos/cromáticos, Watershed) + regressão tabular | Cumpre a `PROIBIÇÃO 1` do PRD e garante interpretabilidade total etapa por etapa ("visão raio-X") no pitch | — Pending |
| Calibração óptica explícita via EXIF (`focal_length_mm`, `zoom_multiple`, área normalizada $A_{px}/(f_{mm}\cdot \text{zoom})^2$) + estudo de ablação | Compensa a variação de até `6,6×` na área em pixels causada pela alternância entre lentes Wide (`4,25 mm`) e Ultrawide (`1,65 mm`) | — Pending |
| Validação 5-fold `StratifiedGroupKFold` por `session_id` (gap > 10 min) com métricas estratificadas por faixa de peso, lente e cenário | Elimina *split leakage* entre rajadas da mesma ave e expõe comportamento real na cauda rara `>1.000 g` | — Pending |
| Quarentenar 15 imagens (`417`, `757`, `823`, `1520`, `1351`, `1465`) e corrigir `367` → `267 g`, reutilizando as 4 pastas multi-aves exclusivamente no módulo Watershed | Limpa ruído severo da regressão de ave única enquanto aproveita dados reais multi-aves para demonstrar separação de instâncias encostadas | — Pending |
| Arquitetura desacoplada FastAPI local + SPA React/TypeScript/Vite/Tailwind em 4 fases (Coarse granularity) | Alinha diretamente ao prazo de 28/11, mantém latência `< 500 ms` e entrega UI polida e orientada a dados para o pitch | — Pending |

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
*Last updated: 2026-11-20 after initialization*
