# Requirements (PRDs)

Extracted from classified PRD documents in `.planning/intel/classifications/` (`docs/proposta_pdi.md.pdf`). Where `merge`-mode conflict detection identified competing scope or acceptance variants against existing `PRD.md` / `.planning/REQUIREMENTS.md`, both variants are emitted verbatim under separate IDs for user resolution (see `.planning/INGEST-CONFLICTS.md`).

## REQ-classical-pdi-preprocessing
- source: docs/proposta_pdi.md.pdf
- description: Executar pré-processamento das imagens usando técnicas clássicas de Processamento Digital de Imagens, compreendendo correção de iluminação e redução de ruído.
- acceptance: absent
- scope: image preprocessing, classical digital image processing

## REQ-bird-litter-segmentation
- source: docs/proposta_pdi.md.pdf
- description: Segmentar as aves em relação à cama do aviário por limiarização e operações morfológicas, utilizando técnicas clássicas de Processamento Digital de Imagens.
- acceptance: absent
- scope: thresholding and morphological segmentation, classical digital image processing

## REQ-geometric-feature-extraction
- source: docs/proposta_pdi.md.pdf
- description: Extrair atributos geométricos de cada ave segmentada, incluindo área projetada, perímetro, eixos maior e menor da elipse ajustada e área convexa.
- acceptance: absent
- scope: geometric feature extraction

## REQ-weight-regression-model
- source: docs/proposta_pdi.md.pdf
- description: Relacionar os atributos geométricos extraídos de cada ave segmentada ao peso real por meio de um modelo de regressão.
- acceptance: absent (source notes literature studies report R² > 0.9 for projected area vs. weight, without stating a formal pass/fail gate)
- scope: weight regression model, broiler weight estimation

## REQ-density-activity-extension
- source: docs/proposta_pdi.md.pdf
- description: Possível extensão para monitoramento de densidade e de atividade das aves, identificando aves paradas por muito tempo como indicador potencial de problemas sanitários.
- acceptance: absent (framed explicitly as a possible future extension)
- scope: density and activity monitoring

## REQ-mvp-input-modality-proposal
- source: docs/proposta_pdi.md.pdf
- description: Entregar uma aplicação que recebe uma imagem ou vídeo e devolve o peso estimado de cada ave e a média do lote.
- acceptance: absent
- scope: image and video MVP application

## REQ-mvp-input-modality-existing
- source: PRD.md
- description: Entregar uma aplicação web local (FastAPI + React) que recebe upload de imagem (com leitura automática de EXIF ou opção de informar lente/zoom) via /api/predict e /api/watershed-demo (REQ-11, REQ-12), excluindo rastreamento de vídeo multi-frame do código v1 (PROIBIÇÃO 5).
- acceptance: Endpoints /api/predict e /api/watershed-demo validados com Pydantic, inferência < 500 ms por imagem em CPU local, e zero rastreamento de vídeo multi-frame no v1 (ver .planning/REQUIREMENTS.md WEB-01, WEB-02 e Out of Scope).
- scope: image and video MVP application

## REQ-watershed-output-proposal
- source: docs/proposta_pdi.md.pdf
- description: Separar aves encostadas por watershed e, de cada ave segmentada, extrair atributos geométricos e aplicar o modelo de regressão para devolver o peso estimado de cada ave e a média do lote.
- acceptance: absent
- scope: watershed segmentation, broiler weight estimation

## REQ-watershed-output-existing
- source: PRD.md
- description: Implementar módulo demonstrativo dedicado de separação de instâncias tocantes via Distance Transform + máximos locais suavizados + Marker-based Watershed nas 4 pastas multi-aves (417, 757, 823, 1520) ou uploads do usuário, retornando mapa de bacias/fronteiras, contagem e área individual segmentada por ave (REQ-07, REQ-14); o peso médio acumulado do lote simulado é calculado separadamente no estúdio de inferência individual (REQ-12).
- acceptance: Retorna mapa de bacias/fronteiras, contagem de aves separadas e área em pixels por ave nas 13 imagens multi-aves (417, 757, 823, 1520) e uploads customizados (ver .planning/REQUIREMENTS.md PDI-05, WEB-04).
- scope: watershed segmentation, broiler weight estimation

## REQ-camera-geometry-proposal
- source: docs/proposta_pdi.md.pdf
- description: Estimar o peso dos frangos sem contato a partir de imagens de uma câmera fixa no teto do aviário, apontada verticalmente para baixo, voltado a produtores integrados e agroindústrias avícolas com câmera de baixo custo em vez de balanças automáticas.
- acceptance: absent
- scope: broiler weight estimation, camera geometry

## REQ-camera-geometry-existing
- source: PRD.md
- description: Calibrar para o MVP v1 imagens capturadas manualmente com smartphone de duas lentes (Wide 4,25 mm e Ultrawide 1,65 mm) e zoom digital variável usando metadados EXIF (focal_length_mm e zoom_multiple da tag Xiaomi 39321) (REQ-05), apresentando a câmera fixa no teto do aviário como arquitetura conceitual no painel de visão de futuro (REQ-14) e diferindo sua implementação para v2 (BARN-01 em .planning/REQUIREMENTS.md).
- acceptance: Calibração óptica EXIF validada nas 114 sessões mistas, estudo de ablação COM vs. SEM calibração óptica em models/evaluation_report.json, e comparação arquitetural MVP vs. câmera fixa no teto na aba dedicada (ver .planning/REQUIREMENTS.md PDI-03, MOD-02, WEB-04, BARN-01).
- scope: broiler weight estimation, camera geometry

## REQ-dataset-coverage-proposal
- source: docs/proposta_pdi.md.pdf
- description: Desenvolver e validar o MVP com o dataset público "Broiler weights of Cobb Avian" (Mendeley Data), descrito como reunindo imagens e pesagens de frangos ao longo de 45 dias de criação.
- acceptance: absent
- scope: Broiler weights of Cobb Avian dataset

## REQ-dataset-coverage-existing
- source: PRD.md
- description: Validar o MVP sobre a base limpa auditada em disco do dataset público "Broiler weights of Cobb Avian" (1.738 imagens, 160 pesos únicos de 116 g a 2.093 g, 188 sessões, 9 datas cobrindo ~7 a ~29 dias de idade, com 15 imagens em quarentena e correção 367 -> 267 g); a expansão para 30–45 dias (2,1–3,2 kg) é diferida para v2 (BARN-02 em .planning/REQUIREMENTS.md).
- acceptance: Manifesto limpo com 1.738 imagens / 160 pesos / 188 sessões e quarentena com 15 imagens, validação 5-fold StratifiedGroupKFold por session_id, e divulgação explícita da lacuna 801–1.216 g e da faixa real 116–2.093 g no painel de auditoria (ver .planning/REQUIREMENTS.md DATA-01, DATA-02, WEB-05, BARN-02).
- scope: Broiler weights of Cobb Avian dataset
