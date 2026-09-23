# Verification record

Local verification performed 2026-09-23. This record distinguishes executed checks from proposed future evaluation.

| Check | Result |
|---|---|
| Official MovieLens archive | Retrieved; official MD5 matched; SHA-256 recorded |
| Source schema | 100,000 unique user/movie pairs; rating range checked; 1,682 movie metadata rows |
| Full training and evaluation | Completed; fixed configurations; validation selection saved before test |
| Real-data smoke configuration | Completed in separate ignored directories; not used for benchmark claims or model selection |
| Python tests | 11 passed, including both fictional and private real-data export parity |
| Node scorer tests | 2 passed |
| Independent metric audit | All four scorers' Recall, NDCG and coverage agreed within 1e-12; 250 users |
| Public site build | 8 allowlisted files; private data/weights excluded |
| Desktop | Inspected at 1280×960; populated controls, shortlist and evidence |
| Mobile | Inspected at 390×844; stacked layout and horizontally scrollable data tables; no page overflow |
| Browser interactions | Two-favorite selection, favorite exclusion, empty search, clear/fallback, local bundle import, return to sandbox, evidence navigation |
| Real-data browser inference | Local bundle loaded; two real favorites selected; six displayed recommendations/scores matched Python CLI to displayed precision |
| Console | No errors/warnings at inspection |
| Screenshots | Desktop/mobile saved using fictional assets only |

The JavaScript/Python parity test compares the same exported numeric values to 1e-10 absolute tolerance, with exact top-10 IDs. Cases include no favorites, one favorite, three favorites, duplicate favorites and all movies selected. The independent audit does not use production ranking/metric functions and does not train a model.

Browser tests are manual tool-driven acceptance checks, not a committed end-to-end browser test suite. They do not establish conformance to every accessibility criterion or performance on real mobile hardware. No online experiment, commercial launch, multiple-seed robustness study, or 2–5-real-favorite quality evaluation was performed. GitHub workflow/deployment status is recorded separately in `deployment.md`.
