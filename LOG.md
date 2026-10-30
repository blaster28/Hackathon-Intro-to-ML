# Submission log, team ____

One line per Kaggle submission. Fill it **before** you look at the score, except the last column.

| Time | Who (codes) | What changed | Why we expect it to help | Validation macro-F1 | Public score |
| --- | --- | --- | --- | --- | --- |
| 15:35 | 8PPQ | HistGradientBoosting + features relatives par (user, sport) + moyenne OOF imbriquée des étiquettes voisines (±0/1/3/12 mois) + copie des doublons + quotas de terciles par groupe | La cible est un tercile par (user, sport) calculé sur train+test : on prédit un classement relatif puis on le coupe aux quotas restants | 0.605 (5-fold ; LogReg 0.563, RF 0.600) | 0.604 |
| 16:09 | 8PPQ | + voisins à vitesse proche (noyau gaussien sur le rang de vitesse, OOF) + corrélation de Spearman étiquette~vitesse et étiquette~dénivelé par (user, sport), OOF | À vitesse égale, la difficulté dépend de ce que la même personne a ressenti à des vitesses proches, et le lien vitesse→difficulté varie d'une personne à l'autre | 0.608 (seed 0 ; seeds 1-3 : 0.6067 ± 0.0005 vs 0.6017 pour la référence) | 0.622 |

Hint 1 (16:20, noted 16:30): already used since v1 (15:35): relative features per user × sport (percentile rank and z-score of each workout within that person's workouts in that sport). Hints 2 and 3 not bought.

## Model comparison (5-fold CV, macro-F1)

| Model | Brut | + doublons | + doublons + quotas |
| --- | --- | --- | --- |
| LogisticRegression | 0.532 | 0.549 | 0.563 |
| RandomForest | 0.595 | 0.596 | 0.600 |
| HistGradientBoosting | 0.587 | 0.595 | 0.605 |

## Improvement experiments (no submission: no change kept)

Same protocol: 5-fold StratifiedKFold `random_state=0`, nested OOF for neighbour features, duplicates then quotas. Rule: keep only a gain ≥ +0.003. Each track is tested on top of what was kept before it.

| # | Experiment | CV macro-F1 (after duplicates + quotas) | Kept? |
| --- | --- | --- | --- |
| 0 | Reference: HGB lr=0.05, 31 leaves, min_samples_leaf=20, l2=1 | 0.6051 | reference |
| 1 | HGB 0.5 / RF 0.5 probability average | 0.6049 | no |
| 1 | HGB 0.7 / RF 0.3 | 0.6055 | no (+0.0004) |
| 1 | HGB 0.3 / RF 0.7 | 0.6042 | no |
| 2 | HGB lr=0.03, leaves=31, msl=20, l2=0 | 0.6047 | no |
| 2 | HGB lr=0.03, leaves=63, msl=20, l2=5 | 0.6075 | no |
| 2 | HGB lr=0.05, leaves=63, msl=20, l2=0 | 0.6048 | no |
| 2 | HGB lr=0.05, leaves=15, msl=50, l2=0 | 0.6043 | no |
| 2 | HGB lr=0.03, leaves=31, msl=20, l2=5 | 0.6051 | no |
| 2 | HGB lr=0.03, leaves=15, msl=50, l2=0 | 0.6003 | no |
| 2 | HGB lr=0.05, leaves=15, msl=20, l2=0 | 0.6016 | no |
| 2 | HGB lr=0.03, leaves=15, msl=20, l2=1 | 0.6007 | no |
| 2 | HGB lr=0.03, leaves=63, msl=50, l2=5 | 0.6086 | no |
| 2 | **HGB lr=0.03, leaves=63, msl=20, l2=0** | **0.6103** | provisionally yes (+0.005), rejected at step 5 |
| 2 | HGB lr=0.05, leaves=63, msl=50, l2=1 | 0.6071 | no |
| 2 | HGB lr=0.03, leaves=15, msl=20, l2=5 | 0.5995 | no |
| 2 | HGB lr=0.03, leaves=15, msl=20, l2=0 | 0.5998 | no |
| 2 | HGB lr=0.03, leaves=15, msl=50, l2=5 | 0.6006 | no |
| 2 | HGB lr=0.05, leaves=15, msl=20, l2=5 | 0.6035 | no |
| 3a | + easy / hard proportions of neighbours (±0, 1, 3, 12 months) | 0.6061 | no (vs 0.6103) |
| 3b | + neighbour mean weighted by exp(-\|Δmonth\|/3) | 0.6060 | no |
| 4 | Quota ranking by P(hard) − P(easy) | 0.6103 | no (identical) |
| 5 | Stability, CV seeds 1/2/3: reference | 0.6013 / 0.6023 / 0.6014 → **0.6017 ± 0.0005** | |
| 5 | Stability, CV seeds 1/2/3: tuned HGB | 0.5999 / 0.6015 / 0.6033 → **0.6016 ± 0.0014** | no: mean gain −0.0001 ± 0.0014 |

Conclusion: the +0.005 of step 2 was selection noise (best of 15 tries on the same split). The submitted reference model stays. Its realistic CV score is ≈ 0.602 rather than 0.605: seed 0 is a slightly favourable split.

## Round 2: new information (no submission: no idea kept)

Same protocol (nested OOF for everything that uses labels, duplicates then quotas). Each idea is scored as the mean over CV seeds 1, 2, 3. Rule: keep only a mean gain ≥ +0.005. Kept ideas would stack in order.

| # | Experiment | Seeds 1 / 2 / 3 | Mean ± std | Gain | Kept? |
| --- | --- | --- | --- | --- | --- |
| 0 | Reference | 0.6013 / 0.6023 / 0.6014 | 0.6017 ± 0.0005 | | reference |
| 1 | Gaussian-kernel neighbours, date + speed rank: exp(-Δm²/18 - Δr²/0.02) | 0.6063 / 0.6012 / 0.6030 | 0.6035 ± 0.0021 | +0.0018 | no |
| 1a | Kernel on speed rank only: exp(-Δr²/0.01) | 0.6064 / 0.6013 / 0.6051 | 0.6042 ± 0.0022 | +0.0026 | no |
| 1b | Date + speed + duration rank | 0.6033 / 0.6047 / 0.6031 | 0.6037 ± 0.0007 | +0.0021 | no |
| 1c | Date + speed + max-speed rank | 0.6007 / 0.6073 / 0.6011 | 0.6031 ± 0.0030 | +0.0014 | no |
| 2 | Per-group OOF Spearman(label, speed rank) and (label, climb rank), plus products with the workout's ranks | 0.6041 / 0.6045 / 0.6081 | 0.6056 ± 0.0018 | +0.0039 | no |
| 3 | Ranking score from HistGradientBoostingRegressor | 0.5981 / 0.6006 / 0.5959 | 0.5982 ± 0.0019 | −0.0034 | no |
| 3 | Mean of ranks (classifier + regressor) | 0.6012 / 0.6032 / 0.6034 | 0.6026 ± 0.0010 | +0.0009 | no |
| 4 | Quotas per user_id instead of (user_id, sport) | 0.6011 / 0.6010 / 0.5997 | 0.6006 ± 0.0006 | −0.0010 | no |
| 5 | altitude_mean_m < -9000 or > 9000 → NaN (15 rows) | 0.5999 / 0.6013 / 0.6035 | 0.6016 ± 0.0015 | −0.0001 | no |
| extra | 1a + 2 together (outside the protocol, chosen after seeing the results) | 0.6074 / 0.6067 / 0.6061 | 0.6067 ± 0.0005 | +0.0050 | **yes, team decision**: gain on all 3 seeds, but chosen after the fact, so check it against the public score |

Also checked: workout_id is shuffled within each user (median Spearman with date 0.004), so it carries no chronological information.

## Round 3

Protocole : `experiments.py` (team_pipeline.py n'est pas modifié pendant les tests), CV 5-fold StratifiedKFold, moyenne ± écart-type sur les seeds 1, 2, 3. OOF imbriqué pour tout ce qui utilise des étiquettes. En CV, les quotas n'utilisent que les étiquettes du fold d'entraînement. Score = macro-F1 après doublons + quotas. Règle : on garde une idée seulement si le gain moyen est ≥ +0.003 ET positif sur chacune des 3 seeds. Les idées gardées s'empilent dans l'ordre, et chaque idée a au plus 2 variantes, fixées avant de voir les résultats.

### Phase 0 : analyse

- **0a. Règle des quotas.** Quand N n'est pas un multiple de 3, quelle classe reçoit les séances en plus ? On a comparé la vraisemblance des comptes du train sous chaque règle. N mod 3 = 1 (57 groupes) : seule « hard reçoit l'extra » est compatible avec tous les groupes (log-vraisemblance −217.5, contre −219.2 et −221.1 pour easy et moderate, chacune impossible dans 1 groupe). N mod 3 = 2 (62 groupes) : seule « moderate + hard » est compatible (−248.9, contre −252.0 pour easy+hard, impossible dans 1 groupe, et −254.7 pour easy+moderate). Conclusion : les extras vont aux classes du haut, ce qui est compatible avec une règle déterministe (un découpage du classement en partant du haut).
- **0b. Erreurs OOF de la référence (seed 1, macro-F1 0.6074).**
  - Par taille de groupe : macro-F1 de 0.53 à 0.56 pour les groupes de ≤ 120 séances, 0.60 pour 121–300, 0.64 pour > 300. Les erreurs easy↔hard passent de 9 % (petits groupes) à 4 % (grands groupes).
  - Par sport : cycling 0.62, running 0.61, commute 0.55, mtb 0.51, orienteering 0.48, walking 0.45.
  - Lecture : l'erreur vient surtout des petits groupes et des sports rares, où les features relatives et les voisins sont estimés sur peu de séances.
- **Doublons et quasi-doublons.** 115 lignes du test appartiennent à un ensemble de lignes test identiques. Quasi-doublons (même user, sport, année, mois, jour, heure, et les 4 mesures à ±1 %) : 211 paires, dont 115 entièrement dans le train, avec 92.2 % d'accord des étiquettes (> 90 %), donc l'idée 4 est testée.
- **Idées écartées par les règles du hackathon.** 15 (LightGBM lambdarank) et 18 (CatBoost ou LightGBM) : seuls les modèles scikit-learn sont autorisés, LightGBM et CatBoost ne sont pas dans scikit-learn. L'idée 18 est remplacée par une moyenne HGB + ExtraTrees (scikit-learn). L'affectation optimale (idée 1) est codée en numpy, sans scipy.

## Confusion matrix, model v2 (HistGradientBoosting + speed-kernel neighbours + per-group correlations, out-of-fold, seed 0)

| | predicted easy | predicted moderate | predicted hard |
| --- | --- | --- | --- |
| **true easy** | 4999 | 1998 | 532 |
| **true moderate** | 1982 | 3716 | 1905 |
| **true hard** | 642 | 1850 | 5104 |

## Confusion matrix, model v1 (HistGradientBoosting, out-of-fold, after post-processing)

| | predicted easy | predicted moderate | predicted hard |
| --- | --- | --- | --- |
| **true easy** | 5001 | 1958 | 570 |
| **true moderate** | 1960 | 3700 | 1943 |
| **true hard** | 679 | 1864 | 5053 |

Error analysis (16:04): almost all errors are between neighbouring classes, and "moderate" is the most confused class (only 3,700 of 7,603 correct, F1 ≈ 0.49 vs ≈ 0.66 for easy and hard), because a middle tercile is squeezed between the other two and a small ranking error tips it to either side; easy and hard are rarely confused with each other (570 and 679), which shows the model ranks effort correctly overall.
