# Submission log, Team_14

One line per Kaggle submission. Fill it **before** you look at the score, except the last column.

| Time | Who (codes) | What changed | Why we expect it to help | Validation macro-F1 | Public score |
| --- | --- | --- | --- | --- | --- |
| 15:35 | 8PPQ | HistGradientBoosting + features relatives par (user, sport) + moyenne OOF imbriquée des étiquettes voisines (±0/1/3/12 mois) + copie des doublons + quotas de terciles par groupe | La cible est un tercile par (user, sport) calculé sur train+test : on prédit un classement relatif puis on le coupe aux quotas restants | 0.605 (5-fold ; LogReg 0.563, RF 0.600) | 0.604 |
| 16:09 | 8PPQ | + voisins à vitesse proche (noyau gaussien sur le rang de vitesse, OOF) + corrélation de Spearman étiquette~vitesse et étiquette~dénivelé par (user, sport), OOF | À vitesse égale, la difficulté dépend de ce que la même personne a ressenti à des vitesses proches, et le lien vitesse→difficulté varie d'une personne à l'autre | 0.608 (seed 0 ; seeds 1-3 : 0.6067 ± 0.0005 vs 0.6017 pour la référence) | 0.622 |
| __:__ | | v3 = v2 + copie des étiquettes des quasi-doublons stricts (même créneau user/sport/année/mois/jour/heure, 4 mesures à ±1 % ; 71 lignes du test) + règle exacte des quotas (N = 3k+1 → l'extra va à hard ; 3k+2 → moderate et hard). Modèle inchangé. | Dans le train, 91 % de ces quasi-doublons ont la même étiquette ; les comptes du train montrent que l'extra va aux classes du haut | 0.6089 (seeds 1-3, vs 0.6071 ; positif sur les 3) | 0.623 |
| __:__ | | v4 = v3 + 16 nouvelles features de voisinage (k-NN dans l'espace standardisé du groupe, résidus label − 2·rang de vitesse pondérés en temps et en vitesse, tendance du label dans le temps), toutes en OOF imbriqué + moyenne de 5 HGB (seeds 0-4) + affectation sous quotas par coût ordinal Σ P(k)·\|k−c\| (linear_sum_assignment) au lieu du tri | Les trois gains sont validés séparément sur 3 à 5 seeds et viennent de mécanismes différents (information, variance, décision), donc ils devraient s'additionner : +0.005 + 0.0025 + 0.002 ≈ +0.009 attendu | **0.6187** (seeds 1-3 : 0.6198 / 0.6191 / 0.6172, vs 0.6089 pour v3, +0.010, positif sur les 3 seeds ; avec les mêmes probabilités, le tri-et-coupe donne 0.6168 / 0.6180 / 0.6132, donc l'affectation ordinale apporte +0.003) | **0.636** (Gate 2 = 0.63 passed) |

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

## Round 3 bis (Johan, 16:20-17:00, after Hint 1): seven more ideas, none kept

Hint 1 ("compare a workout with what that person usually does in that sport") is already what the pipeline does. Same protocol as round 2:
nested OOF, duplicates then quotas, mean of CV seeds 1 / 2 / 3, keep only a mean gain >= +0.005. Reference re-run on this machine: 0.6106 / 0.6071 / 0.6036 -> **0.6071 +- 0.0029**.

| # | Experiment | Seeds 1 / 2 / 3 (full) | Mean | Gain | Kept? |
| --- | --- | --- | --- | --- | --- |
| 6 | Ordinal decomposition: two binary HGB, P(y>=moderate) and P(y>=hard), ranking score = sum | 0.6067 / - / - | 0.6067 | -0.004 | no |
| 7 | Soft vote 0.6 HGB + 0.4 RandomForest on the v2 features | 0.6089 / - / - | 0.6089 | -0.002 | no |
| 8 | Same-slot neighbours: mean label of the person's labelled sessions in the same (year, month, weekday, hour), same (year, month, weekday), and any sport; plus near-duplicates (same slot, duration +-2 min, distance +-0.3 km). Motivation: two sessions in the same slot agree on the label 50 % of the time (baseline 33 %), and 2,778 test rows share a slot with a train row | 0.6081 / 0.6053 / 0.6077 | 0.6071 | +0.000 | no: raw argmax gains +0.005 but the quotas already recover that information |
| 9 | Transductive self-training: pseudo-label validation + test with the quotas, then use the pseudo-labels as extra neighbours (never as training rows; each row excluded from its own pseudo-label) | 0.6038 / 0.6012 / 0.6030 | 0.6026 | -0.0045 | no: pseudo-labels (~62 % correct) add more noise than signal |
| 10 | Sample weights 1/sqrt(group size) so small groups and rare sports (the worst-predicted) weigh more in training (on top of v3) | 0.5995 / 0.6028 / 0.6053 | 0.6025 | -0.0064 vs v3 0.6089 | no: the large groups carry the signal the model generalises from |
| 11 | Pairwise ranker: HGB trained on same-group pairs (feature difference, target "i harder than j"), each test row scored by its mean win-probability against up to 60 known rows of its group (= estimated percentile), then quotas; alone and rank-averaged with the classifier | seed 1: alone 0.5967, mix 0.6082 (v3 0.6117) | | -0.015 / -0.0035 | no |

| 12 | Per-group Spearman rho shrunk toward the global value with weight n/(n+30), so groups with few labels get a prior | 0.6071 / 0.6094 / 0.6065 | 0.6077 | -0.0012 | no |

Reading after rounds 1-3: the model ranks sessions inside a group about as well as the available columns allow. The public gap to the top team (0.631) is 0.009, the gap to 0.66 would need information we do not have (the heart-rate value itself).

## Round 4 (17:00-17:45): four parallel explorations on top of v3 (0.6089), same protocol, each by one agent

| # | Experiment | Seeds 1 / 2 / 3 (full) | Mean | Gain | Seeds positive | Kept? |
| --- | --- | --- | --- | --- | --- | --- |
| 13 | Hierarchical label-free features for small groups: per-user (all sports) rank/z after per-sport median normalisation, global per-sport rank, shrunk z | best 0.6091 / 0.6087 / 0.6107 | 0.6095 | +0.0006 | 1/3 | no. Only 10 of 183 users practise 2 sports, so per-user = per-group for 95 % of rows; the small-group gap comes from label-based features estimated on few labels, not from relative features |
| 14 | Seed-bagging: mean of 5 HGB with random_state 0..4 (early stopping makes the models differ) | 0.6125 / 0.6132 / 0.6089 | 0.6115 | +0.0026 | 3/3 | **yes** (variance reduction, cannot hurt in expectation) |
| 15 | Quota assignment by linear_sum_assignment minimising expected ordinal error sum_k P(k)·\|k−c\| instead of sorting by expected label (same quotas). Max-likelihood cost (−log p): 0.6081; sort by logit difference: 0.6044 | 0.6131 / 0.6126 / 0.6080 (also seeds 4-5: +0.0036 / +0.0004) | 0.6112 | +0.0022 | 5/5 | **yes**. Gain comes from the moderate column: rows the model believes moderate stay in the middle tertile instead of being pushed out by uncertain rows |
| 16 | New label-neighbour features (nested OOF): k-NN (k=5, 15) in the group's standardised feature space (z of speed, duration, gain, max speed, speed std) + mean distance; kernel-weighted residual (label − 2·speed rank) over time and over speed; label trend (ridge slope vs month, previous-3 vs next-3 months) | 0.6132 / 0.6168 / 0.6118 (seeds 4-5: 0.6118 / 0.6165) | 0.6139 | +0.0050 (+0.0035 over 5 seeds) | 5/5 | **yes** |
| 17 | 2-seed averaging + ordinal assignment together (agent cross-check) | 0.6136 / 0.6147 / 0.6115 | 0.6132 | +0.0043 | 3/3 | effects roughly additive |

## Audit (17:30-17:45): adversarial review of the pipeline by 3 reviewers + 1 skeptic per finding

Checked and found correct: all label-based features read only fit-fold labels; apply and fit rows are disjoint everywhere; duplicate copies come only from known rows; quota counts never double-count; target_counts matches every one of the 193 groups (max excess 0); the CV protocol mirrors the test situation (quotas shared between validation and test rows); only scikit-learn estimators (LogisticRegression, RandomForest, HistGradientBoosting, VotingClassifier), no external data, seeds fixed, submission format valid.

Confirmed limitations (none is a validation leak: CV and public scores stay honest):

| # | Finding | Effect | Action |
| --- | --- | --- | --- |
| A1 | 505 exact-duplicate pairs in train (503 single-label); the inner KFold of oof_label_features does not keep twins in the same fold, so ~4.7 % of training rows get their twin's label inside their own kern_mean / nb0_mean (shift 0.02-0.04 on the 0-2 scale) | small train/apply mismatch, expected 0 to +0.003 if fixed | v5 = GroupKFold by duplicate cluster (612 clusters, 1256 train rows) tested on seeds 1-3: 0.6171 / 0.6168 / 0.6188 = 0.6176 vs v4 0.6187 (-0.0011, worse on 2 of 3 seeds) -> **rejected, v4 stays final** |
| A2 | Training rows see 4/5 of the known labels (inner OOF), prediction rows 100 %: count features nb*_n, kern_w are ~1.25x larger at prediction; smoothed means move < 1 % | CV reproduces the same shift, so honest; effect on F1 unmeasured, likely small | v6 = fold-averaged label features for prediction rows (same 80 % density as training rows), seeds 1-3: 0.6189 / 0.6190 / 0.6176 = 0.6185 vs v4 0.6187 -> no gain, **rejected**; the density shift is harmless |
| A3 | HGB early stopping is on automatically (> 10 000 rows): 10 % of the fit rows held out, max_iter = 400 is a ceiling; the notebook did not say so | none on score; explanation accuracy | sentence added in the notebook |

Refuted by the skeptics: test-internal duplicate clusters split by a quota cut (<= 3 rows, ~0 effect); "quota rule overstated" (the rule is supported by the data: aggregate easy-hard imbalance of the r = 2 groups matches (k, k+1, k+1), not the alternative).

## Gate 2: confusion matrix of the final model v4 (HistGradientBoosting + v4 features + ordinal assignment, out-of-fold 5-fold CV, seed 0, printed by team_notebook_v4_commented.ipynb)

| | predicted easy | predicted moderate | predicted hard |
| --- | --- | --- | --- |
| **true easy** | 4955 | 1958 | 616 |
| **true moderate** | 1833 | 3839 | 1931 |
| **true hard** | 618 | 1822 | 5156 |

Error analysis (17:45): versus v2, the moderate row gains +123 correct (3839 vs 3716) and both easy and hard lose fewer rows to the wrong neighbour; easy<->hard swaps stay rare (616 / 618, about 8 % of each class). The remaining errors are still between neighbouring tertiles: a workout close to a person's 33rd or 66th percentile sits on a boundary that the sensor value itself would settle, and we only estimate on which side it falls.

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
