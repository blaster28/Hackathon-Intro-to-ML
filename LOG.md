# Submission log, team ____

One line per Kaggle submission. Fill it **before** you look at the score, except the last column.

| Time | Who (codes) | What changed | Why we expect it to help | Validation macro-F1 | Public score |
| --- | --- | --- | --- | --- | --- |
| 15:35 | 8PPQ | HistGradientBoosting + features relatives par (user, sport) + moyenne OOF imbriquée des étiquettes voisines (±0/1/3/12 mois) + copie des doublons + quotas de terciles par groupe | La cible est un tercile par (user, sport) calculé sur train+test : on prédit un classement relatif puis on le coupe aux quotas restants | 0.605 (5-fold ; LogReg 0.563, RF 0.600) | 0.604 |

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

## Confusion matrix (HistGradientBoosting, out-of-fold, after post-processing)

| | predicted easy | predicted moderate | predicted hard |
| --- | --- | --- | --- |
| **true easy** | 5001 | 1958 | 570 |
| **true moderate** | 1960 | 3700 | 1943 |
| **true hard** | 679 | 1864 | 5053 |

Error analysis: _à rédiger par l'équipe_
