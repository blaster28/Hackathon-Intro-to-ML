# %% [markdown]
# # Hackathon ML 2026 : cette séance sera-t-elle difficile pour cette personne ?
#
# **Idée centrale.** La cible `intensity` est un découpage en **terciles, par (user_id, sport)**, d'une mesure
# des capteurs (sans doute la fréquence cardiaque), calculé sur train + test réunis.
# Vérification : dans les 193 groupes, aucune classe du train ne dépasse N/3 de plus de 1.
# Donc la question n'est pas « cette séance est-elle dure dans l'absolu ? » mais
# « est-elle plus dure que **les autres séances de cette personne dans ce sport** ? ».
#
# Le pipeline en 6 étapes :
# 1. features **relatives** au groupe (rang, z-score), sans étiquettes ;
# 2. features **voisins dans le temps** : niveau moyen des étiquettes connues du même groupe à ±0/1/3/12 mois ;
# 3. **OOF imbriqué** pour que ces features ne fuient jamais l'étiquette ;
# 4. comparaison de 3 familles de modèles en CV 5-fold ;
# 5. post-traitement : **doublons** (copie de l'étiquette) puis **quotas de terciles** par groupe ;
# 6. entraînement final et `submission.csv`.

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SEED = 0
CLASSES = ["easy", "moderate", "hard"]

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y_train = train["intensity"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
print(train.shape, test.shape)

# %% [markdown]
# ## Étape 1 : un « pool » train + test et les features relatives
#
# On empile train et test dans un seul tableau `pool` : les lignes `0 .. len(train)-1` sont le train,
# les suivantes le test. `Y` contient l'étiquette (0 = easy, 1 = moderate, 2 = hard) pour le train et -1 pour le test.
#
# **Pourquoi c'est autorisé :** ces features n'utilisent **aucune étiquette**, seulement les mesures.
# Les calculer sur train + test est donc sans fuite, et c'est le même ensemble que celui sur lequel
# les terciles ont été coupés.
#
# Pour chaque mesure, dans chaque groupe (user, sport) :
# - `_rk` : rang percentile (0 = la plus basse des séances de la personne, 1 = la plus haute). Robuste aux petits groupes (min. 9 séances).
# - `_z` : écart à la moyenne du groupe, en écarts-types.
#
# On ajoute quelques mesures dérivées (dénivelé par km, vitesse max / vitesse moyenne, ...) et
# `speed_local` : la vitesse comparée aux séances du même groupe à ±3 mois (la forme change dans le temps).

# %%
pool = pd.concat([train.drop(columns="intensity"), test], ignore_index=True)
N_TRAIN = len(train)
TRAIN_IDX = np.arange(N_TRAIN)
TEST_IDX = np.arange(N_TRAIN, len(pool))
Y = np.full(len(pool), -1)
Y[TRAIN_IDX] = y_train

RAW = [c for c in test.columns if c != "workout_id"]   # colonnes d'origine, pour détecter les doublons

pool["t"] = (pool["year"] - pool["year"].min()) * 12 + pool["month"] - 1   # index du mois
pool["gid"] = pool.groupby(["user_id", "sport"]).ngroup()                  # identifiant du groupe
pool["group_size"] = pool.groupby("gid")["gid"].transform("size")

pool["gain_per_km"] = pool["altitude_gain_m"] / pool["distance_km"].clip(lower=0.1)
pool["climb_rate"] = pool["altitude_gain_m"] / pool["duration_min"].clip(lower=1)
pool["speed_ratio"] = pool["max_speed_kmh"] / pool["speed_kmh"].clip(lower=0.1)
pool["speed_cv"] = pool["speed_std_kmh"] / pool["speed_kmh"].clip(lower=0.1)

REL = ["duration_min", "distance_km", "speed_kmh", "max_speed_kmh", "speed_std_kmh",
       "altitude_mean_m", "altitude_gain_m", "altitude_loss_m", "n_points", "start_hour_utc",
       "gain_per_km", "climb_rate", "speed_ratio", "speed_cv"]
grp = pool.groupby("gid")
for c in REL:
    pool[c + "_rk"] = grp[c].rank(pct=True)
    pool[c + "_z"] = (pool[c] - grp[c].transform("mean")) / (grp[c].transform("std") + 1e-9)

# One-hot pour les variables catégorielles (nécessaire pour la régression logistique)
pool = pd.concat([pool, pd.get_dummies(pool[["sport", "gender"]], dtype=int)], axis=1)
ONEHOT = [c for c in pool.columns if c.startswith(("sport_", "gender_"))]

# %% [markdown]
# ## Étape 2 : sommes glissantes dans le temps
#
# Une seule fonction sert à toutes les features « voisins dans le temps ».
# On range les valeurs dans une grille `groupe × mois`, on fait une somme cumulée le long des mois,
# et la somme sur une fenêtre `[t - w, t + w]` devient une simple différence de deux cases.
#
# `window_stats(fit, apply, valeurs, w)` répond, pour chaque ligne de `apply` :
# « parmi les lignes de `fit` du **même groupe** et à **au plus w mois**, quelle est la somme des valeurs et combien y en a-t-il ? »

# %%
N_G = pool["gid"].max() + 1
N_T = pool["t"].max() + 1
GID = pool["gid"].to_numpy()
T = pool["t"].to_numpy()


def window_stats(fit_idx, apply_idx, values, w):
    S = np.zeros((N_G, N_T + 1))
    C = np.zeros((N_G, N_T + 1))
    np.add.at(S, (GID[fit_idx], T[fit_idx] + 1), values)
    np.add.at(C, (GID[fit_idx], T[fit_idx] + 1), 1)
    S, C = S.cumsum(1), C.cumsum(1)          # S[g, k] = somme des mois < k
    g, t = GID[apply_idx], T[apply_idx]
    hi = np.minimum(t + w, N_T - 1) + 1
    lo = np.maximum(t - w, 0)
    return S[g, hi] - S[g, lo], C[g, hi] - C[g, lo]


# Vitesse locale (sans étiquette) : vitesse / moyenne des AUTRES séances du groupe à ±3 mois
s, n = window_stats(np.arange(len(pool)), np.arange(len(pool)), pool["speed_kmh"].to_numpy(), 3)
pool["speed_local"] = (pool["speed_kmh"] / ((s - pool["speed_kmh"]) / (n - 1)).replace(0, np.nan)).fillna(1.0)

BASE = (["year", "month", "weekday", "start_hour_utc", "group_size", "speed_local"]
        + REL + [c + "_rk" for c in REL] + [c + "_z" for c in REL] + ONEHOT)
print(len(BASE), "features de base")

# %% [markdown]
# ## Étape 3 : features « voisins » à partir des étiquettes, sans fuite
#
# `label_features(fit, apply)` : pour chaque ligne de `apply`, moyenne des étiquettes (0/1/2) des séances de `fit`
# du même groupe à ±0, 1, 3, 12 mois, plus leur nombre.
# - La moyenne est **lissée vers 1** (moderate) avec un poids `PRIOR = 3` : avec 1 seul voisin, on ne lui fait pas une confiance totale.
# - Le **nombre** de voisins est aussi une feature : le modèle sait si la moyenne est fiable.
#
# **La règle anti-fuite :** une ligne ne voit jamais sa propre étiquette.
# - Lignes à prédire (validation, test) : `fit` = toutes les lignes d'entraînement, disjointes par construction.
# - Lignes d'entraînement : `oof_label_features` découpe l'entraînement en 5 morceaux ; chaque morceau
#   reçoit ses features calculées à partir des 4 autres (OOF = *out-of-fold*).
#
# Dans la CV, tout cela est refait **dans chaque fold** avec seulement les étiquettes du fold d'entraînement
# (OOF imbriqué). Sinon les étiquettes de validation entreraient dans les features d'entraînement et le score de validation serait trop optimiste.

# %%
WINDOWS = [0, 1, 3, 12]
PRIOR = 3.0


def time_label_features(fit_idx, apply_idx):
    out = {}
    for w in WINDOWS:
        s, n = window_stats(fit_idx, apply_idx, Y[fit_idx].astype(float), w)
        out[f"nb{w}_mean"] = (s + PRIOR * 1.0) / (n + PRIOR)
        out[f"nb{w}_n"] = n
    return pd.DataFrame(out)


def oof_label_features(idx, k=5):
    parts = []
    for a, b in KFold(k, shuffle=True, random_state=SEED).split(idx):
        f = label_features(idx[a], idx[b])
        f.index = b
        parts.append(f)
    return pd.concat(parts).sort_index()


def make_X(idx, nb):
    return pd.concat([pool.loc[idx, BASE].reset_index(drop=True), nb.reset_index(drop=True)], axis=1)

# %% [markdown]
# ## Étape 3 bis : deux nouvelles features à partir des étiquettes (ajoutées après la 2e série d'expériences)
#
# **a) Voisins à vitesse proche (noyau gaussien).** Pour chaque séance, moyenne des étiquettes connues du même
# (user, sport), pondérée par `w = exp(-Δrang_vitesse² / 0.01)` : une séance à vitesse très proche compte presque 1,
# une séance à 0.2 de rang d'écart ne compte presque plus. On ajoute la somme des poids (fiabilité de la moyenne).
# Idée : « les autres fois où cette personne a roulé/couru à cette vitesse, c'était dur ou facile ? ».
#
# **b) Relation vitesse→difficulté propre à chaque personne.** Dans chaque groupe, corrélation de Spearman
# (corrélation entre rangs) entre l'étiquette et le rang de vitesse, et entre l'étiquette et le rang de dénivelé.
# Pour certains, aller vite = dur ; pour d'autres, c'est le dénivelé qui compte. On ajoute aussi le produit
# corrélation × rang de la séance.
#
# Les deux utilisent les étiquettes, donc elles passent par la même règle anti-fuite que les voisins temporels :
# `label_features` est appelée en OOF imbriqué, et une ligne ne voit jamais sa propre étiquette.
# Validation sur les seeds de CV 1, 2, 3 : 0.6067 ± 0.0005 contre 0.6017 ± 0.0005 pour la référence (+0.005 sur chaque seed).

# %%
SPEED_RK = pool["speed_kmh_rk"].to_numpy()
GAIN_RK = pool["altitude_gain_m_rk"].to_numpy()


def speed_kernel_features(fit_idx, apply_idx, inv_bw=1 / 0.01, prior=1.0):
    S = np.zeros(len(apply_idx))
    W = np.zeros(len(apply_idx))
    for gid in np.unique(GID[apply_idx]):
        F = fit_idx[GID[fit_idx] == gid]
        pos = np.where(GID[apply_idx] == gid)[0]
        if len(F) == 0:
            continue
        d = SPEED_RK[apply_idx[pos]][:, None] - SPEED_RK[F][None, :]
        w = np.exp(-d * d * inv_bw)
        S[pos] = w @ Y[F]
        W[pos] = w.sum(1)
    return (S + prior * 1.0) / (W + prior), W


def group_correlation_features(fit_idx, apply_idx):
    df = pd.DataFrame({"g": GID[fit_idx], "y": Y[fit_idx], "s": SPEED_RK[fit_idx], "a": GAIN_RK[fit_idx]})
    r = df.groupby("g")[["y", "s", "a"]].rank()          # Spearman = corrélation de Pearson entre rangs
    r["g"] = df["g"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = r.groupby("g").apply(lambda d: pd.Series({"s": d["y"].corr(d["s"]), "a": d["y"].corr(d["a"])}))
    rho = rho.reindex(range(N_G)).fillna(0.0)
    ag = GID[apply_idx]
    rs, ra = rho["s"].to_numpy()[ag], rho["a"].to_numpy()[ag]
    return {"rho_speed": rs, "rho_gain": ra,
            "rho_speed_x_rk": rs * SPEED_RK[apply_idx], "rho_gain_x_rk": ra * GAIN_RK[apply_idx]}


def label_features(fit_idx, apply_idx):
    out = time_label_features(fit_idx, apply_idx)
    out["kern_mean"], out["kern_w"] = speed_kernel_features(fit_idx, apply_idx)
    for k, v in group_correlation_features(fit_idx, apply_idx).items():
        out[k] = v
    return out

# %% [markdown]
# ## Étape 4 : post-traitement
#
# **Doublons.** 505 groupes de séances identiques dans le train (2 seulement avec des étiquettes différentes),
# et 327 lignes du test identiques à une ligne du train : la même séance enregistrée deux fois.
# `duplicate_labels` copie l'étiquette connue (vote majoritaire) sur la copie. Ce sont des données de la compétition, pas des données externes.
#
# **Quotas de terciles.** Dans un groupe de N séances (train + test), chaque classe en a ≈ N/3.
# On connaît déjà les étiquettes de `known` (et des doublons copiés), donc on sait combien de chaque classe
# il reste à attribuer parmi les lignes à prédire. `balance` :
# 1. calcule ce quota restant par classe (arrondi pour tomber juste sur le nombre de lignes) ;
# 2. trie les lignes du groupe par **difficulté prédite** = P(moderate) + 2·P(hard) ;
# 3. donne `easy` aux q₀ plus faibles, `moderate` aux q₁ suivantes, `hard` au reste.
#
# C'est exactement le mécanisme qui a créé les étiquettes : un découpage en terciles d'un classement.
#
# **Validation honnête :** en CV, `apply` = validation + test, `known` = fold d'entraînement. On n'utilise jamais
# les comptes réels de la validation, comme pour le vrai test.

# %%
ROW_KEY = pd.util.hash_pandas_object(pool[RAW], index=False).to_numpy()


def duplicate_labels(known_idx, apply_idx):
    lab = pd.Series(Y[known_idx], index=ROW_KEY[known_idx])
    majority = lab.groupby(level=0).agg(lambda s: s.mode().iloc[0])
    return pd.Series(ROW_KEY[apply_idx]).map(majority).fillna(-1).astype(int).to_numpy()


def integer_quota(rem, m):
    rem = np.clip(rem, 0, None)
    rem = np.ones(3) if rem.sum() == 0 else rem
    rem = rem * m / rem.sum()
    q = np.floor(rem).astype(int)
    for i in np.argsort(-(rem - q))[: m - q.sum()]:
        q[i] += 1
    return q


def balance(known_idx, apply_idx, proba, fixed):
    pred = fixed.copy()
    score = proba @ np.array([0.0, 1.0, 2.0])
    known_counts = np.zeros((N_G, 3))
    np.add.at(known_counts, (GID[known_idx], Y[known_idx]), 1)
    fx = fixed >= 0
    np.add.at(known_counts, (GID[apply_idx][fx], fixed[fx]), 1)
    group_n = pool.groupby("gid").size().to_numpy()
    apply_gid = GID[apply_idx]
    for g in np.unique(apply_gid):
        rows = np.where((apply_gid == g) & ~fx)[0]
        if len(rows) == 0:
            continue
        q = integer_quota(group_n[g] / 3 - known_counts[g], len(rows))
        order = rows[np.argsort(score[rows], kind="stable")]
        pred[order] = np.repeat([0, 1, 2], q)
    return pred

# %% [markdown]
# ## Étape 5 : validation croisée 5-fold et 3 familles de modèles
#
# Pour chaque fold extérieur :
# 1. features voisins des lignes d'entraînement par OOF interne (étape 3) ;
# 2. features voisins de validation + test à partir des étiquettes d'entraînement seulement ;
# 3. entraînement du modèle, puis probabilités sur validation + test ;
# 4. trois scores macro-F1 sur la validation : **brut** (argmax), **+ doublons**, **+ doublons + quotas**.
#
# Les 3 familles :
# - **LogisticRegression** (linéaire, variables standardisées) : la référence simple ;
# - **RandomForest** : moyenne de nombreux arbres profonds, capte les effets non linéaires ;
# - **HistGradientBoosting** : arbres construits l'un après l'autre, chacun corrige les erreurs des précédents.

# %%
MODELS = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)),
    "random_forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                    max_features=0.3, n_jobs=-1, random_state=SEED),
    "hist_gb": lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                                      l2_regularization=1.0, random_state=SEED),
}


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


def run_cv(make_model, n_splits=5):
    preds = {k: np.zeros(N_TRAIN, int) for k in ["raw", "dup", "full"]}
    for fit, val in StratifiedKFold(n_splits, shuffle=True, random_state=SEED).split(TRAIN_IDX, y_train):
        X_fit = make_X(fit, oof_label_features(fit))
        apply = np.concatenate([val, TEST_IDX])
        X_apply = make_X(apply, label_features(fit, apply))
        model = make_model().fit(X_fit, Y[fit])
        proba = model.predict_proba(X_apply)
        fixed = duplicate_labels(fit, apply)
        raw = proba.argmax(1)
        preds["raw"][val] = raw[: len(val)]
        preds["dup"][val] = np.where(fixed >= 0, fixed, raw)[: len(val)]
        preds["full"][val] = balance(fit, apply, proba, fixed)[: len(val)]
    scores = {k: round(macro_f1(y_train, p), 4) for k, p in preds.items()}
    return scores, preds


results = {}
for name, make_model in MODELS.items():
    results[name] = run_cv(make_model)
    print(f"{name:14s}", results[name][0])

# %% [markdown]
# ### Pistes testées et non retenues
#
# Même protocole (CV 5-fold `random_state=0`, OOF imbriqué, doublons puis quotas). Règle : garder seulement un gain ≥ +0.003.
#
# | Piste | Meilleur score CV | Décision |
# | --- | --- | --- |
# | Référence HGB (lr 0.05, 31 feuilles, l2 1) | 0.6051 | |
# | 1. Moyenne des probabilités HGB + RF (0.5/0.5, 0.7/0.3, 0.3/0.7) | 0.6055 | abandonnée (+0.0004) |
# | 2. Réglage HGB (15 combinaisons) : lr 0.03, 63 feuilles, l2 0 | 0.6103 | gain qui s'est révélé être du bruit (voir 5) |
# | 3a. Proportions easy / hard des voisins | 0.6061 | abandonnée |
# | 3b. Moyenne des voisins pondérée par exp(-\|Δmois\|/3) | 0.6060 | abandonnée |
# | 4. Classer par P(hard) − P(easy) au lieu de P(moderate) + 2·P(hard) | 0.6103 | aucun effet |
# | 5. Stabilité sur les seeds 1, 2, 3 : référence vs HGB réglé | 0.6017 ± 0.0005 vs 0.6016 ± 0.0014 | gain moyen −0.0001 → réglage abandonné |
#
# Leçon : le meilleur des 15 réglages était le meilleur **sur ce découpage-là**. Sur trois autres découpages, il ne fait pas mieux
# que la référence. Choisir le maximum parmi beaucoup d'essais sur la même CV donne un score trop optimiste.
# On garde donc le modèle de référence.
#
# ### Deuxième série : nouvelles informations (moyenne sur les seeds de CV 1, 2, 3 ; seuil +0.005)
#
# | Idée | Score moyen (± écart-type) | Gain | Décision |
# | --- | --- | --- | --- |
# | Référence | 0.6017 ± 0.0005 | | |
# | 1. Voisins pondérés par un noyau gaussien date + rang de vitesse | 0.6035 ± 0.0021 | +0.0018 | non |
# | 1a. Noyau sur le rang de vitesse seul | 0.6042 ± 0.0022 | +0.0026 | non (meilleure variante) |
# | 1b. Date + vitesse + rang de durée | 0.6037 ± 0.0007 | +0.0021 | non |
# | 1c. Date + vitesse + rang de vitesse max | 0.6031 ± 0.0030 | +0.0014 | non |
# | 2. Corrélation de Spearman étiquette ~ rang de vitesse / de dénivelé, par groupe (OOF) | 0.6056 ± 0.0018 | +0.0039 | non |
# | 3. Classement par HistGradientBoostingRegressor | 0.5982 ± 0.0019 | −0.0034 | non |
# | 3. Moyenne des rangs classifieur + régresseur | 0.6026 ± 0.0010 | +0.0009 | non |
# | 4. Quotas par user_id au lieu de (user_id, sport) | 0.6006 ± 0.0006 | −0.0010 | non |
# | 5. altitude_mean_m aberrante (15 valeurs) → NaN | 0.6016 ± 0.0015 | −0.0001 | non |
# | Hors protocole : 1a + 2 ensemble | 0.6067 ± 0.0005 | +0.0050 | **adoptée** par décision d'équipe (gain sur les 3 seeds ; choisie après coup, donc à confirmer sur le score public) |
#
# Toutes les features qui utilisent les étiquettes (noyau, corrélations) sont calculées en OOF imbriqué, comme les voisins temporels.
# Lecture : les nouvelles features vont dans le bon sens (+0.002 à +0.004) mais restent sous le seuil.
# Les quotas par (user, sport) sont meilleurs que par user, ce qui confirme que les terciles sont coupés par (user, sport).

# %% [markdown]
# ## Matrice de confusion (Gate 2)
#
# Lignes = vraie classe, colonnes = classe prédite, sur les prédictions hors-fold du meilleur modèle après post-traitement.

# %%
best = max(results, key=lambda k: results[k][0]["full"])
cm = confusion_matrix(y_train, results[best][1]["full"])
print("meilleur modèle :", best)
print(pd.DataFrame(cm, index=["vrai " + c for c in CLASSES], columns=["prédit " + c for c in CLASSES]))

# %% [markdown]
# ## Étape 6 : entraînement final et soumission
#
# Même schéma qu'un fold de CV, mais avec `fit` = tout le train et `apply` = le test :
# features voisins OOF sur le train, features voisins du test à partir de toutes les étiquettes du train,
# puis doublons et quotas.

# %%
X_fit = make_X(TRAIN_IDX, oof_label_features(TRAIN_IDX))
X_test = make_X(TEST_IDX, label_features(TRAIN_IDX, TEST_IDX))
model = MODELS[best]().fit(X_fit, y_train)
proba_test = model.predict_proba(X_test)
fixed_test = duplicate_labels(TRAIN_IDX, TEST_IDX)
pred_test = balance(TRAIN_IDX, TEST_IDX, proba_test, fixed_test)

submission = pd.DataFrame({"workout_id": test["workout_id"], "intensity": np.array(CLASSES)[pred_test]})
submission.to_csv("submission.csv", index=False)
print(submission["intensity"].value_counts(normalize=True).round(3))
print("doublons copiés depuis le train :", (fixed_test >= 0).sum())
