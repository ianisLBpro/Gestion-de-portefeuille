"""
2_Facteurs_de_risque_et_crise_financiere.py

Auteur     : Ianis Le Berre
Module     : Module 2 — Gestion quantitative des risques
             Chapitre 1 — Base du risque
Librairies : numpy, pandas, matplotlib, yfinance, statsmodels
"""

# Facteurs de risque et crise financière
#
# Ce fichier couvre :
#     - Définition des facteurs de risque
#     - Risque systématique vs risque idiosyncratique
#     - Modèles factoriels (OLS)
#     - Application : régression des rendements sur un facteur macroéconomique
#
# Facteur de risque :
#     Variable ou événement qui explique le rendement et la volatilité d'un portefeuille.
#     On distingue deux grandes catégories :
#
#         Risque systématique (market risk) :
#             Affecte la volatilité de l'ensemble des actifs du portefeuille.
#             Ne peut pas être éliminé par la diversification.
#             Exemples : inflation, variation des taux d'intérêt, récession.
#
#         Risque idiosyncratique :
#             Spécifique à un actif ou une classe d'actifs.
#             Peut être réduit par la diversification.
#             Exemples : risque de défaut d'un émetteur obligataire,
#                        taille de la firme, ratio book-to-market, chocs sectoriels.
#
# Actifs utilisés : AAPL, MSFT, AMZN, JPM, JNJ | Période : 2016-2026
# Benchmark : S&P 500 (^GSPC)
# Facteur macro : VIX (indice de volatilité implicite du S&P 500 — proxy du stress de marché)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm


TICKERS = ["AAPL", "MSFT", "AMZN", "JPM", "JNJ"]
START = "2016-01-01"
END = "2026-12-31"
portfolio_weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

# Téléchargement des prix + VIX comme facteur macro
data = yf.download(TICKERS + ["^GSPC", "^VIX"], start=START, end=END,
                   auto_adjust=False, progress=False)
if isinstance(data.columns, pd.MultiIndex):
    adj_close = data["Adj Close"].sort_index()

# Rendements journaliers discrets
returns = adj_close[TICKERS].pct_change().dropna()
market_ret = adj_close["^GSPC"].pct_change().dropna()

# Rendement du portefeuille pondéré
port_ret = returns.dot(portfolio_weights)

# Variation journalière du VIX (facteur de risque macroéconomique)
vix_changes = adj_close["^VIX"].pct_change().dropna()

print(f"Actifs  : {TICKERS}")
print(f"Période : {START} → {END}")


# 1. Visualisation de la dispersion des rendements

# La dispersion des rendements autour de leur moyenne est le signal visuel
# du risque : plus les points s'écartent de zéro, plus le risque est élevé.
#
#     dispersion = rendement_t - E[rendement]
#
#     En période de crise, la dispersion augmente brutalement =>
#     c'est l'empreinte visuelle du risque systématique.

dispersion = port_ret - port_ret.mean()

fig, ax = plt.subplots(figsize=(12, 4))
ax.scatter(dispersion.index, dispersion * 100,
           s=2, color="steelblue", alpha=0.6)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Dispersion des rendements autour de la moyenne — Portefeuille")
ax.set_ylabel("Dispersion (%)")
ax.set_xlabel("Date")
plt.tight_layout()
plt.show()


# 2. Risque systématique vs risque idiosyncratique

# On peut décomposer le risque total d'un actif en deux composantes :
#
#     Risque total = Risque systématique + Risque idiosyncratique
#
#     Risque systématique :
#         Part du risque expliquée par les mouvements du marché.
#         Mesurée par le bêta dans le modèle CAPM :
#             R_i = alpha + beta * R_M + epsilon
#         La variance expliquée par le marché est : beta² * Var(R_M)
#
#     Risque idiosyncratique :
#         Part du risque non expliquée par le marché (résidus epsilon).
#         Var(epsilon) = Var(R_i) - beta² * Var(R_M)
#
#     Décomposition pratique :
#         R² de la régression CAPM = part systématique du risque
#         1 - R²                   = part idiosyncratique du risque

# Alignement des dates
common_idx = port_ret.index.intersection(market_ret.index)
port_ret_a = port_ret.loc[common_idx]
market_ret_a = market_ret.loc[common_idx]

# Régression CAPM : port_ret = alpha + beta * market_ret
X = sm.add_constant(market_ret_a)
model = sm.OLS(port_ret_a, X).fit()
beta = model.params.iloc[1]
alpha = model.params.iloc[0]
r2 = model.rsquared

var_total = port_ret_a.var()
var_systematic = beta**2 * market_ret_a.var()
var_idio = var_total - var_systematic

print("\n--- Décomposition du risque (CAPM) ---")
print(f"  Beta                    : {beta:.4f}")
print(f"  Alpha journalier        : {alpha:.6f}")
print(f"  R²                      : {r2:.4f}")
print(f"  Part systématique (R²)  : {r2:.2%}")
print(f"  Part idiosyncratique    : {1-r2:.2%}")
print(f"  Var totale              : {var_total:.8f}")
print(f"  Var systématique        : {var_systematic:.8f}")
print(f"  Var idiosyncratique     : {var_idio:.8f}")


# 3. Modèle factoriel — régression sur le VIX

# Modèle factoriel : évaluation des facteurs de risque qui expliquent
# le rendement du portefeuille.
#
#     Régression OLS :
#         Variable dépendante   : rendements du portefeuille
#         Variable indépendante : facteur de risque (ici variation du VIX)
#
#     Le VIX mesure la volatilité implicite anticipée par le marché des options
#     sur le S&P 500. C'est un proxy standard du stress de marché :
#         VIX élevé  => incertitude forte => rendements négatifs attendus
#         VIX faible => marché calme
#
#     Un coefficient négatif sur le VIX signifie que lorsque le stress augmente,
#     le portefeuille perd de la valeur — exposition au risque systématique.
#
#     regression = sm.OLS(returns, factor).fit()
#     print(regression.summary())

# Alignement des dates port_ret / vix_changes
common_idx2 = port_ret.index.intersection(vix_changes.index)
port_vix = port_ret.loc[common_idx2]
vix_factor = vix_changes.loc[common_idx2]

# Régression OLS : rendement portefeuille ~ variation VIX
X_vix = sm.add_constant(vix_factor)
reg_vix = sm.OLS(port_vix, X_vix).fit()

print("\n--- Modèle factoriel : Portefeuille ~ ΔVix ---")
print(reg_vix.summary())

coef_vix = reg_vix.params.iloc[1]
pval_vix = reg_vix.pvalues.iloc[1]
r2_vix = reg_vix.rsquared

print(f"\n  Coefficient VIX   : {coef_vix:.4f}")
print(f"  P-value           : {pval_vix:.4f}")
print(f"  Significatif      : {pval_vix < 0.05}")
print(f"  R²                : {r2_vix:.4f}")
print("  Interprétation    : une hausse de 1% du VIX est associée à un")
print(f"                      rendement journalier de {coef_vix*100:.4f}% sur le portefeuille.")


# 4. Visualisation : rendements vs variation du VIX

# Un nuage de points rendement ~ ΔVIX permet de visualiser :
#     - La pente négative (relation inverse rendement/stress)
#     - La dispersion résiduelle (risque idiosyncratique)
#     - La droite de régression (composante systématique)

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(vix_factor * 100, port_vix * 100,
           s=3, alpha=0.4, color="steelblue", label="Observations")

# Droite de régression
x_range = np.linspace(vix_factor.min(), vix_factor.max(), 200)
y_pred = reg_vix.params.iloc[0] + reg_vix.params.iloc[1] * x_range
ax.plot(x_range * 100, y_pred * 100, color="red", linewidth=1.5,
        label=f"OLS={coef_vix:.4f}, R²={r2_vix:.4f}")

ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_title("Rendement du portefeuille vs variation du VIX")
ax.set_xlabel("Variation du VIX (%)")
ax.set_ylabel("Rendement du portefeuille (%)")
ax.legend()
plt.tight_layout()
plt.show()
