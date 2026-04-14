"""
3_Extension_du_modele_à_3_facteurs.py

Auteur     : Ianis Le Berre
Module     : Module 1 — Introduction à la gestion du risque
             Chapitre 3 — Investissement factoriel
Librairies : numpy, pandas, matplotlib, yfinance, pandas_datareader, statsmodels
"""

# Extension du modèle CAPM : Fama-French 5 facteurs (2015)
#
# Ce fichier couvre :
#   - Le modèle de Fama-French à 5 facteurs (2015)
#   - Les facteurs RMW (profitabilité) et CMA (investissement)
#   - Comparaison CAPM vs FF3 vs FF5 via le R² ajusté
#   - Interprétation complète des 5 coefficients
#
# Fama & French (2015) — Extension du modèle 3 facteurs :
#   En 2015, Fama et French ajoutent deux facteurs supplémentaires
#   qui capturent des anomalies non expliquées par SMB et HML.
#
#   R_P = RF + β_M*(R_M-RF) + b_SMB*SMB + b_HML*HML + b_RMW*RMW + b_CMA*CMA + α
#
# Actifs : AAPL, MSFT, AMZN, JPM, JNJ | Période : 2016-2026
# Facteurs : téléchargés via pandas_datareader (Ken French Data Library)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader.data as web
import statsmodels.formula.api as smf



TICKERS = ["AAPL", "MSFT", "AMZN", "JPM", "JNJ"]
START = "2016-01-01"
END = "2026-12-31"

# Téléchargement des prix des 5 actifs
data = yf.download(TICKERS, start=START, end=END,
                   auto_adjust=False, progress=False)
if isinstance(data.columns, pd.MultiIndex):
    data = data["Adj Close"]
data = data.sort_index()

# Rendements journaliers discrets
stock_returns = data.pct_change().dropna()

# Portefeuille pondéré custom
portfolio_weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
port_ret = stock_returns[TICKERS].mul(portfolio_weights, axis=1).sum(axis=1)
stock_returns["Portfolio"] = port_ret

# Téléchargement des facteurs Fama-French 5 journaliers
ff_factors = web.DataReader(
    "F-F_Research_Data_5_Factors_2x3_daily",
    "famafrench", start=START, end=END
)[0]
ff_factors = ff_factors / 100
ff_factors.index = pd.to_datetime(ff_factors.index)
ff_factors.index.name = "Date"

# Alignement des dates — Mkt-RF utilisé directement (CRSP value-weighted)
df = stock_returns[["Portfolio"]].join(ff_factors, how="inner")
df = df.rename(columns={"Mkt-RF": "Mkt_Excess"})

# Excess return du portefeuille
df["Port_Excess"] = df["Portfolio"] - df["RF"]

print("DataFrame avec facteurs FF5 :")
print(df[["Port_Excess", "Mkt_Excess", "SMB", "HML", "RMW", "CMA", "RF"]].head())


# 1. Rappel FF3 — base de comparaison
#
# On réestime le modèle FF3 sur ce même dataset
# pour avoir une base de comparaison propre avec FF5.

model_ff3 = smf.ols(formula="Port_Excess ~ Mkt_Excess + SMB + HML", data=df)
fit_ff3 = model_ff3.fit()
r2_adj_ff3 = fit_ff3.rsquared_adj
alpha_ff3 = fit_ff3.params["Intercept"]
alpha_ff3_annual = ((1 + alpha_ff3) ** 252) - 1

print(f"\nFF3 — R² ajusté   : {r2_adj_ff3:.4f}")
print(f"FF3 — Alpha annualisé : {alpha_ff3_annual:.4%}")


# 2. Les deux nouveaux facteurs : RMW et CMA
#
# Fama & French (2015) ajoutent deux facteurs pour capturer
# des anomalies persistantes non expliquées par le modèle FF3 :
#
#   RMW (Robust Minus Weak) — Facteur de profitabilité :
#       Rendement des entreprises à forte rentabilité opérationnelle
#       MOINS celui des entreprises à faible rentabilité.
#       => b_RMW > 0 : le portefeuille est exposé aux entreprises rentables
#       => b_RMW < 0 : le portefeuille est exposé aux entreprises peu rentables
#
#       Intuition : les entreprises très rentables génèrent du cash
#       et récompensent leurs actionnaires => prime de profitabilité.
#
#   CMA (Conservative Minus Aggressive) — Facteur d'investissement :
#       Rendement des entreprises qui investissent peu (conservatrices)
#       MOINS celui des entreprises qui investissent massivement.
#       => b_CMA > 0 : le portefeuille est conservateur (faible capex)
#       => b_CMA < 0 : le portefeuille est agressif (fort capex/croissance)
#
#       Intuition : les entreprises qui surinvestissent diluent souvent
#       leurs actionnaires => prime pour les entreprises conservatrices.
#
#   Note sur notre portefeuille :
#       AAPL, MSFT, AMZN sont des entreprises très rentables (RMW > 0 attendu)
#       mais aussi des entreprises qui investissent massivement (CMA < 0 attendu).


# 3. Modèle de Fama-French à 5 facteurs
#
# R_P = RF + β_M*(R_M-RF) + b_SMB*SMB + b_HML*HML + b_RMW*RMW + b_CMA*CMA + α
#
#   model = smf.ols(
#       formula="Port_Excess ~ Mkt_Excess + SMB + HML + RMW + CMA",
#       data=df
#   )
#   fit = model.fit()

model_ff5 = smf.ols(
    formula="Port_Excess ~ Mkt_Excess + SMB + HML + RMW + CMA",
    data=df
)
fit_ff5 = model_ff5.fit()

print("\n--- Fama-French 5 facteurs — Coefficients, p-values, significativité ---")
for factor in ["Mkt_Excess", "SMB", "HML", "RMW", "CMA"]:
    coef = fit_ff5.params[factor]
    pval = fit_ff5.pvalues[factor]
    sig = pval < 0.05
    print(f"  {factor:<12} : coef={coef:.4f} | p-value={pval:.4f} | significatif={sig}")


# 4. Alpha FF5
#
# L'alpha du modèle FF5 est plus exigeant que celui du CAPM ou FF3 :
# il représente la performance inexpliquée après contrôle de 5 facteurs de risque.
#
#   Un alpha FF5 significativement positif est rare et difficile à maintenir.
#   Si alpha_FF5 < alpha_FF3, cela signifie que RMW et/ou CMA
#   expliquaient une partie de ce qui semblait être de l'alpha en FF3.
#
#   alpha_annuel = (1 + alpha_journalier)^252 - 1

alpha_ff5 = fit_ff5.params["Intercept"]
alpha_ff5_pval = fit_ff5.pvalues["Intercept"]
alpha_ff5_annual = ((1 + alpha_ff5) ** 252) - 1

print("\n--- Alpha FF5 ---")
print(f"  Alpha journalier  : {alpha_ff5:.6f}")
print(f"  Alpha annualisé   : {alpha_ff5_annual:.4%}")
print(f"  P-value alpha     : {alpha_ff5_pval:.4f}")
print(f"  Significatif      : {alpha_ff5_pval < 0.05}")


# 5. Comparaison CAPM vs FF3 vs FF5
#
# Le R² ajusté croît à chaque ajout de facteurs pertinents.
# Si le gain FF3 → FF5 est faible ou nul, RMW et CMA
# n'apportent pas d'explication supplémentaire significative
# pour ce portefeuille spécifique.
#
# La comparaison des alphas est également instructive :
#   alpha_FF5 < alpha_FF3 => une partie de l'alpha FF3 était
#   en réalité une prime de profitabilité ou d'investissement,
#   pas une vraie surperformance du gérant.

# Réestimation CAPM pour cohérence
model_capm = smf.ols(formula="Port_Excess ~ Mkt_Excess", data=df)
fit_capm = model_capm.fit()
r2_adj_capm = fit_capm.rsquared_adj
alpha_capm_annual = ((1 + fit_capm.params["Intercept"]) ** 252) - 1

r2_adj_ff5 = fit_ff5.rsquared_adj
gain_ff3 = r2_adj_ff3 - r2_adj_capm
gain_ff5 = r2_adj_ff5 - r2_adj_ff3

print("\n--- Comparaison des modèles ---")
print(f"{'Modèle':<8} {'R² ajusté':>12} {'Alpha annualisé':>18} {'Gain R²':>10}")
print(f"{'CAPM':<8} {r2_adj_capm:>12.4f} {alpha_capm_annual:>17.4%} {'—':>10}")
print(f"{'FF3':<8} {r2_adj_ff3:>12.4f} {alpha_ff3_annual:>17.4%} {gain_ff3:>+10.4f}")
print(f"{'FF5':<8} {r2_adj_ff5:>12.4f} {alpha_ff5_annual:>17.4%} {gain_ff5:>+10.4f}")


# 6. Visualisation des coefficients FF5
#
# Visualiser les coefficients permet de voir d'un coup d'oeil
# le profil de risque du portefeuille selon les 5 dimensions Fama-French.
# Un coefficient positif = exposition dans le sens du facteur.
# Un coefficient négatif = exposition dans le sens inverse.

factors = ["Mkt_Excess", "SMB", "HML", "RMW", "CMA"]
coefficients = [fit_ff5.params[f] for f in factors]
colors = ["steelblue" if c > 0 else "tomato" for c in coefficients]

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(factors, coefficients, color=colors, edgecolor="black", linewidth=0.6)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Expositions Fama-French 5 facteurs — Portefeuille pondéré custom")
ax.set_ylabel("Coefficient")
ax.set_xlabel("Facteur")
for i, (f, c) in enumerate(zip(factors, coefficients)):
    ax.text(i, c + (0.01 if c >= 0 else -0.02),
            f"{c:.3f}", ha="center", va="bottom" if c >= 0 else "top",
            fontsize=9)
plt.tight_layout()
plt.show()
