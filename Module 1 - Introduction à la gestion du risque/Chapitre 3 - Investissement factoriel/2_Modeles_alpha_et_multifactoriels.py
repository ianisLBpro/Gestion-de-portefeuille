"""
2_Modeles_alpha_et_multifactoriels.py

Auteur     : Ianis Le Berre
Module     : Module 1 — Introduction à la gestion du risque
             Chapitre 3 — Investissement factoriel
Librairies : numpy, pandas, matplotlib, yfinance, pandas_datareader, statsmodels
"""

# Alpha et Fama-French 3 facteurs
#
# Ce fichier couvre :
#     - L'alpha : performance inexplicable par les facteurs de risque
#     - Le modèle de Fama-French à 3 facteurs (1993)
#     - Les p-values et la significativité statistique des facteurs
#     - L'extraction des coefficients et leur interprétation
#     - Comparaison CAPM vs FF3 via le R² ajusté
#
# Fama & French (1993) — "Common risk factors in the returns on stocks and bonds"
# Journal of Financial Economics, Volume 33, Issue 1, Pages 3-56
#
# Le CAPM prédit que le seul facteur de risque est le marché.
# Fama et French ont montré empiriquement que deux facteurs supplémentaires
# expliquent une part significative des rendements : SMB et HML.
#
#     R_P - RF = beta_M*(Mkt-RF) + b_SMB*SMB + b_HML*HML + alpha
#
# Actifs utilisés : AAPL, MSFT, AMZN, JPM, JNJ | Période : 2016-2026
# Facteurs      : Fama-French 5 journaliers (Ken French Data Library)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader.data as web
import statsmodels.formula.api as smf



TICKERS = ["AAPL", "MSFT", "AMZN", "JPM", "JNJ"]
START = "2016-01-01"
END   = "2026-12-31"

# Téléchargement des prix des 5 actifs
data = yf.download(TICKERS, start=START, end=END,
                   auto_adjust=False, progress=False)
if isinstance(data.columns, pd.MultiIndex):
    adj_close = data["Adj Close"]
adj_close = adj_close.sort_index()

# Rendements journaliers discrets
stock_returns = adj_close.pct_change().dropna()

# Portefeuille pondéré
portfolio_weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
port_ret = stock_returns[TICKERS].mul(portfolio_weights, axis=1).sum(axis=1)
stock_returns["Portfolio"] = port_ret


# Téléchargement des facteurs Fama-French 5 journaliers (inclut RF, SMB, HML, RMW, CMA)
ff_factors = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily",
                             "famafrench", start=START, end=END)[0]
ff_factors = ff_factors / 100
ff_factors.index = pd.to_datetime(ff_factors.index)
ff_factors.index.name = "Date"


# Alignement des dates
# On utilise directement Mkt-RF des données Fama-French (CRSP value-weighted, déjà soustrait de RF)
df = stock_returns[["Portfolio"]].join(ff_factors, how="inner")

# Excess return du portefeuille ; Mkt-RF est déjà l'excess return du marché
df["Port_Excess"] = df["Portfolio"] - df["RF"]
df["Mkt_Excess"]  = df["Mkt-RF"]
print("DataFrame avec facteurs FF :")
print(df[["Port_Excess", "Mkt_Excess", "SMB", "HML", "RF"]].head())




# 1. Rappel CAPM, base de comparaison

# On réestime d'abord le CAPM sur ce même dataset pour avoir
# une base de comparaison propre avec le modèle FF3.
# Le R² ajusté du CAPM sera notre référence.

model_capm  = smf.ols(formula="Port_Excess ~ Mkt_Excess", data=df)
fit_capm    = model_capm.fit()

r2_adj_capm = fit_capm.rsquared_adj
beta_capm   = fit_capm.params["Mkt_Excess"]

print(f"\nCAPM — Beta        : {beta_capm:.4f}")
print(f"CAPM — R² ajusté   : {r2_adj_capm:.4f}")




# 2. Modèle de Fama-French à 3 facteurs

# Fama & French (1993) identifient deux anomalies que le CAPM ne capture pas :
#
#     SMB (Small Minus Big) :
#         Rendement des petites capitalisations MOINS celui des grandes.
#         Historiquement, les small caps surperforment les large caps.
#         => b_SMB > 0 : le portefeuille est exposé aux small caps
#         => b_SMB < 0 : le portefeuille est exposé aux large caps
#
#     HML (High Minus Low) :
#         Rendement des actions "value" (book-to-market élevé)
#         MOINS celui des actions "growth" (book-to-market faible).
#         => b_HML > 0 : le portefeuille est value
#         => b_HML < 0 : le portefeuille est growth
#
# Modèle FF3 :
#
#     R_P - RF = beta_M*(Mkt-RF) + b_SMB*SMB + b_HML*HML + alpha
#
#     model = smf.ols(formula="Port_Excess ~ Mkt_Excess + SMB + HML", data=df)
#     fit   = model.fit()

model_ff3 = smf.ols(formula="Port_Excess ~ Mkt_Excess + SMB + HML", data=df)
fit_ff3   = model_ff3.fit()

beta_ff3  = fit_ff3.params["Mkt_Excess"]
b_smb     = fit_ff3.params["SMB"]
b_hml     = fit_ff3.params["HML"]

print("\n--- Fama-French 3 facteurs — Coefficients ---")
print(f"  Beta marché  : {beta_ff3:.4f}")
print(f"  b_SMB        : {b_smb:.4f}")
print(f"  b_HML        : {b_hml:.4f}")




# 3. P-values et significativité statistique

# P-value :
# Probabilité d'observer un coefficient aussi extrême si ce facteur
# n'avait aucun effet réel (hypothèse nulle H0 : coefficient = 0).
#
#     p-value < 0.05 => on rejette H0 au seuil de 5%
#                    => le facteur est statistiquement significatif
#     p-value > 0.05 => on ne peut pas rejeter H0
#                    => le facteur n'apporte pas d'explication significative
#
#     fit.pvalues["SMB"]        => p-value du facteur SMB
#     fit.pvalues["SMB"] < 0.05 => True si significatif
#
# Interprétation :
#     Un b_SMB significativement négatif sur un portefeuille de mega-caps
#     (AAPL, MSFT, AMZN) est cohérent : ces titres sont des large caps,
#     donc exposés négativement au facteur SMB.
#     Un b_HML significativement négatif serait cohérent pour un portefeuille
#     tech/growth où le book-to-market est faible.

print("\n--- P-values et significativité ---")
for factor in ["Mkt_Excess", "SMB", "HML"]:
    coef  = fit_ff3.params[factor]
    pval  = fit_ff3.pvalues[factor]
    sig   = pval < 0.05
    print(f"  {factor:<12} : coef={coef:.4f} | p-value={pval:.4f} | significatif={sig}")




# 4. Alpha et hypothèse des marchés efficients

# Alpha (ordonnée à l'origine de la régression) :
# Performance du portefeuille inexplicable par les facteurs de risque.
#
#     alpha > 0 : surperformance ajustée du risque => le gérant crée de la valeur
#     alpha = 0 : toute la performance s'explique par les facteurs (EMH)
#     alpha < 0 : sous-performance après ajustement pour le risque
#
# Hypothèse des marchés efficients (EMH, Fama 1970) :
#     Dans un marché efficient, il est impossible de générer de l'alpha
#     de manière persistante => alpha = 0 en espérance.
#     En pratique, un alpha significatif sur données historiques
#     ne garantit pas qu'il persistera dans le futur.
#
# Annualisation de l'alpha :
#     L'alpha de la régression est journalier.
#     => alpha_annuel = (1 + alpha_journalier)^252 - 1
#
#     fit.params["Intercept"]               => alpha journalier
#     ((1 + alpha_journalier) ** 252) - 1   => alpha annualisé

alpha_ff3         = fit_ff3.params["Intercept"]
alpha_ff3_pval    = fit_ff3.pvalues["Intercept"]
alpha_ff3_annual  = ((1 + alpha_ff3) ** 252) - 1

print("\n--- Alpha ---")
print(f"  Alpha journalier  : {alpha_ff3:.6f}")
print(f"  Alpha annualisé   : {alpha_ff3_annual:.4%}")
print(f"  P-value alpha     : {alpha_ff3_pval:.4f}")
print(f"  Significatif      : {alpha_ff3_pval < 0.05}")




# 5. R² ajusté — comparaison CAPM vs FF3

# Le R² ajusté mesure la qualité d'ajustement du modèle
# en pénalisant l'ajout de variables inutiles.
#
#     R² ajusté FF3 > R² ajusté CAPM
#     => SMB et HML apportent une explication supplémentaire
#        de la variance du portefeuille au-delà du seul marché.
#
#     La différence (R²_FF3 - R²_CAPM) représente la part de variance
#     additionnelle expliquée par les facteurs taille et valeur.

r2_adj_ff3 = fit_ff3.rsquared_adj

print("\n--- Comparaison R² ajusté ---")
print(f"  CAPM  : {r2_adj_capm:.4f}")
print(f"  FF3   : {r2_adj_ff3:.4f}")
print(f"  Gain  : +{r2_adj_ff3 - r2_adj_capm:.4f}")
print(f"\n  => FF3 explique {(r2_adj_ff3 - r2_adj_capm):.2%} de variance supplémentaire vs CAPM.")