"""
1_CAPM.py

Auteur     : Ianis Le Berre
Module     : Module 1 — Introduction à la gestion du risque
             Chapitre 3 — Investissement factoriel
Librairies : numpy, pandas, matplotlib, yfinance, pandas_datareader, statsmodels
"""

# Capital Asset Pricing Model (CAPM)
#
# Ce chapitre couvre :
#     - Le concept d'excess return (rendement en excès du taux sans risque)
#     - Le calcul du beta via la covariance
#     - Le calcul du beta via la régression OLS
#     - Le R² et R² ajusté
#
# Le CAPM est le modèle fondateur de la théorie de pricing d'actifs.
# Il prédit que le seul facteur de risque systématique est le marché.
#
#     E(R_P) - RF = beta_P * (E(R_M) - RF)
#
# Actifs utilisés : AAPL, MSFT, AMZN, JPM, JNJ | Période : 2016-2026
# Benchmark      : Facteur Mkt-RF Fama-French (CRSP value-weighted)
# Taux sans risque RF : Fourni par les données Fama-French (journalier)


import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader.data as web
import statsmodels.formula.api as smf

# Supprime les FutureWarnings de pandas_datareader liés au parsing de dates
warnings.filterwarnings("ignore", category=FutureWarning, message=".*date_parser.*")



TICKERS = ["AAPL", "MSFT", "AMZN", "JPM", "JNJ"]
START = "2016-01-01"
END   = "2026-12-31"

# Téléchargement des prix des 5 actifs
data = yf.download(TICKERS, start=START, end=END,
                   auto_adjust=False, progress=False)
if isinstance(data.columns, pd.MultiIndex):
    adj_close = data["Adj Close"]
else:
    adj_close = data[["Adj Close"]]
adj_close = adj_close.sort_index()

# Calcul des rendements journaliers discrets
stock_returns = adj_close.pct_change().dropna()

# Poids du portefeuille pondéré custom (somme = 1)
portfolio_weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

# Rendement journalier du portefeuille pondéré
port_ret = stock_returns[TICKERS].mul(portfolio_weights, axis=1).sum(axis=1)
stock_returns["Portfolio"] = port_ret

print("Rendements journaliers :")
print(stock_returns.head())




# 1. Excess returns

# Excess Return = Return - Risk Free Return
#
# Le rendement en excès mesure ce que le portefeuille rapporte au-delà du taux sans risque
# (placement "garanti", ex. obligation d'État court terme).
#
#     Excess Return_P = R_P - RF
#     Excess Return_M = R_M - RF
#
# Pourquoi travailler avec les excess returns dans le CAPM ?
# Le CAPM compare la rémunération du risque pris par rapport à un investissement sans risque (RF).
# Un investisseur rationnel n'accepte un risque supplémentaire que s'il est compensé par un excess return positif.
#
# Exemple :
#     Investir au Brésil : 10% rendement - 15% RF = -5% excess return
#     Investir aux US    : 10% rendement -  3% RF = +7% excess return
#     => Même rendement brut, mais le contexte de risque est très différent.
#
# Note : le taux sans risque RF est fourni par les données Fama-French
# (colonne "RF", déjà en rendement journalier décimal).

# Téléchargement des facteurs Fama-French journaliers (inclut RF)
ff_factors = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily",
                             "famafrench", start=START, end=END)[0]

# Les facteurs sont fournis en pourcentage => conversion en décimal
ff_factors = ff_factors / 100
ff_factors.index = pd.to_datetime(ff_factors.index)
ff_factors.index.name = "Date"

print("\nFacteurs Fama-French (aperçu) :")
print(ff_factors.head())

# Alignement des dates : jointure entre stock_returns et ff_factors
# On utilise directement Mkt-RF des données Fama-French : c'est le facteur marché officiel
# du CAPM (CRSP value-weighted, déjà soustrait de RF). Plus rigoureux que ^GSPC - RF.
df = stock_returns[["Portfolio"]].join(ff_factors[["RF", "Mkt-RF"]], how="inner")

# Calcul de l'excess return du portefeuille
# Mkt-RF est déjà l'excess return du marché, pas besoin de soustraire RF
df["Port_Excess"] = df["Portfolio"] - df["RF"]
df["Mkt_Excess"]  = df["Mkt-RF"]

print("\nExcess returns (aperçu) :")
print(df[["Portfolio", "RF", "Mkt-RF", "Port_Excess", "Mkt_Excess"]].head())




# 2. Beta via la covariance

# Le beta mesure la sensibilité du portefeuille aux mouvements du marché.
#
#     beta_P = Cov(R_P, R_M) / Var(R_M)
#
#     beta_P > 1 : le portefeuille amplifie les mouvements du marché (plus risqué)
#     beta_P = 1 : le portefeuille se comporte comme le marché
#     beta_P < 1 : le portefeuille amortit les mouvements du marché (moins risqué)
#     beta_P < 0 : le portefeuille évolue en sens inverse du marché
#
# En Python :
#     covariance_matrix      = df[["Port_Excess", "Mkt_Excess"]].cov()
#     covariance_coefficient = covariance_matrix.iloc[0, 1]
#     benchmark_variance     = df["Mkt_Excess"].var()
#     portfolio_beta         = covariance_coefficient / benchmark_variance

covariance_matrix = df[["Port_Excess", "Mkt_Excess"]].cov()
covariance_coeff = covariance_matrix.iloc[0, 1]
benchmark_variance = df["Mkt_Excess"].var()
portfolio_beta_cov = covariance_coeff / benchmark_variance




# 3. Beta via régression OLS

# Le CAPM peut être estimé par régression OLS (Ordinary Least Squares) :
#
#     Port_Excess = alpha + beta * Mkt_Excess + epsilon
#
#     Port_Excess : excess return du portefeuille (variable expliquée y)
#     Mkt_Excess  : excess return du marché (variable explicative X)
#     beta        : pente de la droite de régression = sensibilité au marché
#     alpha       : ordonnée à l'origine (dans le CAPM pur, alpha = 0)
#     epsilon     : résidu = partie non expliquée par le marché
#
# Régression en notation matricielle :
#     y = X · beta + epsilon
#
#     y       : vecteur des excess returns du portefeuille (n x 1)
#     X       : matrice avec une colonne de 1 (constante) + Mkt_Excess (n x 2)
#     beta    : vecteur [alpha, beta_marché] estimé par OLS
#     epsilon : vecteur des résidus
#
#     model = smf.ols(formula="Port_Excess ~ Mkt_Excess", data=df)
#     fit   = model.fit()
#     beta  = fit.params["Mkt_Excess"]

model_capm = smf.ols(formula="Port_Excess ~ Mkt_Excess", data=df)
fit_capm   = model_capm.fit()
beta_ols   = fit_capm.params["Mkt_Excess"]

alpha_ols = fit_capm.params["Intercept"]

print(f"\nBeta (méthode OLS)         : {beta_ols:.4f}")
print(f"Beta (méthode covariance)  : {portfolio_beta_cov:.4f}")
print("Les deux méthodes sont équivalentes et donnent le même résultat.")
print(f"\nAlpha (CAPM)               : {alpha_ols:.6f}")
print("=> Dans le CAPM pur, alpha = 0 : le marché explique intégralement le rendement ajusté du risque.")
print("   Un alpha > 0 indique une surperformance anormale ; un alpha < 0 une sous-performance.\n")




# 3.1 Visualisation de la droite de régression CAPM

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["Mkt_Excess"], df["Port_Excess"],
           alpha=0.3, s=5, color="steelblue", label="Observations journalières")
# Trier les x avant de tracer la droite pour éviter un rendu en zigzag
x_sorted = df["Mkt_Excess"].sort_values()
ax.plot(x_sorted,
        fit_capm.params["Intercept"] + fit_capm.params["Mkt_Excess"] * x_sorted,
        color="red", linewidth=1.5, label=f"Droite OLS (beta={beta_ols:.2f})")
ax.set_title("CAPM — Droite de régression : Port_Excess ~ Mkt_Excess")
ax.set_xlabel("Facteur marché FF (Mkt-RF)")
ax.set_ylabel("Excess return portefeuille")
ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
ax.axvline(0, color="black", linewidth=0.6, linestyle="--")
ax.legend()
plt.tight_layout()
plt.show()




# 4. R² et R² ajusté

# R² (coefficient de détermination) :
# Mesure la proportion de la variance du portefeuille expliquée par le facteur marché.
#
#     R² = 1 - SS_res / SS_tot
#
#     SS_res : somme des carrés des résidus (variance non expliquée par le modèle)
#     SS_tot : variance totale des excess returns du portefeuille
#
#     R² = 0.70 => 70% de la variance du portefeuille s'explique par le marché.
#     Les 30% restants = risque idiosyncratique (spécifique aux titres choisis).
#     Ce risque idiosyncratique peut être réduit par la diversification.
#
# R² ajusté :
# Pénalise l'ajout de variables explicatives inutiles.
# Toujours <= R². Augmente seulement si la variable ajoutée apporte
# une réelle valeur explicative.
#
#     => Préférer le R² ajusté pour comparer des modèles avec
#        des nombres de facteurs différents (CAPM vs FF3 vs FF5).
#
#     fit.rsquared      => R²
#     fit.rsquared_adj  => R² ajusté

r2_capm     = fit_capm.rsquared
r2_adj_capm = fit_capm.rsquared_adj

print(f"\nCAPM — R²        : {r2_capm:.4f}")
print(f"CAPM — R² ajusté : {r2_adj_capm:.4f}")
print(f"\nInterprétation : {r2_capm:.2%} de la variance du portefeuille est expliquée par le seul facteur marché (Mkt-RF Fama-French).")
