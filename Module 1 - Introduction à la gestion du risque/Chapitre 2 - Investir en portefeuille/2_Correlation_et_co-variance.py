"""
2_Correlation_et_co-variance.py

Auteur     : Ianis Le Berre
Module     : Module 1 — Introduction à la gestion du risque
             Chapitre 2 — Investir en portefeuille
Librairies : numpy, pandas, seaborn, matplotlib, yfinance
"""

# Correlation and covariance
#
# Ce chapitre couvre :
#     - La corrélation de Pearson entre actifs
#     - La matrice de covariance et son annualisation
#     - Le calcul de la volatilité d'un portefeuille via la formule matricielle
#     - L'impact de la diversification : pourquoi on ne fait pas juste somme(w_i * sigma_i) ?
#     - La visualisation de la matrice de corrélation avec seaborn
#
# Actifs utilisés : AAPL, MSFT, AMZN, JPM, JNJ | Période : 2016-2026


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf



TICKERS = ["AAPL", "MSFT", "AMZN", "JPM", "JNJ"]
NUM_STOCKS = len(TICKERS)

# Téléchargement des données de prix pour les 5 actifs
data = yf.download(TICKERS, start="2016-01-01", end="2026-12-31", auto_adjust=False, progress=False)
if isinstance(data.columns, pd.MultiIndex):
    adj_close = data["Adj Close"]
adj_close = adj_close.sort_index()

# Calcul des rendements discrets pour chaque actif
StockReturns = adj_close.pct_change().dropna()
print("Rendements journaliers :")
print(StockReturns.head())

# Poids choisis personnellement pour le portefeuille pondéré (somme = 1)
portfolio_weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

# Poids pour portefeuille équipondéré (EW)
portfolio_weights_ew = np.repeat(1 / NUM_STOCKS, NUM_STOCKS)

# Poids par capitalisation boursière (MCAP)
market_capitalizations = np.array([2500, 2200, 1800, 500, 400])  # AAPL, MSFT, AMZN, JPM, JNJ
mcap_weights = market_capitalizations / np.sum(market_capitalizations)




# 1. Matrice de corrélation de Pearson

# Corrélation de Pearson :
# Mesure la relation linéaire entre deux séries de rendements.
#
#     rho(X, Y) = Cov(X, Y) / (sigma_X * sigma_Y)
#
# Valeurs possibles :
#      1  : corrélation parfaitement positive (les actifs bougent ensemble)
#      0  : aucune relation linéaire
#     -1  : corrélation parfaitement négative (ils bougent en sens opposés)
#
# Intérêt en gestion de portefeuille :
#     Des actifs peu corrélés => la diversification réduit le risque global.
#     Des actifs très corrélés => peu de bénéfice à les combiner.
#
# StockReturns.corr() :
#     - Diagonale = 1 (un actif est parfaitement corrélé à lui-même)
#     - Matrice symétrique : rho(A, B) = rho(B, A)

correlation_matrix = StockReturns[TICKERS].corr()
print("\nMatrice de corrélation :")
print(correlation_matrix.round(4))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f",
            cmap="coolwarm", vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, square=True)
ax.set_title("Matrice de corrélation des rendements")
plt.tight_layout()
plt.show()




# 2. Matrice de covariance

# Matrice de covariance (Sigma) :
# Chaque élément (i, j) contient Cov(R_i, R_j).
# La diagonale contient les variances : Var(R_i) = sigma_i²
#
#     Sigma_ij = E[(X_i - mu_i) * (X_j - mu_j)]
#
# StockReturns.cov() utilise ddof=1 par défaut (correction de Bessel, estimateur non biaisé)

cov_mat = StockReturns[TICKERS].cov()
print("\nMatrice de covariance journalière :")
print(cov_mat.round(6))




# 3. Annualisation de la matrice de covariance

# Annualisation de la matrice de covariance :
#     La variance est additive dans le temps (si rendements i.i.d.)
#     => cov_mat_annual = cov_mat * 252
#
#     Attention : on multiplie par 252, PAS par sqrt(252).
#     sqrt(252) s'applique à l'écart-type (volatilité).
#     252 s'applique à la variance (et donc à toute la matrice de covariance).

cov_mat_annual = cov_mat * 252
print("\nMatrice de covariance annualisée :")
print(cov_mat_annual.round(4))




# 4. Volatilité du portefeuille, formule matricielle

# Formule à 2 actifs (développée) :
#
#     sigma_p = sqrt(w1²*sigma1² + w2²*sigma2² + 2 * w1 * w2 * rho12 * sigma1 * sigma2)
#
#     sigma_p  : volatilité du portefeuille
#     w        : poids des actifs
#     sigma    : volatilité individuelle de chaque actif
#     rho_1,2  : corrélation entre les actifs 1 et 2
#
#
# Formule matricielle généralisée (Markowitz) :
#
#     sigma_p = sqrt(wᵀ · Σ · w)
#
#     w     : vecteur colonne des poids (taille n)
#     wᵀ    : vecteur ligne transposé (1 x n)
#     Σ     : matrice de covariance annualisée (n x n)
#     ·     : produit scalaire (dot product)
#
#     port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat_annual, weights)))
#
#
# Pourquoi on ne fait pas juste somme(w_i * sigma_i) ?
# Cette somme ignore les corrélations entre actifs.
# Le terme croisé 2 * wi * wj * rho_ij * sigma_i * sigma_j capture la diversification.
# Si les actifs sont peu corrélés => sigma_portfolio < somme pondérée des sigma_i.
# C'est le principe fondamental de la diversification de Markowitz.

port_vol      = np.sqrt(np.dot(portfolio_weights.T,    np.dot(cov_mat_annual, portfolio_weights)))
port_vol_ew   = np.sqrt(np.dot(portfolio_weights_ew.T, np.dot(cov_mat_annual, portfolio_weights_ew)))
port_vol_mcap = np.sqrt(np.dot(mcap_weights.T,         np.dot(cov_mat_annual, mcap_weights)))

print(f"\nVolatilité annualisée — Pondéré  : {port_vol:.4%}")
print(f"Volatilité annualisée — EW       : {port_vol_ew:.4%}")
print(f"Volatilité annualisée — MCAP     : {port_vol_mcap:.4%}")
