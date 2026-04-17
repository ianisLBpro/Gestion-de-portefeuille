"""
1_Quantification_du_risque.py

Auteur     : Ianis Le Berre
Module     : Module 2 — Gestion quantitative des risques
             Chapitre 1 — Base du risque
Librairies : numpy, pandas, matplotlib, yfinance
"""

# Quantification du risque — Gestion quantitative des risques (QRM)
#
# Ce fichier couvre :
#     - Définition du Quantitative Risk Management (QRM)
#     - Le portefeuille financier et ses composantes
#     - Quantification du rendement d'un portefeuille
#     - Quantification du risque via la matrice de covariance
#     - Volatilité glissante (rolling volatility)
#
# Quantitative Risk Management (QRM) :
#     Étude de l'incertitude quantifiable appliquée à un portefeuille financier.
#     L'objectif est de mesurer le risque pour prendre des décisions optimales
#     d'investissement et maximiser le rendement conditionnellement à l'appétit
#     au risque de l'investisseur.
#
#     La crise financière mondiale de 2007-2009 (Global Financial Crisis) a mis en
#     évidence l'importance critique de la gestion des risques :
#         - Variations massives de la valeur fondamentale des actifs
#         - Incertitude extrême sur les rendements futurs
#         - Forte volatilité des rendements
#         - La gestion du risque comme facteur déterminant de survie
#
# Actifs utilisés : AAPL, MSFT, AMZN, JPM, JNJ | Période : 2016-2026
# Benchmark : S&P 500 (^GSPC)
# Pondérations : 30% AAPL | 25% MSFT | 20% AMZN | 15% JPM | 10% JNJ

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


TICKERS = ["AAPL", "MSFT", "AMZN", "JPM", "JNJ"]
START = "2016-01-01"
END = "2026-12-31"
portfolio_weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

# Téléchargement des prix
data = yf.download(TICKERS + ["^GSPC"], start=START, end=END,
                   auto_adjust=False, progress=False)
if isinstance(data.columns, pd.MultiIndex):
    adj_close = data["Adj Close"].sort_index()
else:
    adj_close = data[["Adj Close"]].sort_index()

print(f"Actifs   : {TICKERS}")
print(f"Période  : {START} → {END}")
print(f"Nombre d'observations : {len(adj_close)}\n")


# 1. Rendement du portefeuille

# Un portefeuille financier est un ensemble d'actifs aux rendements futurs incertains
# (actions, obligations, forex, options...).
#
#     Le rendement du portefeuille est la somme pondérée des rendements individuels :
#
#         R_P = w_1*R_1 + w_2*R_2 + ... + w_n*R_n = w · R
#
#     Deux approches équivalentes en Python :
#         1. returns.dot(weights)              # produit scalaire
#         2. returns.mul(weights).sum(axis=1)  # multiplication élément par élément puis somme
#
#     On utilise .pct_change() pour calculer les rendements discrets journaliers :
#         R_t = (P_t - P_{t-1}) / P_{t-1}

# Rendements journaliers discrets
returns = adj_close[TICKERS].pct_change().dropna()
market_ret = adj_close["^GSPC"].pct_change().dropna()

# Rendement du portefeuille — produit scalaire
port_ret = returns.dot(portfolio_weights)

print("--- Rendements journaliers (5 premières lignes) ---")
print(returns.head())
print(f"\nRendement moyen annualisé du portefeuille : {port_ret.mean()*252:.2%}")
print(f"Rendement moyen annualisé du S&P 500     : {market_ret.mean()*252:.2%}")


# 2. Matrice de covariance et volatilité du portefeuille

# La volatilité = mesure de dispersion des rendements autour de la moyenne.
# En finance, la volatilité est utilisée comme proxy du risque.
#
#     Matrice de covariance annualisée :
#         covariance = returns.cov() * 252
#
#     Structure de la matrice :
#         - Diagonale      : variances individuelles de chaque actif
#         - Hors-diagonale : covariances entre paires d'actifs
#
#     Volatilité du portefeuille (écart-type annualisé) :
#         σ²_P = w^T · Cov · w
#         σ_P  = √(σ²_P)
#
#     En Python :
#         portfolio_variance   = np.transpose(weights) @ covariance @ weights
#         portfolio_volatility = np.sqrt(portfolio_variance)
#
#     Note : @ est l'opérateur de multiplication matricielle en Python (numpy).

# Matrice de covariance annualisée
covariance = returns.cov() * 252

print("\n--- Matrice de covariance annualisée ---")
print(covariance.round(4))

# Volatilité du portefeuille
portfolio_variance = np.transpose(portfolio_weights) @ covariance @ portfolio_weights
portfolio_volatility = np.sqrt(portfolio_variance)

# Volatilités individuelles (racine de la diagonale)
individual_vols = np.sqrt(np.diag(covariance))

print("\n--- Volatilités annualisées ---")
for ticker, vol in zip(TICKERS, individual_vols):
    print(f"  {ticker:<6} : {vol:.2%}")
print(f"  {'Portfolio':<6} : {portfolio_volatility:.2%}  (effet diversification)")


# 3. Volatilité glissante (rolling volatility)

# La volatilité instantanée n'est pas constante dans le temps.
# On peut l'observer via une fenêtre glissante (rolling window) :
#
#     windowed   = portfolio_returns.rolling(30)
#     volatility = windowed.std() * np.sqrt(252)
#
#     La fenêtre de 30 jours est une approximation d'un mois de trading.
#     On annualise en multipliant par √252 (règle de la racine carrée du temps).
#
#     Intérêt :
#         - Observer les régimes de volatilité haute et basse
#         - Identifier les épisodes de stress sur le marché
#         - Comparer la volatilité du portefeuille à celle du marché

window = 30   # fenêtre glissante de 30 jours

# Alignement des dates entre port_ret et market_ret
common_idx = port_ret.index.intersection(market_ret.index)
port_ret_a = port_ret.loc[common_idx]
market_ret_a = market_ret.loc[common_idx]

vol_port = port_ret_a.rolling(window).std() * np.sqrt(252)
vol_market = market_ret_a.rolling(window).std() * np.sqrt(252)

print(f"\n--- Volatilité glissante ({window}j) ---")
print(f"  Volatilité moyenne portefeuille : {vol_port.mean():.2%}")
print(f"  Volatilité moyenne S&P 500      : {vol_market.mean():.2%}")
print(f"  Volatilité max portefeuille     : {vol_port.max():.2%}")
print(f"  Volatilité max S&P 500          : {vol_market.max():.2%}")

# Visualisation
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle("Volatilité glissante — Portefeuille vs S&P 500", fontsize=13, fontweight="bold")

# Panel 1 : volatilité du portefeuille
axes[0].plot(vol_port, color="steelblue", linewidth=0.9, label="Portefeuille")
axes[0].set_ylabel("Volatilité annualisée")
axes[0].set_title("Portefeuille (AAPL/MSFT/AMZN/JPM/JNJ)")
axes[0].legend()

# Panel 2 : volatilité du S&P 500
axes[1].plot(vol_market, color="tomato", linewidth=0.9, label="S&P 500")
axes[1].set_ylabel("Volatilité annualisée")
axes[1].set_title("Benchmark S&P 500")
axes[1].set_xlabel("Date")
axes[1].legend()

plt.tight_layout()
plt.show()
