"""
1_Estimation_du_risque_extreme.py

Auteur     : Ianis Le Berre
Module     : Module 1 — Introduction à la gestion du risque
             Chapitre 4 — Value at Risk
Librairies : numpy, pandas, matplotlib, yfinance
"""

# Estimation du risque extrême : Drawdown, VaR et CVaR
#
# Ce fichier couvre :
#   - Le drawdown historique
#   - La Value at Risk (VaR) historique
#   - La Conditional Value at Risk (CVaR) — Expected Shortfall
#
# Le risque de queue (tail risk) est le risque de résultats extrêmes,
# notamment sur le côté négatif d'une distribution de rendements.
#
# Approches abordées :
#   1. Historical Drawdown
#   2. Value at Risk (VaR)
#   3. Conditional Value at Risk (CVaR)
#
# Actif utilisé : USO (United States Oil Fund ETF)
# Période       : 2007-01-01 → 2017-12-31


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf



TICKER = "USO"
START = "2007-01-01"
END = "2017-12-31"

# Téléchargement des prix
data = yf.download(TICKER, start=START, end=END,
                   auto_adjust=False, progress=False)
adj_close = data["Adj Close"].squeeze().sort_index()

# Rendements journaliers discrets
stock_returns = adj_close.pct_change().dropna()

print(f"Actif : {TICKER} | Période : {START} → {END}")
print(f"\nNombre d'observations : {len(stock_returns)}")
print(stock_returns.head())


# 1. Drawdown historique
#
# Le drawdown mesure la perte en pourcentage par rapport
# au plus haut cumulatif historique.
#
#   Drawdown = (r_t / RM) - 1
#
#   r_t : rendement cumulatif à l'instant t
#   RM  : running maximum — le plus haut cumulatif atteint jusqu'à t
#
#   Un drawdown de -0.80 signifie que l'actif a perdu 80%
#   par rapport à son plus haut historique à cet instant.
#
#   Implémentation :
#       cum_rets    = (1 + stock_returns).cumprod()
#       running_max = np.maximum.accumulate(cum_rets)
#       running_max[running_max < 1] = 1   # plancher à 1 (prix initial)
#       drawdown    = cum_rets / running_max - 1

# Rendements cumulatifs
cum_rets = (1 + stock_returns).cumprod()

# Running maximum (plus haut cumulatif glissant)
running_max = np.maximum.accumulate(cum_rets)
running_max[running_max < 1] = 1  # on plancher à 1 (base 100%)

# Calcul du drawdown
drawdown = cum_rets / running_max - 1

print("\nDrawdown historique :")
print(drawdown.head())
print(f"\nDrawdown maximum (pire perte) : {drawdown.min():.2%}")

# Visualisation
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(drawdown.index, drawdown, color="steelblue", linewidth=0.8, label=TICKER)
ax.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color="steelblue")
ax.set_title(f"Drawdown historique — {TICKER}")
ax.set_ylabel("Drawdown")
ax.set_xlabel("Date")
ax.legend()
plt.tight_layout()
plt.show()


# 2. Value at Risk (VaR) historique
#
# La VaR est un seuil, à un niveau de confiance donné,
# en dessous duquel les pertes ne dépasseront pas (historiquement).
#
#   VaR(95) = -2.3% signifie :
#       "On est certain à 95% que la perte journalière
#       ne dépassera pas -2.3% sur la base des données historiques."
#
#   VaR est couramment calculée aux quantiles 90, 95, 99 et 99.9.
#
#   Calcul en Python :
#       var_level = 95
#       var_95 = np.percentile(stock_returns, 100 - var_level)
#
#   Note : np.percentile(x, 5) renvoie le 5ème percentile,
#   soit le seuil en dessous duquel se trouvent les 5% pires rendements.

var_level = 95
var_95 = np.percentile(stock_returns, 100 - var_level)

print(f"\n--- VaR historique ({var_level}%) ---")
print(f"  VaR({var_level}) journalière : {var_95:.4f} ({var_95:.2%})")

# Visualisation de la distribution avec la VaR
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(stock_returns * 100, bins=50, density=True,
        color="steelblue", edgecolor="white", linewidth=0.3)
ax.axvline(var_95 * 100, color="red", linewidth=1.5,
           label=f"VaR {var_level}: {var_95*100:.2f}%")
ax.set_title(f"Distribution historique des rendements de {TICKER}")
ax.set_xlabel("Rendements (%)")
ax.set_ylabel("Probabilité")
ax.legend()
plt.tight_layout()
plt.show()


# 3. CVaR historique (Conditional Value at Risk — Expected Shortfall)
#
# La CVaR est une estimation de la perte moyenne
# dans les (1 - x)% pires scénarios.
#
#   CVaR(95) = -2.5% signifie :
#       "Dans les 5% pires cas, la perte moyenne est de -2.5%."
#
#   La CVaR est toujours plus négative que la VaR au même niveau :
#       CVaR(95) < VaR(95)   (en valeur absolue, CVaR > VaR)
#
#   Calcul en Python :
#       var_95  = np.percentile(stock_returns, 100 - var_level)
#       cvar_95 = stock_returns[stock_returns <= var_95].mean()
#
#   On filtre les rendements inférieurs ou égaux au seuil VaR,
#   puis on calcule leur moyenne — c'est l'espérance de perte
#   conditionnelle à se trouver dans la queue gauche.

cvar_95 = stock_returns[stock_returns <= var_95].mean()

print(f"\n--- CVaR historique ({var_level}%) ---")
print(f"  VaR({var_level})  journalière : {var_95:.4f} ({var_95:.2%})")
print(f"  CVaR({var_level}) journalière : {cvar_95:.4f} ({cvar_95:.2%})")
print(f"  Ecart VaR → CVaR          : {(cvar_95 - var_95):.4f}")

# Visualisation VaR vs CVaR
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(stock_returns * 100, bins=50, density=True,
        color="steelblue", edgecolor="white", linewidth=0.3)
ax.axvline(var_95 * 100, color="red", linewidth=1.5,
           label=f"VaR  {var_level}: {var_95*100:.2f}%")
ax.axvline(cvar_95 * 100, color="blue", linewidth=1.5,
           label=f"CVaR {var_level}: {cvar_95*100:.2f}%")
ax.set_title(f"Distribution historique des rendements de {TICKER} — VaR vs CVaR")
ax.set_xlabel("Rendements (%)")
ax.set_ylabel("Probabilité")
ax.legend()
plt.tight_layout()
plt.show()
