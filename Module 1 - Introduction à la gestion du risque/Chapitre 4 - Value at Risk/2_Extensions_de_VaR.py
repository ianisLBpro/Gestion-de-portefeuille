"""
2_Extensions_de_VaR.py

Auteur     : Ianis Le Berre
Module     : Module 1 — Introduction à la gestion du risque
             Chapitre 4 — Value at Risk
Librairies : numpy, pandas, matplotlib, yfinance, scipy
"""

# Extensions de la Value at Risk
#
# Ce fichier couvre :
#   - Comparaison des quantiles VaR et CVaR (90, 95, 99)
#   - VaR paramétrique (distribution normale)
#   - Mise à l'échelle temporelle de la VaR (règle de la racine carrée du temps)
#
# Actif utilisé : USO (United States Oil Fund ETF)
# Période       : 2007-01-01 → 2017-12-31


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm



TICKER = "USO"
START = "2007-01-01"
END = "2017-12-31"

# Téléchargement des prix
data = yf.download(TICKER, start=START, end=END,
                   auto_adjust=False, progress=False)
adj_close = data["Adj Close"].squeeze().sort_index()
stock_returns = adj_close.pct_change().dropna()

print(f"Actif : {TICKER} | Période : {START} → {END}")
print(f"Nombre d'observations : {len(stock_returns)}\n")


# 1. Comparaison des quantiles VaR et CVaR
#
# La VaR et la CVaR sont couramment calculées à plusieurs niveaux
# de confiance pour avoir une vue complète du profil de risque :
#
#   Niveau 90 → capture les 10% pires rendements
#   Niveau 95 → capture les  5% pires rendements
#   Niveau 99 → capture les  1% pires rendements
#
#   Plus le niveau de confiance est élevé, plus la VaR est négative
#   (on capture des événements de plus en plus rares et extrêmes).
#
#   La CVaR est toujours plus négative que la VaR au même niveau :
#   elle représente la moyenne des pertes au-delà du seuil VaR.

var_levels = [90, 95, 99]

print("--- Comparaison VaR et CVaR par quantile ---")
print(f"{'Niveau':<10} {'VaR':>10} {'CVaR':>10}")
for level in var_levels:
    var = np.percentile(stock_returns, 100 - level)
    cvar = stock_returns[stock_returns <= var].mean()
    print(f"  {level}%      {var*100:>8.2f}%  {cvar*100:>8.2f}%")

# Récupération des valeurs pour la visualisation
var_90 = np.percentile(stock_returns, 10)
var_95 = np.percentile(stock_returns, 5)
var_99 = np.percentile(stock_returns, 1)
cvar_90 = stock_returns[stock_returns <= var_90].mean()
cvar_95 = stock_returns[stock_returns <= var_95].mean()
cvar_99 = stock_returns[stock_returns <= var_99].mean()

# Visualisation des quantiles VaR et CVaR
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(stock_returns * 100, bins=50, density=True,
        color="steelblue", edgecolor="white", linewidth=0.3)

# VaR — lignes solides
ax.axvline(var_90 * 100, color="red", linewidth=1.5, label=f"VaR 90: {var_90*100:.2f}%")
ax.axvline(var_95 * 100, color="green", linewidth=1.5, label=f"VaR 95: {var_95*100:.2f}%")
ax.axvline(var_99 * 100, color="blue", linewidth=1.5, label=f"VaR 99: {var_99*100:.2f}%")

# CVaR — lignes pointillées
ax.axvline(cvar_90 * 100, color="red", linewidth=1.5, linestyle="--",
           label=f"CVaR 90: {cvar_90*100:.2f}%")
ax.axvline(cvar_95 * 100, color="green", linewidth=1.5, linestyle="--",
           label=f"CVaR 95: {cvar_95*100:.2f}%")
ax.axvline(cvar_99 * 100, color="blue", linewidth=1.5, linestyle="--",
           label=f"CVaR 99: {cvar_99*100:.2f}%")

ax.set_title(f"Distribution historique des rendements de {TICKER} — VaR & CVaR par quantile")
ax.set_xlabel("Rendements (%)")
ax.set_ylabel("Probabilité")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()


# 2. VaR paramétrique
#
# Les valeurs historiques empiriques sont celles qui se sont réellement produites.
# Mais comment simuler la probabilité d'un événement qui ne s'est jamais produit ?
#
#   => On échantillonne depuis une loi de probabilité (ici la loi normale).
#
#   VaR paramétrique (loi normale) :
#       mu  = np.mean(stock_returns)
#       std = np.std(stock_returns)
#       VaR = norm.ppf(confidence_level, mu, std)
#
#   norm.ppf(q, mu, std) est la fonction quantile (inverse de la CDF)
#   de la loi normale : elle renvoie le rendement en dessous duquel
#   se trouvent q% des observations selon la loi N(mu, std).
#
#   confidence_level = 0.05 correspond à un VaR(95) :
#   on cherche le seuil en dessous duquel se trouvent 5% des rendements.
#
#   Avantage : permet d'explorer des scénarios jamais observés historiquement.
#   Limite   : suppose que les rendements suivent une loi normale,
#              ce qui sous-estime souvent les queues épaisses (fat tails).

mu = np.mean(stock_returns)
std = np.std(stock_returns)

print("\n--- VaR paramétrique (loi normale) ---")
for level in var_levels:
    confidence_level = (100 - level) / 100
    var_param = norm.ppf(confidence_level, mu, std)
    var_hist = np.percentile(stock_returns, 100 - level)
    print(f"  VaR({level})  paramétrique : {var_param*100:.4f}% | historique : {var_hist*100:.4f}%")


# 3. Mise à l'échelle temporelle — règle de la racine carrée du temps
#
# La VaR journalière peut être projetée sur un horizon plus long
# grâce à la règle de la racine carrée du temps :
#
#   VaR(T jours) = VaR(1 jour) × √T
#
#   Cette règle repose sur l'hypothèse que les rendements sont
#   indépendants et identiquement distribués (i.i.d.) au fil du temps.
#
#   Exemple : VaR(95) à 1 jour = -2.35%
#             VaR(95) à 5 jours = -2.35% × √5 ≈ -5.25%
#
#   Plus l'horizon s'allonge, plus la VaR augmente en valeur absolue,
#   mais de façon décroissante (croissance en racine carrée, pas linéaire).

forecast_horizons = [1, 5, 10, 21, 63, 252]  # 1j, 1sem, 2sem, 1mois, 1trim, 1an

print(f"\n--- Mise à l'échelle temporelle — VaR(95) ---")
print(f"{'Horizon (j)':<15} {'VaR(95) scalée':>15}")
for t in forecast_horizons:
    var_scaled = var_95 * np.sqrt(t)
    print(f"  {t:<13} {var_scaled*100:>13.4f}%")

# Visualisation de la VaR scalée dans le temps
horizons_range = np.arange(1, 101)
var_scaled_list = var_95 * np.sqrt(horizons_range)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(horizons_range, np.abs(var_scaled_list) * 100,
        color="steelblue", linewidth=1.8)
ax.set_title(f"VaR(95) mise à l'échelle temporelle — {TICKER}")
ax.set_xlabel("Horizon T+i (jours)")
ax.set_ylabel("VaR(95) prévisionnelle (%)")
plt.tight_layout()
plt.show()
