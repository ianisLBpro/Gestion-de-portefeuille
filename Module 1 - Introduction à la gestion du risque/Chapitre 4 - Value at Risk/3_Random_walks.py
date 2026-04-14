"""
3_Random_walks.py

Auteur     : Ianis Le Berre
Module     : Module 1 — Introduction à la gestion du risque
             Chapitre 4 — Value at Risk
Librairies : numpy, pandas, matplotlib, yfinance
"""

# Marches aléatoires et simulation Monte Carlo
#
# Ce fichier couvre :
#   - La marche aléatoire (random walk) en finance
#   - Simulation d'une trajectoire de prix
#   - Simulation Monte Carlo multi-trajectoires
#   - VaR Monte Carlo
#
# Une marche aléatoire en finance modélise l'évolution d'un prix
# comme une séquence de chocs aléatoires successifs tirés
# depuis une distribution de probabilité.
#
#   Contrairement aux valeurs empiriques historiques (qui se sont
#   réellement produites), la simulation Monte Carlo permet d'explorer
#   des scénarios qui ne se sont jamais produits historiquement.
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
stock_returns = adj_close.pct_change().dropna()

# Paramètres empiriques utilisés pour les simulations
mu = np.mean(stock_returns)
std = np.std(stock_returns)

print(f"Actif : {TICKER} | Période : {START} → {END}")
print(f"Mu journalier  : {mu:.6f} ({mu*252:.2%} annualisé)")
print(f"Std journalière: {std:.6f} ({std*np.sqrt(252):.2%} annualisée)\n")


# 1. Marche aléatoire — une seule trajectoire
#
# En finance, une marche aléatoire simple modélise un prix futur
# comme le produit cumulatif de rendements tirés aléatoirement :
#
#   S_t = S_0 * ∏(1 + r_i)   pour i = 1, ..., T
#
#   où r_i ~ N(mu, std) sont des rendements journaliers simulés.
#
#   Implémentation :
#       rand_rets         = np.random.normal(mu, std, T) + 1
#       forecasted_values = S0 * rand_rets.cumprod()
#
#   On ajoute 1 aux rendements pour obtenir des facteurs de croissance
#   (ex. +2% → facteur 1.02), puis on en calcule le produit cumulatif
#   pour reconstruire la trajectoire de prix.

T = 252   # horizon de simulation : 1 an de trading
S0 = 10   # prix initial arbitraire

np.random.seed(42)  # reproductibilité
rand_rets = np.random.normal(mu, std, T) + 1
forecasted_values = S0 * rand_rets.cumprod()

print("--- Marche aléatoire — une trajectoire ---")
print(f"  Prix initial S0  : {S0}")
print(f"  Prix final simulé: {forecasted_values[-1]:.4f}")
print(f"  Rendement total  : {(forecasted_values[-1]/S0 - 1):.2%}")

# Visualisation d'une trajectoire
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(forecasted_values, color="steelblue", linewidth=1.2)
ax.set_title(f"Marche aléatoire — {TICKER} (1 trajectoire, T={T} jours)")
ax.set_xlabel("Jour")
ax.set_ylabel("Prix simulé")
plt.tight_layout()
plt.show()


# 2. Simulation Monte Carlo — trajectoires multiples
#
# Une simulation Monte Carlo consiste à répéter un grand nombre
# de marches aléatoires indépendantes pour cartographier
# l'ensemble des trajectoires possibles d'un actif.
#
#   Pour N simulations de T jours :
#       for i in range(N):
#           rand_rets = np.random.normal(mu, std, T) + 1
#           forecasted_values = S0 * rand_rets.cumprod()
#
#   On obtient un faisceau de N trajectoires qui matérialise
#   l'incertitude sur l'évolution future du prix.
#
#   Les trajectoires basses représentent les scénarios défavorables,
#   les trajectoires hautes les scénarios favorables.
#   La dispersion du faisceau croît avec le temps (effet √T).

N = 500  # nombre de simulations

np.random.seed(0)
simulations = np.zeros((T, N))  # matrice T lignes x N colonnes

for i in range(N):
    rand_rets = np.random.normal(mu, std, T) + 1
    simulations[:, i] = S0 * rand_rets.cumprod()

# Statistiques sur les prix finaux simulés
final_prices = simulations[-1, :]
print(f"\n--- Monte Carlo ({N} simulations, T={T} jours) ---")
print(f"  Prix final médian   : {np.median(final_prices):.4f}")
print(f"  Prix final min      : {np.min(final_prices):.4f}")
print(f"  Prix final max      : {np.max(final_prices):.4f}")
print(f"  Ecart-type final    : {np.std(final_prices):.4f}")

# Visualisation du faisceau de trajectoires
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(simulations, linewidth=0.4, alpha=0.5)
ax.set_title(f"Simulation Monte Carlo — {TICKER} ({N} trajectoires, T={T} jours)")
ax.set_xlabel("Jour")
ax.set_ylabel("Prix simulé")
plt.tight_layout()
plt.show()


# 3. VaR Monte Carlo
#
# La VaR Monte Carlo est calculée en appliquant np.percentile
# à l'ensemble des rendements simulés sur toutes les trajectoires.
#
#   sim_returns contient tous les rendements journaliers simulés
#   (N simulations * T jours), soit N * T observations.
#
#   var_95 = np.percentile(sim_returns, 5)
#
#   Avantage par rapport à la VaR historique :
#     - Peut capturer des pertes jamais observées historiquement
#     - Paramétrable (on peut modifier mu et std indépendamment)
#     - Base naturelle pour la VaR de produits dérivés (options, etc.)
#
#   Limite :
#     - Sensible aux hypothèses de distribution (normalité ici)
#     - Sous-estime les queues épaisses si la loi normale est inadaptée

# Génération de tous les rendements simulés (rendements bruts sans +1)
np.random.seed(0)
sim_returns = []
for i in range(N):
    rand_rets = np.random.normal(mu, std, T)
    sim_returns.append(rand_rets)

sim_returns = np.array(sim_returns)  # shape : (N, T)

# VaR Monte Carlo aux trois quantiles habituels
print(f"\n--- VaR Monte Carlo ({N} simulations) ---")
for level in [90, 95, 99]:
    var_mc = np.percentile(sim_returns, 100 - level)
    var_hist = np.percentile(stock_returns, 100 - level)
    print(f"  VaR({level})  Monte Carlo : {var_mc*100:.4f}% | historique : {var_hist*100:.4f}%")
