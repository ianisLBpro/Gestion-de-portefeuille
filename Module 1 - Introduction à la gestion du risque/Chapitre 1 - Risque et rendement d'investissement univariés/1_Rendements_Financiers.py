"""
1_Rendements_Financiers.py

Auteur     : Ianis Le Berre
Module     : Module 1 — Introduction à la gestion du risque
             Chapitre 1 — Risque et rendement d'investissement univariés
Librairies : numpy, pandas, matplotlib, yfinance
"""

# Financial returns
#
# Ce chapitre couvre :
#     - Le chargement de données de prix via yfinance
#     - Le calcul des rendements discrets et logarithmiques
#     - La visualisation de la série temporelle et de la distribution
#
# Actif utilisé : AAPL (Apple) | Période : 2015-2026


# Importation des librairies nécessaires

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf




# 1. Charger les données de prix d'un actif financier depuis Yahoo Finance

# Description des paramètres utilisés dans yf.download :
# - TICKER : Le ou les tickers de l'action à analyser
# - yf.download() : Fonction pour télécharger les données de prix depuis Yahoo Finance
# - start="2015-01-01" : La date de début de la période d'analyse
# - end="2024-12-31" : La date de fin de la période d'analyse
# - auto_adjust=False : Conserve Close et Adj Close séparément
#     => On utilise "Adj Close" manuellement pour les rendements
#     => Avec auto_adjust=True, "Close" serait déjà ajusté mais "Adj Close" disparaîtrait du DataFrame
# - progress=False : Pour éviter l'affichage de la barre de progression
#
# - if isinstance(data.columns, pd.MultiIndex) : Vérifie si les colonnes sont un MultiIndex
#     (cas où yfinance retourne des données avec plusieurs niveaux de colonnes)
# - data.columns.get_level_values(0) : Récupère le premier niveau des colonnes si c'est un MultiIndex
# - data.sort_index() : S'assure que les données sont triées par date
# - print(data.head()) : Affiche les premières lignes du DataFrame

TICKER = "AAPL"
data = yf.download(TICKER, start="2015-01-01", end="2026-12-31", auto_adjust=False, progress=False)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
data = data.sort_index()
print(f"Données de prix chargées pour {TICKER} :")
print(data.head())


# Dans un fichier CSV, on aurait utilisé :
# StockPrices = pd.read_csv("StockData.csv", parse_dates=["Date"])
# StockPrices = StockPrices.sort_values(by="Date")
# StockPrices.set_index("Date", inplace=True) 




# 2. Calculer les rendements discrets et logarithmiques 

# Rendement discret (simple return) :
#
#     R_t = (P_t - P_{t-1}) / P_{t-1}
#
# Où :
#     P_t     = prix à la période actuelle
#     P_{t-1} = prix à la période précédente
#
# Propriétés des rendements discrets :
# - S'agrègent sur les ACTIFS (cross-sectional) : R_portfolio = somme(w_i * R_i)
# - Ne s'additionnent pas dans le temps (biais de composition)
# En Python :
# - data["Adj Close"].pct_change() : calcule automatiquement (P_t - P_{t-1}) / P_{t-1} pour chaque ligne
#
#
# Rendement logarithmique (log return) :
#
#     r_t = ln(P_t / P_{t-1})
#
# ou de façon équivalente :
#
#     r_t = ln(P_t) - ln(P_{t-1})
#
# Propriétés des rendements logarithmiques :
# - S'agrègent dans le TEMPS (time-additive) : r_annuel = r_j1 + r_j2 + ... + r_j252
# - Supposent une distribution normale, ce qui facilite les calculs statistiques
# En Python :
# - np.log(data["Adj Close"] / data["Adj Close"].shift(1))
#
#
# À savoir :
# - Pour de petites variations de prix, log return ≈ discrete return
# - Pour de grandes variations, le log return est toujours inférieur au discrete return
#   car ln(1 + x) < x pour x > 0
# - On travaille sur "Adj Close" (cours ajusté) et non "Close" :
#   un dividende ou un split crée un saut artificiel dans "Close" qui fausserait
#   les rendements calculés

data["Returns"] = data["Adj Close"].pct_change()
data["LogReturns"] = np.log(data["Adj Close"] / data["Adj Close"].shift(1))
returns     = data["Returns"].dropna()
log_returns = data["LogReturns"].dropna()
print("\nRendements calculés (discrets et logarithmiques) :")
print(data[["Adj Close", "Returns", "LogReturns"]].head(8))




# 3. Visualiser et comparer les rendements discrets et logarithmiques

# plt.subplots(nrows, ncols, figsize=(largeur, hauteur)) :
#   - Crée une figure avec une grille de sous-graphiques (axes)
#   - nrows=2, ncols=2 : grille 2x2 => 4 graphiques au total
#   - figsize=(14, 10) : dimensions de la figure en pouces
#   - Retourne fig (la figure globale) et axes (tableau numpy 2D des sous-graphiques)
#
# Accès aux sous-graphiques : axes[ligne, colonne] (indices 0-based)
#   - axes[0, 0] : haut-gauche    axes[0, 1] : haut-droite
#   - axes[1, 0] : bas-gauche     axes[1, 1] : bas-droite
#
# axes[i, j].plot(x, y, ...) : courbe linéaire
#   - color : couleur de la courbe (nom CSS ou code hex)
#   - linewidth : épaisseur du trait
#
# axes[i, j].hist(data, bins=n, ...) : histogramme
#   - bins : nombre d'intervalles (plus il est élevé, plus la distribution est fine)
#   - color : couleur des barres
#
# axes[i, j].set_title() / set_xlabel() / set_ylabel() : titres et labels des axes
#
# plt.tight_layout() : ajuste automatiquement les marges pour éviter les chevauchements
# plt.show()         : affiche la figure

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Ligne 1 — Rendements discrets
# Graphique haut-gauche : série temporelle des rendements discrets
axes[0, 0].plot(returns.index, returns, color="steelblue", linewidth=0.6)
axes[0, 0].set_title(f"Rendements discrets journaliers de {TICKER}")
axes[0, 0].set_xlabel("Date")
axes[0, 0].set_ylabel("Rendements discrets")

# Graphique haut-droite : histogramme des rendements discrets
axes[0, 1].hist(returns, bins=75, color="seagreen")
axes[0, 1].set_title(f"Distribution des rendements discrets de {TICKER}")
axes[0, 1].set_xlabel("Rendements discrets")
axes[0, 1].set_ylabel("Fréquence")

# Ligne 2 — Rendements logarithmiques
# Graphique bas-gauche : série temporelle des rendements logarithmiques
axes[1, 0].plot(log_returns.index, log_returns, color="darkorange", linewidth=0.6)
axes[1, 0].set_title(f"Rendements logarithmiques journaliers de {TICKER}")
axes[1, 0].set_xlabel("Date")
axes[1, 0].set_ylabel("Rendements logarithmiques")

# Graphique bas-droite : histogramme des rendements logarithmiques
axes[1, 1].hist(log_returns, bins=75, color="tomato")
axes[1, 1].set_title(f"Distribution des rendements logarithmiques de {TICKER}")
axes[1, 1].set_xlabel("Rendements logarithmiques")
axes[1, 1].set_ylabel("Fréquence")

plt.tight_layout()
plt.show()
