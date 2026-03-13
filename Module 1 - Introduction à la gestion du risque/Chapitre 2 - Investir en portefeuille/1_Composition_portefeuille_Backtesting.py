'''
3_Composition_Portefeuille_Backtesting

Ce chapitre couvre :
    - Le calcul du rendement d'un portefeuille multi-actifs pondéré
    - Les portefeuilles équipondérés et pondérés par capitalisation boursière
    - Le backtesting via les rendements cumulés

    
On choisis volontairement une pondération aléatoires qui sera 0.30, 0.25, 0.20, 0.15, 0.10 pour les 5 actifs. 
L'objectif est de montrer comment calculer le rendement d'un portefeuille avec n'importe quelle pondération, 
puis de comparer avec un portefeuille équipondéré (poids égaux) pour illustrer l'impact de la composition 
du portefeuille sur les rendements.

Actifs utilisés : AAPL, MSFT, AMZN, JPM, JNJ | Période : 2016-2026
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


TICKERS = ["AAPL", "MSFT", "AMZN", "JPM", "JNJ"]
NUM_STOCKS = len(TICKERS)

# Téléchargement des données de prix pour les 5 actifs
data = yf.download(TICKERS, start="2016-01-01", end="2026-12-31", auto_adjust=False, progress=False)
if isinstance(data.columns, pd.MultiIndex):
    # On garde uniquement le niveau "Adj Close" pour tous les tickers
    adj_close = data["Adj Close"]
else:
    adj_close = data[["Adj Close"]]

adj_close = adj_close.sort_index()

# Calcul des rendements discrets pour chaque actif
# pct_change() calcule (P_t - P_{t-1}) / P_{t-1} pour chaque colonne
StockReturns = adj_close.pct_change().dropna()
print("Rendements journaliers :")
print(StockReturns.head())




# 1. Rendement d'un portefeuille pondéré

'''
Formule du rendement de portefeuille :

    R_p = R_a1 * w_a1 + R_a2 * w_a2 + ... + R_an * w_an

Où :
    R_p   : rendement du portefeuille à la date t
    R_an  : rendement de l'actif n à la date t
    w_an  : poids de l'actif n dans le portefeuille (somme des poids = 1)

En Python :
    StockReturns.mul(weights, axis=1).sum(axis=1)

    - .mul(weights, axis=1) : multiplie chaque colonne par le poids correspondant
    - .sum(axis=1)          : somme ligne par ligne => rendement journalier du portefeuille
'''

# Poids personnalisés (somme = 1)
portfolio_weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

# Calcul du rendement journalier du portefeuille pondéré
port_ret = StockReturns.mul(portfolio_weights, axis=1).sum(axis=1)
StockReturns["Portfolio"] = port_ret

print(f"\nRendement journalier du portefeuille pondéré :")
print(StockReturns["Portfolio"].head())




# 1.1. Visualisation des rendements journaliers du portefeuille pondéré

'''
Backtesting visuel : observer le comportement journalier du portefeuille sur toute la période d'analyse.

    StockReturns["Portfolio"].plot()
'''

fig, ax = plt.subplots(figsize=(12, 5))
StockReturns["Portfolio"].plot(ax=ax, linewidth=0.8, color="steelblue")
ax.set_title("Rendements journaliers du portefeuille pondéré")
ax.set_xlabel("Date")
ax.set_ylabel("Rendement journalier")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.tight_layout()
plt.show()




# 1.2. Rendements cumulés du portefeuille pondéré

''' 
On utilise .cumprod() pour calculer les rendements cumulés, ce qui permet de visualiser la performance totale du portefeuille.

    CumulativeReturns = (1 + StockReturns).cumprod() - 1

    - (1 + R)    : transforme chaque rendement en facteur de croissance
    - .cumprod() : produit cumulatif de ces facteurs jour après jour
    - - 1        : revient en % de gain depuis l'origine (0 = point de départ)

Pourquoi .cumprod() et pas .cumsum() ?
    .cumsum() additionne les rendements : approximation valable uniquement
    pour de très petits rendements sur courte période.
    .cumprod() est exact car chaque jour la base est le capital de la veille
    (intérêts composés).

    Exemple :
        +1% j1, -0.5% j2 =>
        .cumprod() : (1.01 * 0.995) - 1 = +0.5045%
        .cumsum()  :  0.01 + (-0.005)   = +0.5000%   (légère erreur)
'''

fig, ax = plt.subplots(figsize=(12, 6))
((1 + StockReturns["Portfolio"]).cumprod() - 1).plot(ax=ax, linewidth=1.5, color="steelblue")
ax.set_title("Rendements cumulés du portefeuille pondéré")
ax.set_xlabel("Date")
ax.set_ylabel("Rendement cumulé")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.tight_layout()
plt.show()




# 2. Portefeuille équipondéré (Equally Weighted)

'''
Portefeuille équipondéré (Equal Weight) :
Chaque actif reçoit le même poids : w_i = 1 / N

    np.repeat(1/numstocks, numstocks) => [0.2, 0.2, 0.2, 0.2, 0.2]

    .iloc[:,0:numstocks] sélectionne uniquement les colonnes des actifs (exclut la colonne "Portfolio" déjà ajoutée à StockReturns)
'''

numstocks = NUM_STOCKS
portfolio_weights_ew = np.repeat(1 / numstocks, numstocks)

port_ret_ew = StockReturns.iloc[:, 0:numstocks].mul(portfolio_weights_ew, axis=1).sum(axis=1)
StockReturns["Portfolio_EW"] = port_ret_ew

print(f"\nPoids équipondérés : {portfolio_weights_ew}")
print(f"\nRendement journalier du portefeuille équipondéré :")
print(StockReturns["Portfolio_EW"].head())




# 2.1. Visualisation des rendements journaliers du portefeuille équipondéré

fig, ax = plt.subplots(figsize=(12, 5))
StockReturns["Portfolio_EW"].plot(ax=ax, linewidth=0.8, color="darkorange")
ax.set_title("Rendements journaliers du portefeuille équipondéré")
ax.set_xlabel("Date")
ax.set_ylabel("Rendement journalier")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.tight_layout()
plt.show()




# 3. Rendements cumulés : Pondéré vs Équipondéré

'''
Comme expliqué en 1.2. Comparer les rendements cumulés du portefeuille pondéré 
et du portefeuille équipondéré permet de visualiser l'impact du choix de pondération sur la performance long terme.
'''

CumulativeReturns = ((1 + StockReturns[["Portfolio", "Portfolio_EW"]]).cumprod() - 1)

fig, ax = plt.subplots(figsize=(12, 6))
CumulativeReturns[["Portfolio", "Portfolio_EW"]].plot(ax=ax, linewidth=1.5)
ax.set_title("Rendements cumulés : Pondéré vs Équipondéré")
ax.set_xlabel("Date")
ax.set_ylabel("Rendement cumulé")
ax.legend(["Pondéré", "Équipondéré"])
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.tight_layout()
plt.show()




# 4. Portefeuille pondéré par capitalisation boursière (Market-Cap)

'''
Capitalisation boursière :
Valeur totale des actions d'une entreprise en circulation sur le marché.

    Market cap = Prix * Nombre d'actions en circulation

Poids par market cap :

    w_mcap_n = mcap_n / somme(mcap_i)

Logique : les plus grandes entreprises ont plus de poids.
C'est la méthode utilisée par les grands indices (S&P 500, CAC 40, MSCI World).

    mcap_weights = market_capitalizations / sum(market_capitalizations)

Note : les market caps varient chaque jour en réalité.
On utilise ici des valeurs fixes approximatives (ordre de grandeur 2020).
'''

# Capitalisations boursières approximatives en milliards USD
market_capitalizations = np.array([2500, 2200, 1800, 500, 400])  # AAPL, MSFT, AMZN, JPM, JNJ

mcap_weights = market_capitalizations / sum(market_capitalizations)

print("\nPoids par capitalisation boursière :")
for ticker, weight in zip(TICKERS, mcap_weights):
    print(f"  {ticker} : {weight:.2%}")

port_ret_mcap = StockReturns.iloc[:, 0:numstocks].mul(mcap_weights, axis=1).sum(axis=1)
StockReturns["Portfolio_MCAP"] = port_ret_mcap




# 4.1. Visualisation des rendements journaliers du portefeuille Market-Cap

fig, ax = plt.subplots(figsize=(12, 5))
StockReturns["Portfolio_MCAP"].plot(ax=ax, linewidth=0.8, color="seagreen")
ax.set_title("Rendements journaliers du portefeuille pondéré par capitalisation boursière")
ax.set_xlabel("Date")
ax.set_ylabel("Rendement journalier")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.tight_layout()
plt.show()




# 5. Comparaison des 3 stratégies de pondération — rendements cumulés

'''
Backtesting comparatif des 3 approches de pondération :

    - Pondéré (custom)  : 30% AAPL, 25% MSFT, 20% AMZN, 15% JPM, 10% JNJ
    - Équipondéré (EW)  : 20% par actif
    - Market-Cap (MCAP) : poids proportionnels à la capitalisation boursière

La comparaison des rendements cumulés permet de visualiser
l'impact du choix de pondération sur la performance long terme.
'''

CumulativeReturns_3 = ((1 + StockReturns[["Portfolio", "Portfolio_EW", "Portfolio_MCAP"]]).cumprod() - 1)

fig, ax = plt.subplots(figsize=(12, 6))
CumulativeReturns_3["Portfolio"].plot(ax=ax, linewidth=1.5, color="steelblue")
CumulativeReturns_3["Portfolio_EW"].plot(ax=ax, linewidth=1.5, color="darkorange")
CumulativeReturns_3["Portfolio_MCAP"].plot(ax=ax, linewidth=1.5, color="seagreen")
ax.set_title("Rendements cumulés : comparaison des 3 stratégies de pondération")
ax.set_xlabel("Date")
ax.set_ylabel("Rendement cumulé")
ax.legend(["Pondéré", "Équipondéré", "Market-Cap"])
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.tight_layout()
plt.show()

# Rendement total final de chaque stratégie
print("\nRendement total sur la période :")
for col, label in zip(["Portfolio", "Portfolio_EW", "Portfolio_MCAP"],
                      ["Pondéré", "Équipondéré", "Market-Cap"]):
    total = CumulativeReturns_3[col].iloc[-1]
    print(f"  {label} : {total:.2%}")
