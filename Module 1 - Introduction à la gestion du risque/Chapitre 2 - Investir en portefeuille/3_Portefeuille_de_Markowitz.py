"""
3_Portefeuille_de_Markowitz.py

Auteur     : Ianis Le Berre
Module     : Module 1 — Introduction à la gestion du risque
             Chapitre 2 — Investir en portefeuille
Librairies : numpy, pandas, matplotlib, yfinance
"""

# Théorie Moderne du Portefeuille (Markowitz, 1952) :
# En simulant un grand nombre de portefeuilles aléatoires et en traçant (volatilité, rendement) pour chacun,
# on obtient un nuage délimité par la frontière efficiente : l'ensemble des portefeuilles qui maximisent
# le rendement pour un niveau de risque donné.
#
# Sharpe Ratio (1966) :
#
#     S = (Ra - rf) / sigma_a
#
#     Ra      : rendement du portefeuille
#     rf      : taux sans risque
#     sigma_a : volatilité du portefeuille
#
#     Un Sharpe élevé => bon rendement par unité de risque pris.
#
# Deux portefeuilles remarquables sur la frontière efficiente :
#
#     MSR (Max Sharpe Ratio) :
#         Maximise le ratio rendement/risque.
#         => Portefeuille tangent à la Capital Allocation Line.
#         Limite : repose sur les rendements passés, difficiles à prédire.
#
#     GMV (Global Minimum Volatility) :
#         Minimise la volatilité sans contrainte sur le rendement.
#         La volatilité est plus prévisible que les rendements
#         => souvent plus robuste en pratique que le MSR.
#
# Règle : choisir un portefeuille sur le bord supérieur gauche du nuage.
#
# Actifs utilisés : AAPL, MSFT, AMZN, JPM, JNJ | Période : 2016-2026


import numpy as np
import pandas as pd
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
stock_returns = adj_close.pct_change().dropna()
print("Rendements journaliers :")
print(stock_returns.head())


# Poids choisis personnellement pour le portefeuille pondéré (somme = 1)
portfolio_weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

# Poids pour portefeuille équipondéré (EW)
portfolio_weights_ew = np.repeat(1 / NUM_STOCKS, NUM_STOCKS)

# Poids par capitalisation boursière (MCAP)
market_capitalizations = np.array([2500, 2200, 1800, 500, 400])  # AAPL, MSFT, AMZN, JPM, JNJ
mcap_weights = market_capitalizations / np.sum(market_capitalizations)


# Matrice de covariance
cov_mat = stock_returns[TICKERS].cov()

# Annualisation de la matrice de covariance
cov_mat_annual = cov_mat * 252




# 1. Calcul de la volatilité d'un portefeuille via la formule matricielle

# Simulation Monte Carlo :
# On génère N_PORTFOLIOS portefeuilles aléatoires avec des poids positifs dont la somme = 1,
# on calcule rendement et volatilité annualisés pour chacun via la formule matricielle :
#
#     p_ret = np.dot(w, mean_returns_annual)
#     p_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat_annual, w)))
#
# Plus N est grand, mieux on approxime la frontière efficiente.
# Ici 100 000 portefeuilles pour un nuage dense.

np.random.seed(42)
N_PORTFOLIOS = 100_000

# Rendements moyens annualisés de chaque actif
mean_returns_annual = stock_returns[TICKERS].mean() * 252

# Stockage des résultats : une ligne par portefeuille simulé
results = np.zeros((N_PORTFOLIOS, NUM_STOCKS + 2))

for i in range(N_PORTFOLIOS):
    # Poids aléatoires positifs dont la somme = 1
    w = np.random.random(NUM_STOCKS)
    w = w / np.sum(w)

    p_ret = np.dot(w, mean_returns_annual)
    p_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat_annual, w)))

    results[i, :NUM_STOCKS] = w
    results[i, NUM_STOCKS] = p_vol
    results[i, NUM_STOCKS + 1] = p_ret

df_portfolios = pd.DataFrame(results, columns=TICKERS + ["Volatility", "Returns"])
print("\nExemple de portefeuilles simulés :")
print(df_portfolios.head())




# 2. Sélection du MSR (Max Sharpe Ratio)

# MSR : portefeuille avec le Sharpe le plus élevé.
#
#     df.sort_values("Sharpe", ascending=False).iloc[0]
#
# On utilise rf = 0 pour simplifier (pas de taux sans risque).

risk_free = 0
df_portfolios["Sharpe"] = (df_portfolios["Returns"] - risk_free) / df_portfolios["Volatility"]

MSR = df_portfolios.sort_values(by=["Sharpe"], ascending=False)
MSR_weights = np.array(MSR.iloc[0, 0:NUM_STOCKS])
MSR_vol     = MSR.iloc[0]["Volatility"]
MSR_ret     = MSR.iloc[0]["Returns"]

print("\nPortefeuille MSR (Max Sharpe Ratio) :")
for ticker, w in zip(TICKERS, MSR_weights):
    print(f"  {ticker} : {w:.2%}")
print(f"  Volatilité : {MSR_vol:.4%} | Rendement : {MSR_ret:.4%} | Sharpe : {MSR.iloc[0]['Sharpe']:.4f}")




# 3. Sélection du GMV (Global Minimum Volatility)

# GMV : portefeuille avec la volatilité la plus faible.
#
#     df.sort_values("Volatility", ascending=True).iloc[0]

GMV = df_portfolios.sort_values(by=["Volatility"], ascending=True)
GMV_weights = np.array(GMV.iloc[0, 0:NUM_STOCKS])
GMV_vol     = GMV.iloc[0]["Volatility"]
GMV_ret     = GMV.iloc[0]["Returns"]

print("\nPortefeuille GMV (Global Minimum Volatility) :")
for ticker, w in zip(TICKERS, GMV_weights):
    print(f"  {ticker} : {w:.2%}")
print(f"  Volatilité : {GMV_vol:.4%} | Rendement : {GMV_ret:.4%} | Sharpe : {GMV.iloc[0]['Sharpe']:.4f}")




# 4. Visualisation de la frontière efficiente de Markowitz

# Le nuage de points représente les 100 000 portefeuilles simulés.
# Chaque point est coloré selon son Sharpe ratio.
# MSR et GMV sont les deux points remarquables sur la frontière efficiente.

fig, ax = plt.subplots(figsize=(10, 7))

scatter = ax.scatter(df_portfolios["Volatility"], df_portfolios["Returns"],
                     c=df_portfolios["Sharpe"], cmap="viridis", alpha=0.5, s=5)
plt.colorbar(scatter, ax=ax, label="Sharpe Ratio")

ax.scatter(MSR_vol, MSR_ret, color="red",  s=100, zorder=5,
           label=f"MSR  (Sharpe={MSR.iloc[0]['Sharpe']:.2f})")
ax.scatter(GMV_vol, GMV_ret, color="blue", s=100, zorder=5,
           label=f"GMV  (Vol={GMV_vol:.2%})")

ax.set_title("Frontière Efficiente de Markowitz")
ax.set_xlabel("Volatilité annualisée")
ax.set_ylabel("Rendement annualisé espéré")
ax.legend()
plt.tight_layout()
plt.show()

# Frontière efficiente avec les portefeuilles custom comme points de référence

fig, ax = plt.subplots(figsize=(10, 7))

scatter = ax.scatter(df_portfolios["Volatility"], df_portfolios["Returns"],
                     c=df_portfolios["Sharpe"], cmap="viridis", alpha=0.5, s=5)
plt.colorbar(scatter, ax=ax, label="Sharpe Ratio")

ax.scatter(MSR_vol, MSR_ret, color="red",  s=100, zorder=5,
           label=f"MSR  (Sharpe={MSR.iloc[0]['Sharpe']:.2f})")
ax.scatter(GMV_vol, GMV_ret, color="blue", s=100, zorder=5,
           label=f"GMV  (Vol={GMV_vol:.2%})")

# Ajout des 3 portefeuilles custom comme points de référence
port_vol_custom = np.sqrt(np.dot(portfolio_weights.T,    np.dot(cov_mat_annual, portfolio_weights)))
port_vol_ew     = np.sqrt(np.dot(portfolio_weights_ew.T, np.dot(cov_mat_annual, portfolio_weights_ew)))
port_vol_mcap   = np.sqrt(np.dot(mcap_weights.T,         np.dot(cov_mat_annual, mcap_weights)))

port_ret_custom = np.dot(portfolio_weights,    mean_returns_annual)
port_ret_ew     = np.dot(portfolio_weights_ew, mean_returns_annual)
port_ret_mcap   = np.dot(mcap_weights,         mean_returns_annual)

ax.scatter(port_vol_custom, port_ret_custom, color="steelblue", s=100, zorder=5, label="Pondéré custom")
ax.scatter(port_vol_ew,     port_ret_ew,     color="darkorange", s=100, zorder=5, label="Équipondéré")
ax.scatter(port_vol_mcap,   port_ret_mcap,   color="seagreen",   s=100, zorder=5, label="Market-Cap")

ax.set_title("Frontière Efficiente — Comparaison des portefeuilles custom")
ax.set_xlabel("Volatilité annualisée")
ax.set_ylabel("Rendement annualisé espéré")
ax.legend()
plt.tight_layout()
plt.show()