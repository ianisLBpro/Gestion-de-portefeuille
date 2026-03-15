'''
3.1_Markowitz_Optimisation — scipy.optimize

Ce fichier complète le fichier 3_Portefeuille_de_Markowitz en remplaçant
la simulation Monte Carlo par une optimisation convexe exacte via scipy.optimize.

Différence fondamentale :
    Monte Carlo  : approximation par énumération de 100 000 portefeuilles aléatoires
                   => le MSR/GMV trouvé est le meilleur des tirages, pas le vrai optimum
    scipy.optimize : résolution analytique du problème d'optimisation
                   => résultat exact, reproductible, et extensible avec des contraintes réelles

Les deux approches doivent converger vers des résultats très proches.
Un écart important signalerait que 100 000 simulations étaient insuffisantes.

Actifs utilisés : AAPL, MSFT, AMZN, JPM, JNJ | Période : 2016-2026
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize



TICKERS = ["AAPL", "MSFT", "AMZN", "JPM", "JNJ"]
NUM_STOCKS = len(TICKERS)
START = "2016-01-01"
END   = "2026-01-01"

# Téléchargement des données
data = yf.download(TICKERS, start=START, end=END,
                   auto_adjust=False, progress=False)
adj_close = data["Adj Close"].sort_index()

# Rendements journaliers
StockReturns = adj_close.pct_change().dropna()

# Paramètres annualisés
mean_returns_annual = StockReturns[TICKERS].mean() * 252
cov_mat_annual      = StockReturns[TICKERS].cov() * 252

# Poids des 3 portefeuilles custom (pour comparaison finale)
portfolio_weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
portfolio_weights_ew   = np.repeat(1 / NUM_STOCKS, NUM_STOCKS)
market_capitalizations = np.array([2500, 2200, 1800, 500, 400])
mcap_weights           = market_capitalizations / sum(market_capitalizations)

print("Rendements annualisés moyens :")
for t, r in zip(TICKERS, mean_returns_annual):
    print(f"  {t} : {r:.2%}")




# 1. Fonctions utilitaires

'''
Deux fonctions sont nécessaires pour l'optimisation :

    portfolio_performance(weights) :
        Calcule le rendement et la volatilité annualisés d'un portefeuille.
            p_ret = wT · mean_returns_annual
            p_vol = sqrt(wT · Sigma · w)

    neg_sharpe(weights) :
        Retourne le Sharpe ratio négatif.
        scipy.optimize minimise => on minimise -Sharpe pour maximiser Sharpe.
        rf = 0 pour simplifier (cohérent avec le fichier 3).

Contraintes communes à toutes les optimisations :
    - Somme des poids = 1       (contrainte d'égalité)
    - Chaque poids entre 0 et 1 (pas de vente à découvert)
'''

def portfolio_performance(weights):
    # Rendement annualisé du portefeuille
    p_ret = np.dot(weights, mean_returns_annual)
    # Volatilité annualisée via formule matricielle
    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat_annual, weights)))
    return p_ret, p_vol

def neg_sharpe(weights, rf=0):
    p_ret, p_vol = portfolio_performance(weights)
    return -(p_ret - rf) / p_vol

def portfolio_volatility(weights):
    _, p_vol = portfolio_performance(weights)
    return p_vol

# Contraintes et bornes communes
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
bounds      = [(0, 1)] * NUM_STOCKS

# Point de départ : portefeuille équipondéré
w0 = np.repeat(1 / NUM_STOCKS, NUM_STOCKS)




# 2. Optimisation MSR (Max Sharpe Ratio)

'''
MSR — Maximisation du Sharpe Ratio :

    max  (R_p - RF) / sigma_p
    s.t. sum(w) = 1
         0 <= w_i <= 1

scipy.optimize.minimize résout ce problème en cherchant
le vecteur w qui minimise -Sharpe sous les contraintes données.

Algorithme : SLSQP (Sequential Least Squares Programming)
    => adapté aux problèmes avec contraintes d'égalité et de borne.

    result = minimize(neg_sharpe, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints)
    result.x => poids optimaux
'''

result_msr = minimize(neg_sharpe, w0,
                      method="SLSQP",
                      bounds=bounds,
                      constraints=constraints)

MSR_weights_opt = result_msr.x
MSR_ret_opt, MSR_vol_opt = portfolio_performance(MSR_weights_opt)
MSR_sharpe_opt  = -result_msr.fun

print("\nMSR (Max Sharpe Ratio) — scipy.optimize")
for ticker, w in zip(TICKERS, MSR_weights_opt):
    print(f"  {ticker} : {w:.2%}")
print(f"  Volatilité : {MSR_vol_opt:.4%} | Rendement : {MSR_ret_opt:.4%} | Sharpe : {MSR_sharpe_opt:.4f}")




# 3. Optimisation GMV (Global Minimum Volatility)

'''
GMV — Minimisation de la volatilité :

    min  sqrt(wT · Sigma · w)
    s.t. sum(w) = 1
         0 <= w_i <= 1

Ici on minimise directement la volatilité du portefeuille,
sans contrainte sur le rendement.
La volatilité étant plus prévisible que les rendements,
le GMV est souvent plus robuste en pratique que le MSR.
'''

result_gmv = minimize(portfolio_volatility, w0,
                      method="SLSQP",
                      bounds=bounds,
                      constraints=constraints)

GMV_weights_opt = result_gmv.x
GMV_ret_opt, GMV_vol_opt = portfolio_performance(GMV_weights_opt)
GMV_sharpe_opt  = (GMV_ret_opt) / GMV_vol_opt

print("\nGMV (Global Minimum Volatility)")
for ticker, w in zip(TICKERS, GMV_weights_opt):
    print(f"  {ticker} : {w:.2%}")
print(f"  Volatilité : {GMV_vol_opt:.4%} | Rendement : {GMV_ret_opt:.4%} | Sharpe : {GMV_sharpe_opt:.4f}")




# 4. Frontière efficiente — optimisation exacte

'''
Frontière efficiente par optimisation :
Pour chaque niveau de rendement cible entre le GMV et le rendement max,
on minimise la volatilité sous contrainte de rendement.

    min  sqrt(wT · Sigma · w)
    s.t. sum(w) = 1
         0 <= w_i <= 1
         R_p = target_return    (contrainte supplémentaire)

On obtient ainsi la frontière efficiente exacte point par point,
contrairement à Monte Carlo qui l'approxime par un nuage de points aléatoires.
'''

# Plage de rendements cibles entre GMV et rendement max
target_returns = np.linspace(GMV_ret_opt, mean_returns_annual.max(), 100)
frontier_vols  = []

for target in target_returns:
    constraints_ef = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w, t=target: portfolio_performance(w)[0] - t}
    ]
    res = minimize(portfolio_volatility, w0,
                   method="SLSQP",
                   bounds=bounds,
                   constraints=constraints_ef)
    frontier_vols.append(res.fun if res.success else np.nan)

frontier_vols = np.array(frontier_vols)

# Coordonnées des 3 portefeuilles custom
port_vol_custom = np.sqrt(np.dot(portfolio_weights.T,    np.dot(cov_mat_annual, portfolio_weights)))
port_vol_ew_    = np.sqrt(np.dot(portfolio_weights_ew.T, np.dot(cov_mat_annual, portfolio_weights_ew)))
port_vol_mcap_  = np.sqrt(np.dot(mcap_weights.T,         np.dot(cov_mat_annual, mcap_weights)))

port_ret_custom = np.dot(portfolio_weights,    mean_returns_annual)
port_ret_ew_    = np.dot(portfolio_weights_ew, mean_returns_annual)
port_ret_mcap_  = np.dot(mcap_weights,         mean_returns_annual)

fig, ax = plt.subplots(figsize=(10, 7))

# Frontière efficiente avec scipy.optimize
ax.plot(frontier_vols, target_returns,
        color="black", linewidth=2, label="Frontière efficiente")

# MSR 
ax.scatter(MSR_vol_opt, MSR_ret_opt, color="red",  s=120, zorder=5,
           label=f"MSR  (Sharpe={MSR_sharpe_opt:.2f})")
# GMV
ax.scatter(GMV_vol_opt, GMV_ret_opt, color="blue", s=120, zorder=5,
           label=f"GMV  (Vol={GMV_vol_opt:.2%})")

# Portefeuilles custom
ax.scatter(port_vol_custom, port_ret_custom, color="steelblue",  s=100, zorder=5, label="Pondéré custom")
ax.scatter(port_vol_ew_,    port_ret_ew_,    color="darkorange", s=100, zorder=5, label="Équipondéré")
ax.scatter(port_vol_mcap_,  port_ret_mcap_,  color="seagreen",   s=100, zorder=5, label="Market-Cap")

ax.set_title("Frontière Efficiente de Markowitz avec scipy.optimize")
ax.set_xlabel("Volatilité annualisée")
ax.set_ylabel("Rendement annualisé espéré")
ax.legend()
plt.tight_layout()
plt.show()




