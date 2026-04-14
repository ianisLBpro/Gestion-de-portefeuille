"""
3_Theorie_moderne_du_portefeuille.py

Auteur     : Ianis Le Berre
Module     : Module 2 — Gestion quantitative des risques
             Chapitre 1 — Base du risque
Librairies : numpy, pandas, matplotlib, yfinance, scipy
"""

# Théorie Moderne du Portefeuille (MPT)
#
# Ce fichier couvre :
#     - Le trade-off risque/rendement
#     - L'appétit au risque de l'investisseur
#     - La théorie moderne du portefeuille (Markowitz, 1952)
#     - La frontière efficiente
#     - Le portefeuille à variance minimale
#
# Théorie Moderne du Portefeuille (MPT) — Markowitz (1952, Nobel 1990) :
#     Un portefeuille efficient est celui qui génère le rendement espéré
#     le plus élevé pour un niveau de risque donné.
#
#     Le vecteur de poids optimal w* résout :
#         max_w  E[w^T * r]
#         s.c.   w^T * Σ * w = σ²_cible
#
#     En faisant varier σ²_cible, on trace la frontière efficiente :
#     l'ensemble des couples (risque, rendement) des portefeuilles optimaux.
#
# Actifs utilisés : AAPL, MSFT, AMZN, JPM, JNJ | Période : 2016-2026
# Benchmark : S&P 500 (^GSPC) — non utilisé dans ce fichier (pas de régression CAPM)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize


TICKERS = ["AAPL", "MSFT", "AMZN", "JPM", "JNJ"]
START = "2016-01-01"
END = "2026-12-31"
portfolio_weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

# Téléchargement des prix
data = yf.download(TICKERS, start=START, end=END,
                   auto_adjust=False, progress=False)
if isinstance(data.columns, pd.MultiIndex):
    adj_close = data["Adj Close"].sort_index()

# Rendements journaliers discrets
returns = adj_close[TICKERS].pct_change().dropna()

# Paramètres annualisés
mean_returns = returns.mean() * 252   # rendements espérés annualisés
cov_matrix = returns.cov() * 252      # matrice de covariance annualisée

print(f"Actifs   : {TICKERS}")
print(f"Période  : {START} → {END}")
print("\nRendements espérés annualisés :")
for ticker, ret in zip(TICKERS, mean_returns):
    print(f"  {ticker:<6} : {ret:.2%}")


# 1. Trade-off risque/rendement

# Le trade-off risque/rendement est le principe fondamental de la finance :
#     - Une incertitude plus grande (plus de risque) doit être compensée
#       par un rendement espéré plus élevé.
#     - Le rendement moyen historique est utilisé comme proxy
#       du rendement futur espéré.
#
#     Appétit au risque de l'investisseur :
#         Pour chaque niveau de risque σ, l'investisseur définit
#         le rendement minimum qu'il exige.
#         Cela crée un ensemble de couples (σ, R) qui caractérisent
#         son profil de risque.
#
#     Modifier les pondérations du portefeuille = ajuster l'exposition au risque.

# Rendement et volatilité du portefeuille custom
port_ret_custom = returns.dot(portfolio_weights)
ret_custom = port_ret_custom.mean() * 252
vol_custom = np.sqrt(portfolio_weights @ cov_matrix @ portfolio_weights)
sharpe_custom = ret_custom / vol_custom   # sans taux sans risque pour simplifier

print("\n--- Portefeuille custom (pondérations fixes) ---")
print(f"  Rendement annualisé : {ret_custom:.2%}")
print(f"  Volatilité annualisée : {vol_custom:.2%}")
print(f"  Ratio de Sharpe (approx.) : {sharpe_custom:.4f}")


# 2. Simulation Monte Carlo de portefeuilles aléatoires

# Avant de calculer la frontière efficiente analytiquement,
# on peut visualiser l'espace risque/rendement en simulant
# un grand nombre de portefeuilles avec des pondérations aléatoires.
#
#     Pour chaque simulation :
#         1. Tirer des poids aléatoires normalisés (somme = 1)
#         2. Calculer le rendement espéré : w^T * mu
#         3. Calculer la volatilité       : sqrt(w^T * Cov * w)
#         4. Calculer le ratio de Sharpe  : rendement / volatilité
#
#     Le nuage de points obtenu matérialise l'ensemble des portefeuilles
#     réalisables. La frontière efficiente en est l'enveloppe supérieure gauche.

np.random.seed(42)
N_simulations = 5000
sim_returns = np.zeros(N_simulations)
sim_vols = np.zeros(N_simulations)
sim_sharpes = np.zeros(N_simulations)
sim_weights = np.zeros((N_simulations, len(TICKERS)))

for i in range(N_simulations):
    # Pondérations aléatoires normalisées
    w = np.random.random(len(TICKERS))
    w /= w.sum()
    sim_weights[i] = w

    r = w @ mean_returns
    v = np.sqrt(w @ cov_matrix @ w)

    sim_returns[i] = r
    sim_vols[i] = v
    sim_sharpes[i] = r / v

print(f"\n--- Simulation Monte Carlo ({N_simulations} portefeuilles) ---")
print(f"  Rendement min/max : {sim_returns.min():.2%} / {sim_returns.max():.2%}")
print(f"  Volatilité min/max : {sim_vols.min():.2%} / {sim_vols.max():.2%}")
print(f"  Sharpe max         : {sim_sharpes.max():.4f}")


# 3. Frontière efficiente — optimisation numérique

# La frontière efficiente est calculée en résolvant un problème
# d'optimisation pour chaque niveau de rendement cible :
#
#     Minimiser  : w^T * Σ * w     (variance du portefeuille)
#     Sous contraintes :
#         w^T * mu  = R_cible      (rendement cible atteint)
#         sum(w)    = 1            (les poids somment à 1)
#         w_i >= 0                 (pas de vente à découvert)
#
#     scipy.optimize.minimize avec méthode SLSQP permet de résoudre
#     ce problème de programmation quadratique sous contraintes.

def portfolio_volatility(weights, cov):
    # Volatilité annualisée d'un portefeuille.
    return np.sqrt(weights @ cov @ weights)


def portfolio_return(weights, mu):
    # Rendement annualisé espéré d'un portefeuille.
    return weights @ mu


# Contrainte : somme des poids = 1
constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
# Bornes : poids entre 0 et 1 (pas de vente à découvert)
bounds = tuple((0, 1) for _ in TICKERS)
# Poids initiaux : équipondéré
w0 = np.array([1 / len(TICKERS)] * len(TICKERS))

# Balayage des niveaux de rendement cibles
target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 100)
frontier_vols = []
frontier_rets = []

for target in target_returns:
    # Contrainte additionnelle : rendement = target
    constraints_eff = constraints + [
        {"type": "eq", "fun": lambda w, t=target: portfolio_return(w, mean_returns) - t}
    ]
    result = minimize(portfolio_volatility, w0,
                      args=(cov_matrix,),
                      method="SLSQP",
                      bounds=bounds,
                      constraints=constraints_eff)
    if result.success:
        frontier_vols.append(result.fun)
        frontier_rets.append(target)

frontier_vols = np.array(frontier_vols)
frontier_rets = np.array(frontier_rets)

# Portefeuille à variance minimale (point le plus à gauche de la frontière)
min_vol_idx = np.argmin(frontier_vols)
min_vol = frontier_vols[min_vol_idx]
min_vol_ret = frontier_rets[min_vol_idx]

print("\n--- Portefeuille à variance minimale ---")
print(f"  Volatilité minimale    : {min_vol:.2%}")
print(f"  Rendement associé      : {min_vol_ret:.2%}")

# Portefeuille à Sharpe maximum
sharpe_frontier = frontier_rets / frontier_vols
max_sharpe_idx = np.argmax(sharpe_frontier)
max_sharpe_vol = frontier_vols[max_sharpe_idx]
max_sharpe_ret = frontier_rets[max_sharpe_idx]

print("\n--- Portefeuille à Sharpe maximum ---")
print(f"  Volatilité    : {max_sharpe_vol:.2%}")
print(f"  Rendement     : {max_sharpe_ret:.2%}")
print(f"  Sharpe        : {sharpe_frontier[max_sharpe_idx]:.4f}")


# 4. Visualisation de la frontière efficiente

# Le graphique (volatilité, rendement) permet de visualiser :
#     - Le nuage des portefeuilles simulés (espace réalisable)
#     - La frontière efficiente (enveloppe supérieure optimale)
#     - Le portefeuille à variance minimale (point le plus à gauche)
#     - Le portefeuille à Sharpe maximum (meilleur ratio risque/rendement)
#     - Le portefeuille custom (pondérations fixes initiales)
#
#     Un investisseur rationnel choisit toujours un portefeuille
#     sur la frontière efficiente, jamais en dessous.
#     Son choix sur la frontière dépend de son appétit au risque.

fig, ax = plt.subplots(figsize=(10, 7))

# Nuage Monte Carlo coloré par Sharpe
sc = ax.scatter(sim_vols * 100, sim_returns * 100,
                c=sim_sharpes, cmap="viridis",
                s=4, alpha=0.5, label="Portefeuilles simulés")
plt.colorbar(sc, ax=ax, label="Ratio de Sharpe")

# Frontière efficiente
ax.plot(frontier_vols * 100, frontier_rets * 100,
        color="red", linewidth=2.5, label="Frontière efficiente")

# Portefeuille à variance minimale
ax.scatter(min_vol * 100, min_vol_ret * 100,
           color="blue", s=120, zorder=5,
           label=f"Variance minimale ({min_vol:.1%}, {min_vol_ret:.1%})")

# Portefeuille à Sharpe maximum
ax.scatter(max_sharpe_vol * 100, max_sharpe_ret * 100,
           color="gold", edgecolors="black", s=120, zorder=5,
           label=f"Sharpe max ({max_sharpe_vol:.1%}, {max_sharpe_ret:.1%})")

# Portefeuille custom
ax.scatter(vol_custom * 100, ret_custom * 100,
           color="orange", marker="D", s=100, zorder=5,
           label=f"Portefeuille custom ({vol_custom:.1%}, {ret_custom:.1%})")

ax.set_title("Frontière efficiente — Portefeuille AAPL/MSFT/AMZN/JPM/JNJ")
ax.set_xlabel("Volatilité annualisée (%)")
ax.set_ylabel("Rendement espéré annualisé (%)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
