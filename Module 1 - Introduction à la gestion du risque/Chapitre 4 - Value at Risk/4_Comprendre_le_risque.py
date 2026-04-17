"""
4_Comprendre_le_risque.py

Auteur     : Ianis Le Berre
Module     : Module 1 — Introduction à la gestion du risque
             Chapitre 4 — Value at Risk
Librairies : numpy, pandas, matplotlib, yfinance, scipy
"""

# Synthèse complète des outils de mesure du risque
#
# Ce fichier synthétise l'ensemble des concepts abordés dans le chapitre 4 :
#   - Drawdown maximum
#   - VaR et CVaR historiques (90, 95, 99%)
#   - VaR paramétrique (loi normale)
#   - VaR Monte Carlo
#   - Scaling temporel VaR(95)
#   - Dashboard de visualisation synthétique
#
# Actifs : AAPL, MSFT, AMZN, JPM, JNJ | Période : 2016-2026
# Pondérations : 30% AAPL | 25% MSFT | 20% AMZN | 15% JPM | 10% JNJ


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm



TICKERS = ["AAPL", "MSFT", "AMZN", "JPM", "JNJ"]
START = "2016-01-01"
END = "2026-12-31"
portfolio_weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

# Téléchargement des prix
data = yf.download(TICKERS, start=START, end=END,
                   auto_adjust=False, progress=False)
if isinstance(data.columns, pd.MultiIndex):
    adj_close = data["Adj Close"]
else:
    adj_close = data[["Adj Close"]]
adj_close = adj_close.sort_index()

# Rendements journaliers discrets
stock_returns = adj_close.pct_change().dropna()

# Rendements du portefeuille pondéré
port_ret = stock_returns[TICKERS].mul(portfolio_weights, axis=1).sum(axis=1)

# Paramètres empiriques du portefeuille
mu = np.mean(port_ret)
std = np.std(port_ret, ddof=1)

print("=" * 60)
print("  Synthèse des outils de mesure du risque")
print(f"  Portefeuille : {TICKERS}")
print(f"  Période      : {START} → {END}")
print("=" * 60)
print(f"\nMu journalier   : {mu:.6f} ({mu*252:.2%} annualisé)")
print(f"Std journalière : {std:.6f} ({std*np.sqrt(252):.2%} annualisée)")




# 1. Drawdown maximum
#
# Le drawdown mesure la perte cumulée par rapport
# au plus haut historique atteint par le portefeuille.
#
#   Drawdown = (r_t / RM) - 1

cum_rets = (1 + port_ret).cumprod()
running_max = np.maximum.accumulate(cum_rets)
running_max[running_max < 1] = 1
drawdown = cum_rets / running_max - 1
max_dd = drawdown.min()

print(f"\n[Drawdown]")
print(f"  Drawdown maximum       : {max_dd:.2%}")
print(f"  Date du drawdown max   : {drawdown.idxmin().date()}")




# 2. VaR et CVaR historiques aux 3 niveaux
#
# VaR historique  : seuil en dessous duquel se trouvent les x% pires rendements.
# CVaR historique : moyenne des rendements au-delà de ce seuil (expected shortfall).

print(f"\n[VaR & CVaR historiques — portefeuille]")
print(f"  {'Niveau':<8} {'VaR':>10} {'CVaR':>10}")
for level in [90, 95, 99]:
    var = np.percentile(port_ret, 100 - level)
    cvar = port_ret[port_ret <= var].mean()
    print(f"  {level}%      {var*100:>8.2f}%  {cvar*100:>8.2f}%")




# 3. VaR paramétrique
#
# La VaR paramétrique suppose que les rendements suivent une loi normale N(mu, std).
# Elle permet d'estimer des pertes jamais observées historiquement.
#
#   VaR = norm.ppf(confidence_level, mu, std)

print(f"\n[VaR paramétrique — loi normale]")
print(f"  {'Niveau':<8} {'VaR param':>12} {'VaR hist':>12}")
for level in [90, 95, 99]:
    var_param = norm.ppf((100 - level) / 100, mu, std)
    var_hist = np.percentile(port_ret, 100 - level)
    print(f"  {level}%      {var_param*100:>10.2f}%  {var_hist*100:>10.2f}%")




# 4. VaR Monte Carlo
#
# La VaR Monte Carlo génère N trajectoires de rendements simulés
# depuis N(mu, std) et en extrait le quantile de perte souhaité.

np.random.seed(0)
N = 1000
T = 252
sim_returns = np.random.normal(mu, std, (N, T))

print(f"\n[VaR Monte Carlo — {N} simulations]")
print(f"  {'Niveau':<8} {'VaR MC':>10} {'VaR hist':>10}")
for level in [90, 95, 99]:
    var_mc = np.percentile(sim_returns, 100 - level)
    var_hist = np.percentile(port_ret, 100 - level)
    print(f"  {level}%      {var_mc*100:>8.2f}%  {var_hist*100:>8.2f}%")




# 5. Scaling temporel VaR(95)
#
# VaR(T jours) = VaR(1 jour) x √T
# Repose sur l'hypothèse de rendements i.i.d.

var_95 = np.percentile(port_ret, 5)
print(f"\n[Scaling temporel — VaR(95) = {var_95*100:.2f}% à 1 jour]")
for t in [1, 5, 10, 21, 63, 252]:
    print(f"  Horizon {t:>4}j : {var_95 * np.sqrt(t) * 100:.2f}%")

print("\n" + "=" * 60)
print("  Fin du récapitulatif")
print("=" * 60)





# 6. Visualisation synthétique — dashboard risque portefeuille

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Synthèse du risque — Portefeuille AAPL / MSFT / AMZN / JPM / JNJ",
             fontsize=13, fontweight="bold")

# Panel 1 : Drawdown historique du portefeuille
ax = axes[0, 0]
ax.plot(drawdown.index, drawdown * 100, color="steelblue", linewidth=0.8)
ax.fill_between(drawdown.index, drawdown * 100, 0, alpha=0.25, color="steelblue")
ax.set_title("Drawdown historique — portefeuille")
ax.set_ylabel("Drawdown (%)")

# Panel 2 : Distribution des rendements + VaR + CVaR
ax = axes[0, 1]
ax.hist(port_ret * 100, bins=60, density=True,
        color="steelblue", edgecolor="white", linewidth=0.2)
for level, color in zip([90, 95, 99], ["gold", "orange", "red"]):
    v = np.percentile(port_ret, 100 - level)
    c = port_ret[port_ret <= v].mean()
    ax.axvline(v * 100, color=color, linewidth=1.5,
               label=f"VaR {level}: {v*100:.2f}%")
    ax.axvline(c * 100, color=color, linewidth=1.5, linestyle="--",
               label=f"CVaR {level}: {c*100:.2f}%")
ax.set_title("VaR & CVaR — Distribution des rendements")
ax.set_xlabel("Rendements (%)")
ax.set_ylabel("Probabilité")
ax.legend(fontsize=7)

# Panel 3 : Scaling temporel VaR(95)
ax = axes[1, 0]
horizons = np.arange(1, 253)
ax.plot(horizons, np.abs(var_95 * np.sqrt(horizons)) * 100,
        color="steelblue", linewidth=1.8)
ax.set_title("VaR(95) mise à l'échelle temporelle")
ax.set_xlabel("Horizon (jours)")
ax.set_ylabel("VaR(95) projetée (%)")

# Panel 4 : Monte Carlo — faisceau de trajectoires du portefeuille
ax = axes[1, 1]
S0 = 100  # base 100
trajec = np.zeros((T, 200))
np.random.seed(0)
for i in range(200):
    r = np.random.normal(mu, std, T) + 1
    trajec[:, i] = S0 * r.cumprod()
ax.plot(trajec, linewidth=0.3, alpha=0.4)
ax.set_title("Monte Carlo — 200 trajectoires simulées")
ax.set_xlabel("Jour")
ax.set_ylabel("Valeur du portefeuille (base 100)")

plt.tight_layout()
plt.show()