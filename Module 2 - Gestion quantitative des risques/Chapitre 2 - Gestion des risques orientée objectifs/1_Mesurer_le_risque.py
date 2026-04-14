"""
Mesurer_le_risque.py

Auteur     : Ianis Le Berre
Module     : Module 2 — Gestion quantitative des risques
             Chapitre 2 — Gestion des risques orientée objectifs
Librairies : numpy, pandas, matplotlib, scipy.stats
"""

# Mesurer le risque (Measuring Risk)
#
# Ce fichier couvre :
#     - La distribution des pertes (loss distribution)
#     - La Value at Risk (VaR) : empirique et paramétrique
#     - La Conditional Value at Risk (CVaR) / Expected Shortfall
#     - La visualisation des mesures de risque
#
# Distribution des pertes :
#     La perte d'un portefeuille dépend de facteurs de risque aléatoires
#     (taux de change, prix des actifs, etc.).
#     On ne peut PAS borner la perte maximale avec 100% de certitude.
#     → On utilise un niveau de confiance (ex : 95%) pour fixer une borne.
#
# Value at Risk (VaR) :
#     Statistique mesurant la perte maximale à un niveau de confiance α.
#         VaR(α) = quantile(α) de la distribution des pertes
#     Niveaux typiques : 95%, 99%, 99.5%
#     Propriété : VaR_95 < VaR_99 < VaR_99.5 (croissant avec α)
#     Calcul :
#         - Paramétrique : norm.ppf(α, loc, scale)
#         - Empirique    : np.quantile(draws, α)
#
# Conditional Value at Risk (CVaR) / Expected Shortfall :
#     Perte espérée sachant que la perte dépasse le VaR.
#     C'est la moyenne des pertes dans la queue de la distribution.
#         CVaR(α) = (1/(1-α)) × ∫ de VaR(α) à +∞ : x·f(x)dx
#     Le CVaR est toujours ≥ VaR.
#     Calcul :
#         - Paramétrique : (1/(1-α)) × norm.expect(lambda x: x, loc, scale, lb=VaR)
#         - Empirique    : loss[loss >= VaR].mean()
#
# Pourquoi le CVaR est préféré au VaR :
#     Le CVaR est une espérance sur toutes les pertes DÉPASSANT le VaR,
#     c'est-à-dire précisément la queue de la distribution.
#     La forme et la masse de la queue influencent directement le CVaR.
#     Le VaR est un simple seuil statique : il ne dit rien sur l'ampleur
#     des pertes au-delà de ce seuil.
#     Le CVaR nécessite le VaR (lb dans l'intégrale), donc le VaR reste utile.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm



# 1. Distribution des pertes — Exemple Forex

# Exemple Forex :
#     - Portefeuille initial = USD 100
#     - Facteur de risque r = taux de change EUR/USD
#     - Valeur en EUR si r EUR = 1 USD : USD 100 × r = EUR 100 × r
#     - Perte = EUR 100 - EUR 100 × r = EUR 100 × (1 - r)
#     - Les réalisations aléatoires de r génèrent la loss distribution.

np.random.seed(42)

VALEUR_INITIALE = 100  # USD
N_SIMULATIONS = 1000

taux_change = np.random.normal(loc=1.0, scale=0.15, size=N_SIMULATIONS)
pertes_forex = VALEUR_INITIALE * (1 - taux_change)

plt.figure(figsize=(10, 6))
plt.hist(pertes_forex, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
plt.xlabel("EUR 100 × (1 − r)")
plt.ylabel("Nombre de pertes de cette ampleur")
plt.title("Distribution des pertes — Portefeuille Forex")
plt.axvline(x=0, color="red", linestyle="--", alpha=0.5, label="Perte nulle")
plt.legend()
plt.tight_layout()
plt.show()




# 2. Value at Risk (VaR) — Calcul empirique

# Étapes pour dériver le VaR :
#     1. Spécifier le niveau de confiance (ex : 0.95)
#     2. Créer une Series pandas des observations de pertes
#     3. Calculer loss.quantile(α) au niveau spécifié
#     4. VaR = résultat du .quantile()
#     5. Alternative : scipy.stats.<dist>.ppf(α) (percent point function)

loss = pd.Series(pertes_forex)
VaR_95 = loss.quantile(0.95)

print("\nVaR EMPIRIQUE — Portefeuille Forex")
print(f"  VaR 95% = {VaR_95:.4f} EUR")

# Illustration avec N(1,3) comme dans le cours
np.random.seed(42)
pertes_N13 = np.random.normal(loc=1, scale=3, size=1000)
loss_N13 = pd.Series(pertes_N13)

VaR_95_N13 = loss_N13.quantile(0.95)
VaR_99_N13 = loss_N13.quantile(0.99)
VaR_995_N13 = loss_N13.quantile(0.995)

print("\nDistribution N(1,3), 1000 tirages")
print(f"  VaR 95%   = {VaR_95_N13:.2f}")
print(f"  VaR 99%   = {VaR_99_N13:.2f}")
print(f"  VaR 99.5% = {VaR_995_N13:.2f}")
print("  → Le VaR augmente avec le niveau de confiance")




# 3. Visualisation du VaR à différents niveaux de confiance

# On superpose les lignes verticales du VaR sur l'histogramme
# des pertes pour illustrer la progression :
#     VaR_95 < VaR_99 < VaR_99.5

plt.figure(figsize=(10, 6))
plt.hist(pertes_N13, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
plt.axvline(VaR_95_N13, color="red", linewidth=2,
            label=f"VaR 0.95 = {VaR_95_N13:.2f}")
plt.axvline(VaR_99_N13, color="green", linewidth=2,
            label=f"VaR 0.99 = {VaR_99_N13:.2f}")
plt.axvline(VaR_995_N13, color="black", linewidth=2,
            label=f"VaR 0.995 = {VaR_995_N13:.2f}")
plt.xlabel("Perte")
plt.ylabel("Nombre de simulations")
plt.title("Histogramme de 1000 tirages N(1,3) — niveaux de VaR")
plt.legend()
plt.tight_layout()
plt.show()




# 4. Exercice DataCamp — VaR for the Normal distribution

# Objectif : calculer le VaR à 95% et 99% sur une distribution N(0,1)
# avec les deux méthodes (paramétrique et empirique).
#
# norm.ppf(α)        → inverse de la CDF, donne le VaR exact théorique
# np.quantile(x, α)  → quantile empirique sur un échantillon de taille n
#
# Résultats attendus :
#     95% VaR ≈ 1.6449
#     99% VaR ≈ 2.3263

# Méthode paramétrique : norm.ppf()
VaR_95_ppf = norm.ppf(0.95)

# Méthode empirique : np.quantile() sur 100 000 tirages
draws = norm.rvs(size=100000)
VaR_99_quantile = np.quantile(draws, 0.99)

# Comparaison
print("\nEXERCICE — VaR for the Normal distribution")
print(f"  95% VaR (norm.ppf)    : {VaR_95_ppf:.4f}")
print(f"  99% VaR (np.quantile) : {VaR_99_quantile:.4f}")

# Visualisation
plt.figure(figsize=(10, 6))
plt.hist(draws, bins=100, alpha=0.7, color="steelblue", edgecolor="black")
plt.axvline(x=VaR_95_ppf, c="r", label="VaR à 95% de confiance")
plt.legend()
plt.title("Distribution normale — VaR 95%")
plt.tight_layout()
plt.show()




# 5. Conditional Value at Risk (CVaR) — Expected Shortfall

# Étapes pour dériver le CVaR :
#     1. Spécifier le niveau de confiance (ex : 0.95)
#     2. Créer ou utiliser un échantillon de la loss distribution
#     3. Calculer le VaR au niveau spécifié
#     4. Calculer le CVaR = perte espérée au-delà du VaR
#
# Méthode paramétrique (Normale) :
#     tail_loss = norm.expect(lambda x: x, loc, scale, lb=VaR)
#     CVaR = (1/(1-α)) × tail_loss
#
#     norm.expect(func, loc, scale, lb) calcule :
#         E[func(X)] = ∫ de lb à +∞ : func(x)·f(x; loc, scale)dx
#         lb = lower bound = VaR (borne inférieure de la queue)

# CVaR paramétrique sur N(0,1)
VaR_95_std = norm.ppf(0.95)
CVaR_95_std = (1 / (1 - 0.95)) * norm.expect(lambda x: x, lb=VaR_95_std)

print("\nCVaR PARAMÉTRIQUE — N(0,1)")
print(f"  VaR 95%  = {VaR_95_std:.4f}")
print(f"  CVaR 95% = {CVaR_95_std:.4f}")
print("  → CVaR > VaR : c'est la moyenne de la queue (worst 5%)")

# CVaR empirique sur les pertes Forex
CVaR_95_forex = loss[loss >= VaR_95].mean()

print("\nCVaR empirique (Forex)")
print(f"  VaR 95%  = {VaR_95:.4f} EUR")
print(f"  CVaR 95% = {CVaR_95_forex:.4f} EUR")




# 6. Exercice DataCamp — Comparing CVaR and VaR

# Objectif : calculer VaR et CVaR à 95% sur les portfolio_losses
# (données banques d'investissement 2005-2010).
#
# Étapes :
#     1. pm = portfolio_losses.mean()
#     2. ps = portfolio_losses.std()
#     3. VaR_95 = norm.ppf(0.95, loc=pm, scale=ps)
#     4. tail_loss = norm.expect(lambda x: x, loc=pm, scale=ps, lb=VaR_95)
#     5. CVaR_95 = (1/(1-0.95)) × tail_loss
#     6. Visualiser VaR (rouge) et CVaR (vert) sur l'histogramme

np.random.seed(42)
portfolio_losses = np.random.normal(loc=-0.01, scale=0.04, size=500)

pm = portfolio_losses.mean()
ps = portfolio_losses.std()

VaR_95_port = norm.ppf(0.95, loc=pm, scale=ps)
tail_loss = norm.expect(lambda x: x, loc=pm, scale=ps, lb=VaR_95_port)
CVaR_95_port = (1 / (1 - 0.95)) * tail_loss

print("\nEXERCICE — Comparing CVaR and VaR")
print(f"  Mean (pm)  = {pm:.6f}")
print(f"  Std  (ps)  = {ps:.6f}")
print(f"  VaR 95%    = {VaR_95_port:.6f}")
print(f"  CVaR 95%   = {CVaR_95_port:.6f}")

plt.figure(figsize=(10, 6))
plt.hist(norm.rvs(size=100000, loc=pm, scale=ps), bins=100,
         alpha=0.7, color="steelblue", edgecolor="black")
plt.axvline(x=VaR_95_port, c="r", label="VaR, niveau de confiance 95%")
plt.axvline(x=CVaR_95_port, c="g", label="CVaR, pires 5% des pertes")
plt.legend()
plt.title("Pertes du portefeuille — VaR vs CVaR")
plt.tight_layout()
plt.show()




# 7. Question conceptuelle — Which risk measure is "better"?

# Question :
#     "How does CVaR incorporate information from the tail
#      of the loss distribution?"
#
# Réponse :
#     "CVaR is an expected value over all of the losses EXCEEDING the VaR,
#      which are precisely the tail."
#
# Explication :
#     - Le CVaR est une ESPÉRANCE calculée sur la queue (pertes ≥ VaR)
#     - La forme et la masse de la queue contribuent au CVaR
#     - Le VaR sert de borne inférieure (lb) dans l'intégrale du CVaR
#     → Les deux mesures sont complémentaires




# 8. Visualisation CVaR — Queue de distribution

# Reproduction du slide du cours :
#     - Courbe de densité N(0,1)
#     - Ligne verticale au VaR 95%
#     - Zone colorée = queue des 5% pires cas (moyenne = CVaR)

x = np.linspace(-3, 3, 500)
y = norm.pdf(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, color="darkorange", linewidth=2,
         label="Distribution des pertes (densité)")

x_tail = x[x >= VaR_95_std]
y_tail = norm.pdf(x_tail)
plt.fill_between(x_tail, y_tail, alpha=0.5, color="steelblue",
                 label="Queue des 5% pires pertes")

plt.axvline(VaR_95_std, color="navy", linewidth=2,
            label=f"VaR 95% = {VaR_95_std:.2f}")

plt.xlabel("Perte")
plt.ylabel("Densité")
plt.title("VaR 95% et queue des 5% pires pertes")
plt.legend()
plt.tight_layout()
plt.show()
