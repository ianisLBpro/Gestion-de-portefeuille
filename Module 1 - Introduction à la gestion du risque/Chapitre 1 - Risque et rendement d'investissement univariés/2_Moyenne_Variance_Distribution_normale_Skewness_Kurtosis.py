'''
2_Moyenne_Variance_Distribution_normale

Les 4 moments d'une distribution
Toute distribution de probabilité est caractérisée par 4 moments :

    1er moment - Moyenne (μ)        : tendance centrale, rendement moyen attendu
    2ème moment - Variance (σ²)     : dispersion autour de la moyenne = risque
    3ème moment - Skewness          : asymétrie de la distribution
    4ème moment - Kurtosis          : épaisseur des queues de distribution

Une distribution NORMALE générale est caractérisée par μ et σ².
La distribution NORMALE STANDARD est un cas particulier où μ = 0 et σ = 1.
Dans les deux cas : Skewness = 0 et Kurtosis = 3.

Les rendements financiers sont rarement normaux :
    - Skewness souvent négatif  => pertes extrêmes plus fréquentes que les gains extrêmes
    - Kurtosis souvent > 3      => queues plus épaisses que la normale (leptokurtose)

A savoir :
- Rendements discrets => analyse de risque et gestion de portefeuille
    - Calcul de statistiques descriptives (moyenne, volatilité, skewness, kurtosis)
    - Construction de portefeuille : R_portfolio = somme(w_i * R_i)
    - Présentation de résultats (rapports, clients)

- Log-rendements => modélisation mathématique
    - Black-Scholes, GBM, Monte Carlo
    - Agrégation dans le temps : r_annuel = somme des r_journaliers
    - Modèles statistiques : régression, GARCH

Règle : analyse de risque => discret | modélisation => log
'''

# Importation des librairies nécessaires et téléchargement des données

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm, skew, kurtosis, shapiro, jarque_bera


TICKER = "AAPL"
data = yf.download(TICKER, start="2015-01-01", end="2024-12-31",
                   auto_adjust=False, progress=False)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
data = data.sort_index()
data["Returns"] = data["Adj Close"].pct_change()
returns = data["Returns"].dropna()




# 1. Calcul de la moyenne journalière et annualisée 

'''
Moyenne (μ) - 1er moment :
Représente le rendement moyen attendu sur une période.
En Python : 
np.mean(returns)

Annualisation de la moyenne :
    On ne multiplie pas simplement par 252, on utilise la formule composée :

        R_annuel_moyen = (1 + R_journalier)^252 - 1

Pourquoi la formule composée et pas x252 ?
    => Un rendement de 0.1% par jour ne donne pas 25.2% sur un an 
        car chaque jour les intérêts se capitalisent sur la base précédente.
    => La multiplication simple est une approximation acceptable
        uniquement pour de très petits rendements (< 0.01%).

Convention : 252 jours de bourse par an (marchés fermés week-ends + jours fériés)
'''

mu_journalier = np.mean(returns)
mu_annualise = (1 + mu_journalier) ** 252 - 1

print(f"Rendement journalier moyen de {TICKER}  : {mu_journalier:.4%}")
print(f"Rendement annualisé moyen de {TICKER}   : {mu_annualise:.4%}")




# 2. Calcul de la variance et de la volatilité annualisée

'''
Variance (σ²) - 2ème moment :
    Mesure la dispersion des rendements autour de la moyenne.
    Exprimée dans l'unité au carré => peu intuitive à lire.

Écart-type (σ) = racine carrée de la variance :
    => Exprimé dans la même unité que les rendements => lisible
    => En finance, σ est appelé "volatilité"
    => Plus σ est élevé, plus l'actif est risqué (dispersion des rendements plus grande)

Propriété clé de la loi normale (règle des σ) :
    Dans une distribution normale, autour de la moyenne μ :
        ± 1σ contient environ 68.2% des observations  (34.1% de chaque côté)
        ± 2σ contient environ 95.4% des observations  (13.6% supplémentaires de chaque côté)
        ± 3σ contient environ 99.7% des observations  (2.1% supplémentaires de chaque côté)
=> Un rendement au-delà de 3σ est donc théoriquement très rare (0.3%)

En pratique sur les marchés, ces événements arrivent plus souvent
car les rendements ne sont pas parfaitement normaux (queues épaisses)

Annualisation de la volatilité (Square Root of Time Rule) :
Si les rendements sont indépendants d'un jour à l'autre,
la variance s'accumule linéairement dans le temps :

    Var(252 jours) = 252 * σ²_journalier

En prenant la racine carrée des deux côtés :

    σ_mensuel = σ_journalier * np.sqrt(21)
    σ_annuel  = σ_journalier * np.sqrt(252)

On multiplie par la racine du temps, PAS par le temps lui-même
'''

# ddof=1 : correction de Bessel, on divise par (n-1) au lieu de n
# Donne un estimateur non biaisé de la variance sur un échantillon
# np.std sans ddof=1 divise par n => légèrement sous-estime la vraie volatilité
sigma_journalier = np.std(returns, ddof=1)
variance_journaliere = sigma_journalier ** 2
sigma_mensuel = sigma_journalier * np.sqrt(21)
sigma_annualise = sigma_journalier * np.sqrt(252)


print(f"\nVariance journalière de {TICKER}        : {variance_journaliere:.6f}")
print(f"Volatilité journalière de {TICKER}      : {sigma_journalier:.4%}")
print(f"Volatilité mensuelle moyenne de {TICKER}: {sigma_mensuel:.4%}")
print(f"Volatilité annuelle moyenne de {TICKER} : {sigma_annualise:.4%}")



# 3. Comparaison de la distribution des rendements avec la distribution normale

'''
Objectif : vérifier visuellement si les rendements suivent une loi normale.

On superpose :
    - L'histogramme des rendements reels de l'actif selectionne (ce qu'on observe)
    - La courbe d'une loi normale théorique construite avec les mêmes μ et σ
Si les rendements étaient normaux, les deux se superposeraient parfaitement.

Ce qu'on observe en pratique :
    - Pic central plus élevé que la normale : les petits rendements sont très fréquents
    - Queues plus épaisses que la normale : les événements extrêmes arrivent plus souvent que ce que la normale prédit
    - Légère asymétrie : skewness non nul, dans la plus part des cas négatif

Pourquoi density=True dans hist() ?
    Sans density=True, l'axe Y affiche des fréquences brutes (nombre d'observations).
    norm.pdf() retourne des densités de probabilité (valeurs entre 0 et 1).
    => Les deux ne sont pas à la même échelle => la superposition serait incorrecte.
    Avec density=True, l'histogramme est normalisé en densité => même échelle que norm.pdf().

Pourquoi norm.pdf() et pas norm.cdf() ?
    pdf (Probability Density Function) => hauteur de la courbe à chaque point x
    cdf (Cumulative Distribution Function) => probabilité cumulée jusqu'à x
    => Pour tracer une courbe de distribution, on utilise toujours pdf.
'''

# Création de la figure pour comparer les distributions des rendements réels et de la normale théorique
fig, ax = plt.subplots(figsize=(10, 6))

# Histogramme des rendements réels
ax.hist(returns, bins=75, density=True,
        color="seagreen", alpha=0.75, label=f"{TICKER} rendements réels")

# Courbe normale théorique avec même μ : moyenne et σ : écart-type
x = np.linspace(returns.min(), returns.max(), 300)

# La fonction norm.pdf(x, loc=μ, scale=σ) calcule la densité de probabilité de la normale
ax.plot(x, norm.pdf(x, mu_journalier, sigma_journalier),
        color="crimson", linewidth=2,
        label=f"Normale théorique (μ={mu_journalier:.4f}, σ={sigma_journalier:.4f})")
ax.set_title(f"Distribution des rendements de {TICKER} vs Loi Normale")
ax.set_xlabel("Rendements")
ax.set_ylabel("Densité de probabilité")
ax.legend()
plt.tight_layout()
plt.show()




# 4. Skewness et Kurtosis

'''
Skewness (asymétrie) - 3ème moment :
Mesure dans quelle direction la distribution est asymétrique.

        = 0 : symétrique (cas de la normale)
        > 0 : queue droite plus longue => gains extrêmes plus fréquents
        < 0 : queue gauche plus longue => pertes extrêmes plus fréquentes

    En Python : scipy.stats.skew()

Kurtosis (aplatissement) - 4ème moment :
Mesure l'épaisseur des queues de distribution.

        = 3 : queues identiques à la normale (mésokurtique)
        > 3 : queues plus épaisses (leptokurtique) => événements extrêmes plus fréquents
        < 3 : queues plus fines (platykurtique)    => événements extrêmes plus rares

Excess Kurtosis = Kurtosis - 3 (référence = 0 au lieu de 3)
- scipy.stats.kurtosis() retourne l'excess kurtosis par défaut (fisher=True)
- Pour le kurtosis brut : kurtosis(data, fisher=False)
'''

# Calcul de la skewness et de la kurtosis des rendements
skewness = skew(returns)
kurt_excess = kurtosis(returns)
# Si fisher=True : excess kurtosis (kurtosis - 3), référence = 0
# Si fisher=False : kurtosis brut, référence = 3
kurt_brut = kurtosis(returns, fisher=False)
print(f"\nSkewness de {TICKER}         : {skewness:.4f} (normale = 0)")
print(f"Excess Kurtosis de {TICKER}  : {kurt_excess:.4f}  (normale = 0)")
print(f"Kurtosis brut de {TICKER}    : {kurt_brut:.4f}  (normale = 3)")




# 5. Test de normalité de Shapiro-Wilk

'''
Objectif : tester statistiquement si les rendements suivent une loi normale.
La visualisation seule ne suffit pas, on utilise un test statistique.

Hypothèses du test :
    H0 (hypothèse nulle)        : les données suivent une loi normale
    H1 (hypothèse alternative)  : les données ne suivent PAS une loi normale

Règle de décision avec le seuil alpha = 0.05 (5%) :

    La p-value représente la probabilité d'obtenir ces données SI H0 était vraie.

    p-value <= 0.05 :
        => Moins de 5% de chances que ces données soient normales
        => On rejette H0 => les rendements ne suivent PAS une loi normale

    p-value > 0.05 :
        => Plus de 5% de chances que ces données soient normales
        => On ne rejette pas H0 => on ne peut pas exclure la normalité

    Exemple concret :
        p-value = 0.001 => seulement 0.1% de chances que ce soit normal
                        => clairement non normal
        p-value = 0.80  => 80% de chances que ce soit normal
                        => probablement normal

                        
Limite du test de Shapiro-Wilk :
Conçu pour les petits échantillons (n < 5000).
Sur de grandes séries financières, le test rejette quasi-systématiquement H0
même pour des déviations mineures, on prend un sous-échantillon de 5000 obs maximum.

Alternative pour grands échantillons le test de Jarque-Bera :
Basé directement sur le skewness et le kurtosis :

        JB = (n/6) * (S² + (K-3)²/4)

Sous H0, JB suit un chi² à 2 degrés de liberté.
'''

# Shapiro-Wilk sur un sous-échantillon de 5000 obs maximum
echantillon = returns.sample(min(len(returns), 5000), random_state=42)
stat_sw, p_value_sw = shapiro(echantillon)

# Jarque-Bera sur toute la série
stat_jb, p_value_jb = jarque_bera(returns)

# Affichage des résultats des tests de normalité
print(f"\n--- Shapiro-Wilk ---")
print(f"Statistique : {stat_sw:.4f} | p-value : {p_value_sw:.10f}")
if p_value_sw <= 0.05:
    print(f"=> p-value de {p_value_sw:.10f} : seulement {p_value_sw*100:.8f}% de chances que {TICKER} soit normal")
    print(f"=> H0 rejetée : les rendements de {TICKER} ne suivent PAS une loi normale")
else:
    print(f"=> p-value de {p_value_sw:.10f} : {p_value_sw*100:.8f}% de chances que {TICKER} soit normal")
    print(f"=> H0 non rejetée : on ne peut pas exclure la normalité")

print(f"\n--- Jarque-Bera ---")
print(f"Statistique : {stat_jb:.4f} | p-value : {p_value_jb:.10f}")
if p_value_jb <= 0.05:
    print(f"=> p-value de {p_value_jb:.10f} : seulement {p_value_jb*100:.8f}% de chances que {TICKER} soit normal")
    print(f"=> H0 rejetée : les rendements de {TICKER} ne suivent PAS une loi normale")
else:
    print(f"=> p-value de {p_value_jb:.10f} : {p_value_jb*100:.8f}% de chances que {TICKER} soit normal")
    print(f"=> H0 non rejetée : on ne peut pas exclure la normalité")