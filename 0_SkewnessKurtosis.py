from __future__ import annotations

"""
Skewness Kurtosis analiza.
"""

import os

SEED = 39
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd

import time
from datetime import datetime, timedelta
import pytz

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import label_ranking_average_precision_score, roc_auc_score

try:
    from qiskit_machine_learning.utils import algorithm_globals
    algorithm_globals.random_seed = SEED
except Exception:
    pass

np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# Konfiguracija
# ============================================================
CSV_PATH    = "/loto7_4622_k42.csv"
OUT_TXT     = "/0_SkewnessKurtosis.txt"
N_MIN, N_MAX = 1, 39
K           = 7
LAG         = 5
WINDOWS     = (20, 50, 100)
BACKTEST_N  = 100


def stamp() -> str:
    return datetime.now(pytz.timezone("Europe/Belgrade")).strftime("%d.%m.%Y_%H.%M.%S")


T0 = time.time()
print()
print("🔁 0_SkewnessKurtosis — start ", stamp())
print()


# ============================================================
# 1) Učitavanje CSV-a (bez headera, 7 kolona)
# ============================================================
df = pd.read_csv(CSV_PATH, header=None)
df = df.iloc[:, :K].astype(int)
draws = df.values
N = draws.shape[0]
print(f"✅ CSV učitan: {CSV_PATH}")
print(f"   broj izvlačenja: {N}, brojeva po kolu: {K}")
print()



import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# 2) Skewness i kurtosis po 7 kolona izvlačenja
# ============================================================
print("Skewness po 7 kolona (pozicije u izvlačenju):")
print(df.skew())
print()
print("Excess Kurtosis po 7 kolona:")
print(df.kurtosis())
print()


# ============================================================
# 3) Izvedene statistike kola (suma, neparnih, niskih, raspon)
# ============================================================
stats_df = pd.DataFrame({
    "suma":     draws.sum(axis=1),
    "neparnih": (draws % 2 == 1).sum(axis=1),
    "niskih":   (draws <= 19).sum(axis=1),
    "raspon":   draws.max(axis=1) - draws.min(axis=1),
}).astype(float)

print("Skewness izvedenih statistika kola:")
print(stats_df.skew())
print()
print("Excess Kurtosis izvedenih statistika kola:")
print(stats_df.kurtosis())
print()

# Log-transformacija sume kola za poređenje simetrije
suma_log = np.log1p(stats_df["suma"])
print(f"Suma kola: skew={stats_df['suma'].skew():.4f}, kurt={stats_df['suma'].kurtosis():.4f}")
print(f"log1p(suma): skew={suma_log.skew():.4f}, kurt={suma_log.kurtosis():.4f}")
print()


# ============================================================
# 4) Vizuelizacija: hist + boxplot + QQ za 4 izvedene statistike
# ============================================================
fig, axes = plt.subplots(len(stats_df.columns), 3, figsize=(15, 4 * len(stats_df.columns)))
for i, col in enumerate(stats_df.columns):
    series = stats_df[col].dropna()
    series.hist(ax=axes[i, 0], bins=30, color="steelblue", edgecolor="white")
    axes[i, 0].set_title(f"{col} — Histogram")

    axes[i, 1].boxplot(series.values, vert=True)
    axes[i, 1].set_title(f"{col} — Box Plot")
    axes[i, 1].set_xticks([1], [col])

    stats.probplot(series.values, dist="norm", plot=axes[i, 2])
    axes[i, 2].set_title(f"{col} — QQ Plot")

plt.tight_layout()
PNG_PATH = "/0_SkewnessKurtosis.png"
plt.savefig(PNG_PATH)
plt.show()
print(f"🖼️  Plot snimljen u: {PNG_PATH}")
print()







elapsed = time.time() - T0
with open(OUT_TXT, "a", encoding="utf-8") as f:
    f.write(f"\n--- {stamp()} (seed={SEED}, N={N}) ---\n")
    f.write("skewness_po_kolonama: " + str(df.skew().round(4).to_dict()) + "\n")
    f.write("kurtosis_po_kolonama: " + str(df.kurtosis().round(4).to_dict()) + "\n")
    f.write("skewness_izvedenih:   " + str(stats_df.skew().round(4).to_dict()) + "\n")
    f.write("kurtosis_izvedenih:   " + str(stats_df.kurtosis().round(4).to_dict()) + "\n")
    f.write(f"suma_log1p: skew={suma_log.skew():.4f}, kurt={suma_log.kurtosis():.4f}\n")
    f.write(f"plot={PNG_PATH}\n")
    f.write(f"ukupno_vreme={str(timedelta(seconds=int(elapsed)))}  ({elapsed:.1f} s)\n")
print(f"📝 Snimljeno u: {OUT_TXT}")

print()
print("🔁 0_SkewnessKurtosis — stop ", stamp())
print(f"⏱️  Ukupno vreme: {str(timedelta(seconds=int(elapsed)))}  ({elapsed:.1f} s)")
print()



"""

🔁 0_SkewnessKurtosis — start  28.05.2026_11.46.25

✅ CSV učitan: /data/loto7_4622_k42.csv
   broj izvlačenja: 4622, brojeva po kolu: 7

Skewness po 7 kolona (pozicije u izvlačenju):
0    1.339360
1    0.732800
2    0.303771
3   -0.054454
4   -0.344746
5   -0.758994
6   -1.403680
dtype: float64

Excess Kurtosis po 7 kolona:
0    1.872824
1    0.214157
2   -0.473269
3   -0.560941
4   -0.367676
5    0.269614
6    2.229009
dtype: float64

Skewness izvedenih statistika kola:
suma       -0.021804
neparnih   -0.036596
niskih      0.032400
raspon     -0.700106
dtype: float64

Excess Kurtosis izvedenih statistika kola:
suma       -0.176951
neparnih   -0.257632
niskih     -0.211563
raspon      0.158393
dtype: float64

Suma kola: skew=-0.0218, kurt=-0.1770
log1p(suma): skew=-0.6444, kurt=0.7326

🖼️  Plot snimljen u: /0_SkewnessKurtosis.png

📝 Snimljeno u: /0_SkewnessKurtosis.txt

🔁 0_SkewnessKurtosis — stop  28.05.2026_11.46.30
⏱️  Ukupno vreme: 0:00:04  (4.9 s)

"""




"""
Tumačenje rezultata (loto7_4622_k42.csv, N=4622):

Skewness (asimetrija):
  > 0  rep ide udesno (gornji deo dalje od proseka)
  < 0  rep ide ulevo
  |skew| < 0.5  praktično simetrično
  0.5-1         umereno asimetrično
  > 1           jako asimetrično

Excess kurtosis (pandas vraća excess, ne sirovu):
  = 0  kao normalna raspodela
  > 0  oštriji vrh, deblji repovi (leptokurtic)
  < 0  pljosnatiji vrh, kraći repovi (platikurtic)


Skewness po 7 kolona (sortirano izvlačenje, pozicije 0..6):
  kol. 0 (najmanji broj): +1.34  jako desno asimetrično (najmanji obično mali, retko izleti gore)
  kol. 1:                 +0.73  umereno desno
  kol. 2:                 +0.30  blago desno
  kol. 3:                 -0.05  simetrično (medijan pozicija)
  kol. 4:                 -0.34  blago levo
  kol. 5:                 -0.76  umereno levo
  kol. 6 (najveći broj):  -1.40  jako levo asimetrično

To je očekivan obrazac: krajnje pozicije sortiranog kola su pritisnute
na zid opsega [1..39], srednja je slobodna i Gaussovska.
Kurtoza prati istu krivu (1.87 i 2.23 na krajevima — oštri vrhovi,
deblji repovi; ~0 u sredini).


Izvedene statistike kola:
  suma:     skew=-0.02, kurt=-0.18   praktično normalna (centralna granična teorema)
  neparnih: skew=-0.04, kurt=-0.26   simetrično oko 3.5/7
  niskih:   skew=+0.03, kurt=-0.21   simetrično oko 3.5/7
  raspon:   skew=-0.70, kurt=+0.16   umereno levo asimetrično (raspon blizu max 38)


log1p(suma): skew=-0.64, kurt=+0.73
  Log transformacija ovde POGORŠAVA simetriju, jer je originalna suma već skoro normalna.
  Log korigovati treba samo desno asimetrične podatke (skew > +1), a ovde to nije slučaj.


Šta to znači praktično:
  1. Loto generator je statistički "fer" — izvedene statistike
     (suma, neparnih, niskih) su normalne, što potvrđuje slučajnost.
  2. Predikcija "broj po poziciji" je problematična jer kol. 0 i kol. 6
     imaju krute granice — zato je urađen multi-label pristup po brojevima 1..39.
  3. Filteri po sumi i parnosti su pouzdani (Gauss → odbacuju ekstremne kombinacije).
  4. Raspon je manje koristan kao filter jer ima blagu asimetriju i nije Gauss.
"""








"""
Kako da izračunata statistika postane konkretan filter/rerank za NEXT predikciju:

1. Šta se dobija iz CSV-a (jednom, kao prior)

μ, σ za 4 statistike: suma, neparnih, niskih, raspon — to su Gaussovi (skew ≈ 0) → svaka kombinacija dobija Gauss log-PDF.
Po-pozicija histogram (kol. 0..6) — kol. 0 favorizuje brojeve 1–10, kol. 6 brojeve 30–39 (zbog skew ±1.4).


2. Kako ulazi u predikciju

Postoje 4 nivoa, od najlakše do najboljeg:

A) Tvrdi filter (najprostije)

Regresor da top-1 kombinaciju.
Ako joj je suma van μ ± 2σ ili neparnih van [2..5] → odbaciti, uzeti sledeću kombinaciju.


B) Soft rerank (preporučujem)

Iz regresora uzme top-15 brojeva po skoru (oko 5000 mogućih C(15,7) kombinacija).
Za svaku kombinaciju izračunati rerank score:
score = Σ regresor_skor(broj)
      + λ · ( log N(suma | μ_s, σ_s)
            + log N(neparnih | μ_o, σ_o)
            + log N(niskih | μ_l, σ_l)
            + log N(raspon | μ_r, σ_r) )
λ kontroliše koliko statistika "kažnjava" loše kombinacije. λ ≈ 1-3.
Top-1 po score = NEXT.


C) Per-pozicija prior

Sortiraš kombinaciju i za svaku poziciju (0..6) dodaje log P(broj | pozicija) iz istorijskog histograma.
Hvata onu blagu krivu skewness-a po kolonama (kol. 0 nije baš 39, kol. 6 nije baš 1).


D) Bayes ansambl (kombinacija B+C)

score = model_likelihood + λ₁ · gauss_stats + λ₂ · per_pozicija
Najmoćnije. 

Filtrira kombinacije koje statistički ne liče na pravo loto kolo (npr. svih 7 brojeva ispod 19 — niskih=7/7, daleko od μ ≈ 3.5; ili suma 60, daleko od μ ≈ 140).
Ne menja "koji brojevi" — menja koja kombinaciju od top-N.




4_StatsRerank.py 

Učita CSV, izračuna μ,σ za 4 statistike + per-pozicija histogram.
Pozove svoje regresore (3) unutar fajla.
Generiše top-15 brojeva, evaluira sve C(15,7), rerank po formuli iz B (ili D).
Snimi NEXT pick + TXT + plot raspodele kandidata.

"""


