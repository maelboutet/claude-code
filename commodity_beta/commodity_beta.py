"""
Mesure du bêta commodity d'actions de la chaîne de valeur énergie / matières premières.

Modèle (régression multifactorielle sur rendements log) :

    r_action = alpha + beta_marche * r_indice + sum_k beta_k * r_facteur_k + epsilon

- beta_marche : exposition au risque actions "générique" (couvrable par un short indice / future indice)
- beta_k      : exposition au facteur matière première k (couvrable par un short future commodity)

Les bêtas d'une régression multiple sont directement des ratios de couverture :
pour 1 EUR investi dans l'action, vendre beta_k EUR de notionnel du future k
et beta_marche EUR de notionnel du future indice.

Un facteur peut être un "spread" de rendements "A-B" (ex. "HO=F-BZ=F" ≈ marge de raffinage :
rendement du diesel moins rendement du Brent).

Usage :
    python commodity_beta.py                         # télécharge via yfinance
    python commodity_beta.py --csv prix.csv          # prix déjà en local (colonnes = tickers)
    python commodity_beta.py --demo                  # données synthétiques (vérifie le moteur)
    python commodity_beta.py --freq D --window 104 --plot
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

# --------------------------------------------------------------------------------------
# Configuration : action -> indice de marché + facteurs matières premières
# Tickers Yahoo Finance. Adaptez librement.
# --------------------------------------------------------------------------------------
UNIVERSE: dict[str, dict] = {
    # GTT : membranes de cuves GNL -> sensible au gaz européen (TTF), US (Henry Hub)
    # et à l'arbitrage GNL (spread TTF - HH)
    "GTT.PA": {
        "name": "GTT",
        "market": "^STOXX",
        "factors": ["TTF=F", "NG=F", "BZ=F"],
    },
    # Motor Oil Hellas : raffineur -> prix du brut + marge de raffinage (diesel - Brent)
    "MOH.AT": {
        "name": "Motor Oil Hellas",
        "market": "^STOXX",
        "factors": ["BZ=F", "HO=F-BZ=F"],
    },
}

# Facteurs cotés en USD : convertis en EUR si --eur (cohérent avec un portefeuille en EUR)
USD_TICKERS = {"BZ=F", "CL=F", "NG=F", "HO=F", "RB=F", "HG=F", "GC=F", "ALI=F"}
EURUSD = "EURUSD=X"

FACTOR_LABELS = {
    "BZ=F": "Brent",
    "CL=F": "WTI",
    "TTF=F": "Gaz TTF",
    "NG=F": "Gaz Henry Hub",
    "HO=F": "Diesel NY (ULSD)",
    "RB=F": "Essence RBOB",
    "HG=F": "Cuivre",
    "HO=F-BZ=F": "Crack diesel (ULSD - Brent)",
    "^STOXX": "STOXX Europe 600",
}


# --------------------------------------------------------------------------------------
# Données
# --------------------------------------------------------------------------------------
def needed_tickers(universe: dict, eur: bool) -> list[str]:
    tickers = set()
    for stock, cfg in universe.items():
        tickers.add(stock)
        tickers.add(cfg["market"])
        for f in cfg["factors"]:
            tickers.update(f.split("-") if _is_spread(f) else [f])
    if eur:
        tickers.add(EURUSD)
    return sorted(tickers)


def _is_spread(factor: str) -> bool:
    # "^STOXX" ne contient pas de "-", mais un ticker comme "BRK-B" en contiendrait un :
    # on ne considère comme spread que les écritures "X=F-Y=F" / "A-B" où les deux côtés
    # sont des tickers connus de l'utilisateur. Ici : présence de "=F-".
    return "=F-" in factor


def download_prices(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    missing = [t for t in tickers if t not in px.columns or px[t].dropna().empty]
    if missing:
        raise RuntimeError(f"Pas de données pour : {missing}")
    return px


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def synthetic_prices(start: str = "2018-01-01", n_weeks: int = 400, seed: int = 0) -> pd.DataFrame:
    """Prix simulés avec des bêtas connus, pour valider la mécanique d'estimation."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_weeks * 5, freq="B")
    n = len(idx)
    mkt = rng.normal(0, 0.010, n)
    brent = 0.3 * mkt + rng.normal(0, 0.020, n)
    ttf = 0.1 * mkt + 0.3 * brent + rng.normal(0, 0.040, n)
    hh = rng.normal(0, 0.035, n)
    ho = 0.9 * brent + rng.normal(0, 0.012, n)
    fx = rng.normal(0, 0.004, n)
    # Vrais bêtas : GTT  mkt 0.8, TTF 0.15, HH 0.05, Brent 0.10
    #               MOH  mkt 0.9, Brent 0.40, crack 0.60
    gtt = 0.8 * mkt + 0.15 * ttf + 0.05 * hh + 0.10 * brent + rng.normal(0, 0.015, n)
    moh = 0.9 * mkt + 0.40 * brent + 0.60 * (ho - brent) + rng.normal(0, 0.014, n)
    rets = pd.DataFrame(
        {"^STOXX": mkt, "BZ=F": brent, "TTF=F": ttf, "NG=F": hh, "HO=F": ho,
         "GTT.PA": gtt, "MOH.AT": moh, EURUSD: fx},
        index=idx,
    )
    return 100 * np.exp(rets.cumsum())


# --------------------------------------------------------------------------------------
# Rendements & facteurs
# --------------------------------------------------------------------------------------
def to_returns(prices: pd.DataFrame, freq: str, eur: bool) -> pd.DataFrame:
    """Rendements log à la fréquence demandée.

    L'hebdomadaire (W-FRI) est recommandé : les clôtures de Paris/Athènes et les
    settlements ICE/NYMEX ne sont pas synchrones, ce qui biaise les bêtas journaliers
    vers zéro (effet Epps).
    """
    px = prices.ffill()
    if eur and EURUSD in px.columns:
        # prix USD -> EUR : P_eur = P_usd / EURUSD
        for t in px.columns:
            if t in USD_TICKERS:
                px[t] = px[t] / px[EURUSD]
    if freq.upper() != "D":
        px = px.resample(freq).last()
    # Les futures continus et le gaz peuvent passer par des prix <= 0 : on les écarte
    px = px.where(px > 0)
    return np.log(px).diff().iloc[1:]


def build_factor(rets: pd.DataFrame, factor: str) -> pd.Series:
    if _is_spread(factor):
        a, b = factor.split("-", 1)
        return (rets[a] - rets[b]).rename(factor)
    return rets[factor].rename(factor)


# --------------------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------------------
@dataclass
class BetaResult:
    stock: str
    betas: pd.Series
    tstats: pd.Series
    r2: float
    r2_market_only: float
    nobs: int
    resid_vol_ann: float

    @property
    def commodity_r2_increment(self) -> float:
        """Part de variance expliquée en plus par les commodities, au-delà du marché."""
        return self.r2 - self.r2_market_only


def estimate(rets: pd.DataFrame, stock: str, cfg: dict, hac_lags: int, periods_per_year: int,
             orthogonalize: bool = False) -> BetaResult:
    y = rets[stock]
    X = pd.concat([rets[cfg["market"]].rename(cfg["market"])]
                  + [build_factor(rets, f) for f in cfg["factors"]], axis=1)
    data = pd.concat([y, X], axis=1).dropna()
    y, X = data[stock], data.drop(columns=stock)

    if orthogonalize:
        # Retire de chaque facteur commodity sa composante "marché" :
        # le bêta marché capte alors l'effet total du marché, et les bêtas commodity
        # l'effet *propre* aux matières premières. (Pour la couverture, garder False.)
        m = X[cfg["market"]]
        for f in cfg["factors"]:
            X[f] = sm.OLS(X[f], sm.add_constant(m)).fit().resid

    model = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    mkt_only = sm.OLS(y, sm.add_constant(X[[cfg["market"]]])).fit()

    return BetaResult(
        stock=stock,
        betas=model.params.drop("const"),
        tstats=model.tvalues.drop("const"),
        r2=model.rsquared,
        r2_market_only=mkt_only.rsquared,
        nobs=int(model.nobs),
        resid_vol_ann=float(model.resid.std() * np.sqrt(periods_per_year)),
    )


def rolling_betas(rets: pd.DataFrame, stock: str, cfg: dict, window: int) -> pd.DataFrame:
    """Bêtas glissants (OLS) pour juger de la stabilité de l'exposition."""
    from statsmodels.regression.rolling import RollingOLS

    X = pd.concat([rets[cfg["market"]].rename(cfg["market"])]
                  + [build_factor(rets, f) for f in cfg["factors"]], axis=1)
    data = pd.concat([rets[stock], X], axis=1).dropna()
    res = RollingOLS(data[stock], sm.add_constant(data.drop(columns=stock)), window=window).fit()
    return res.params.drop(columns="const")


# --------------------------------------------------------------------------------------
# Portefeuille & couverture
# --------------------------------------------------------------------------------------
def portfolio_exposures(results: dict[str, BetaResult], positions_eur: dict[str, float]) -> pd.Series:
    """Exposition en EUR de notionnel à chaque facteur = somme(position_i * beta_i,k)."""
    expo: dict[str, float] = {}
    for stock, notional in positions_eur.items():
        for f, b in results[stock].betas.items():
            expo[f] = expo.get(f, 0.0) + notional * b
    return pd.Series(expo).sort_index()


def hedge_table(exposures: pd.Series) -> pd.DataFrame:
    """Notionnel à vendre (négatif) pour neutraliser chaque facteur.

    Un facteur spread "A-B" (ex. crack) se couvre en vendant A et en achetant B
    pour le même notionnel ; on le décompose donc sur ses deux jambes.
    """
    legs: dict[str, float] = {}
    for f, e in exposures.items():
        if _is_spread(f):
            a, b = f.split("-", 1)
            legs[a] = legs.get(a, 0.0) - e
            legs[b] = legs.get(b, 0.0) + e
        else:
            legs[f] = legs.get(f, 0.0) - e
    out = pd.DataFrame({"notionnel_couverture_eur": pd.Series(legs)})
    out["instrument"] = [FACTOR_LABELS.get(t, t) for t in out.index]
    return out.sort_index()


# --------------------------------------------------------------------------------------
# Affichage
# --------------------------------------------------------------------------------------
def print_result(res: BetaResult, cfg: dict) -> None:
    print(f"\n=== {cfg['name']} ({res.stock}) — {res.nobs} obs ===")
    tab = pd.DataFrame({"beta": res.betas, "t-stat (HAC)": res.tstats})
    tab.index = [FACTOR_LABELS.get(i, i) for i in tab.index]
    print(tab.round(3).to_string())
    print(f"R² total : {res.r2:.1%} | R² marché seul : {res.r2_market_only:.1%} "
          f"| apport commodities : {res.commodity_r2_increment:+.1%}")
    print(f"Vol idiosyncratique annualisée : {res.resid_vol_ann:.1%}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--csv", help="CSV de prix (index = dates, colonnes = tickers)")
    src.add_argument("--demo", action="store_true", help="données synthétiques")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--freq", default="W-FRI", help="D, W-FRI, ME… (défaut : hebdo)")
    p.add_argument("--window", type=int, default=52, help="fenêtre des bêtas glissants (périodes)")
    p.add_argument("--no-eur", action="store_true", help="laisser les commodities en USD")
    p.add_argument("--orthogonalize", action="store_true",
                   help="facteurs commodity orthogonalisés au marché (lecture, pas couverture)")
    p.add_argument("--positions", default="GTT.PA=100000,MOH.AT=100000",
                   help="positions en EUR, ex. GTT.PA=100000,MOH.AT=50000")
    p.add_argument("--plot", action="store_true", help="graphiques des bêtas glissants")
    args = p.parse_args()

    eur = not args.no_eur
    if args.demo:
        prices = synthetic_prices()
    elif args.csv:
        prices = load_csv(args.csv)
    else:
        prices = download_prices(needed_tickers(UNIVERSE, eur), args.start, args.end)

    rets = to_returns(prices, args.freq, eur)
    ppy = {"D": 252, "W": 52, "M": 12}.get(args.freq.upper()[0], 52)
    hac_lags = {"D": 5, "W": 4, "M": 2}.get(args.freq.upper()[0], 4)

    results: dict[str, BetaResult] = {}
    for stock, cfg in UNIVERSE.items():
        results[stock] = estimate(rets, stock, cfg, hac_lags, ppy, args.orthogonalize)
        print_result(results[stock], cfg)

    positions = {k: float(v) for k, v in (kv.split("=") for kv in args.positions.split(","))}
    expo = portfolio_exposures(results, positions)
    print("\n=== Exposition du portefeuille (EUR de notionnel équivalent) ===")
    print(expo.rename(index=lambda i: FACTOR_LABELS.get(i, i)).round(0).to_string())
    if args.orthogonalize:
        print("\n(Facteurs orthogonalisés : relancer sans --orthogonalize pour les ratios de couverture.)")
    else:
        print("\n=== Couverture (négatif = vendre des futures / short indice) ===")
        print(hedge_table(expo).round(0).to_string())

    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(UNIVERSE), 1, figsize=(11, 4 * len(UNIVERSE)), sharex=True)
        for ax, (stock, cfg) in zip(np.atleast_1d(axes), UNIVERSE.items()):
            rb = rolling_betas(rets, stock, cfg, args.window)
            rb.rename(columns=lambda c: FACTOR_LABELS.get(c, c)).plot(ax=ax, lw=1.3)
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_title(f"{cfg['name']} — bêtas glissants {args.window} périodes")
            ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig("rolling_betas.png", dpi=130)
        print("\nGraphique enregistré : rolling_betas.png")


if __name__ == "__main__":
    main()
