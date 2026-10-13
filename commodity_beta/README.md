# Bêta commodity d'actions énergie / matières premières

Régression multifactorielle (rendements log hebdomadaires, erreurs standard Newey-West) :

    r_action = α + β_marché · r_indice + Σ β_k · r_commodity_k + ε

Les β obtenus sont directement les ratios de couverture : pour 1 € d'action, vendre β_k € de
notionnel du future k (et β_marché € de future indice) pour neutraliser le risque prix et le
risque actions. Il ne reste alors que l'alpha et le risque spécifique.

## Données Refinitiv / LSEG (par défaut)

Workspace (ou Eikon) doit être ouvert sur la machine : la LSEG Data Library s'y connecte
en session desktop. Pour chaque RIC, le script prend `TRDPRC_1` (actions, indices),
sinon `SETTLE` (futures), sinon `MID_PRICE` / `BID` (change).

| Rôle | RIC |
|---|---|
| GTT | `GTT.PA` |
| Motor Oil Hellas | `MORr.AT` |
| STOXX Europe 600 | `.STOXX` |
| Brent ICE (1er contrat) | `LCOc1` |
| Gasoil ICE | `LGOc1` |
| Gaz TTF (ICE Endex) | `TFMBMc1` |
| Henry Hub | `NGc1` |
| EUR/USD | `EUR=` |

```bash
pip install -r requirements.txt
python commodity_beta.py --save-csv prix.csv --plot      # télécharge et met en cache
python commodity_beta.py --csv prix.csv                  # relance sans Workspace
python commodity_beta.py --positions GTT.PA=150000,MORr.AT=80000
python commodity_beta.py --orthogonalize                 # lecture "effet commodity pur"
python commodity_beta.py --demo                          # test sur données simulées
python commodity_beta.py --source yahoo                  # alternative Yahoo Finance
```

L'univers (actions, indice, facteurs) se configure dans `SOURCES` en tête de script.
Un facteur `"LGOc1 - LCOc1"` (espaces autour du tiret) est un spread de rendements :
ici la marge de raffinage gasoil. Les commodities en USD sont converties en EUR via `EUR=`.
