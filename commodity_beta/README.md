# Bêta commodity d'actions énergie / matières premières

Régression multifactorielle (rendements log hebdomadaires, erreurs standard Newey-West) :

    r_action = α + β_marché · r_indice + Σ β_k · r_commodity_k + ε

Les β obtenus sont directement les ratios de couverture : pour 1 € d'action, vendre β_k € de
notionnel du future k (et β_marché € de future indice) pour neutraliser le risque prix et le
risque actions. Il ne reste alors que l'alpha et le risque spécifique.

```bash
pip install -r requirements.txt
python commodity_beta.py --plot                       # Yahoo Finance, depuis 2019
python commodity_beta.py --positions GTT.PA=150000,MOH.AT=80000
python commodity_beta.py --orthogonalize              # lecture "effet commodity pur"
python commodity_beta.py --demo                       # test sur données simulées
python commodity_beta.py --csv mes_prix.csv           # vos propres prix (Bloomberg, etc.)
```

L'univers (actions, indice, facteurs) se configure dans `UNIVERSE` en tête de script.
Un facteur `"HO=F-BZ=F"` est un spread de rendements (ici une marge de raffinage diesel).
