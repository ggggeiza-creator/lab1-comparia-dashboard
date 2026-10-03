# Lab 1 compar:IA : taille du modèle vs difficulté

Tableau de bord Streamlit qui compare 6 modèles de langage (3 classes de taille) sur 8 prompts de 4 niveaux de difficulté : qualité (grille /10), énergie estimée par compar:IA et plus petite classe suffisante par niveau.

## Fichiers

| Fichier | Rôle |
|---|---|
| `lab1_log.csv` | Journal des 48 réponses, 19 colonnes de l'annexe A du CDC |
| `lab1_log_complet.csv` | Même journal + 10 colonnes compar:IA (tokens, coût, GPU, note plafonnée…), lu par le tableau de bord |
| `dashboard.py` | Tableau de bord (9 onglets) |
| `export_synthese.py` | Génère `lab1_synthese.csv` (niveau × classe, sans filtre, seuils 6 / 7 / 8) |
| `reco.py` | Génère `recommandations.csv` et `recommandations.md` (routage par type de tâche) |
| `test_dashboard.py` | Tests des règles de calcul (données de test intégrées) |

## Utilisation

```bash
pip install -r requirements-dev.txt   # ou requirements.txt sans les tests
streamlit run dashboard.py
python3 export_synthese.py
python3 reco.py
python3 -m pytest -q
```

## Règles de calcul

- Note /10 = exactitude (4) + complétude (2) + consignes (2) + clarté (2), plafonnée à 4 en cas d'erreur critique.
- Classe suffisante : note moyenne ≥ seuil (7 par défaut) **et** aucune erreur critique.
- Économie d'énergie = 1 − énergie moyenne de la classe / énergie moyenne XL, au même niveau.

## Écarts au cahier des charges

- Aucun modèle ≤ 4B disponible sur compar:IA : classes redéfinies S 15-60B, L 100-400B, XL > 400B (paramètres totaux).
- Mistral 3 Large remplacé par DeepSeek V4 Pro (substitution aléatoire de la plateforme).
- Règle stricte sur P4 : balises ```` ```json ```` = texte hors JSON, −1 point de consignes.
- Tableau de bord Streamlit à la place de `lab1_dashboard.html` (non fourni).

## Déploiement (Streamlit Community Cloud)

Fichier principal : `dashboard.py`. Choisir Python 3.12 dans *Advanced settings* (fichier `.python-version`).
