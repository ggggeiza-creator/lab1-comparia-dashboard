# Recommandations de routage (Lab 1 compar:IA)

Source : `lab1_log_complet.csv` (48 réponses, 6 modèles). Règle : plus petite classe avec une note moyenne ≥ 7/10 sans erreur critique ; si aucune classe ne passe, classe S avec contrôle automatique obligatoire. Classes : S 15-60B, L 100-400B, XL > 400B (paramètres totaux).

## Par type de tâche

| Tâche | Prompts | Classe | Modèle | Note classe | EC | Énergie moy. (Wh) | Économie vs XL | Condition |
|---|---|---|---|---|---|---|---|---|
| Question simple / conversion | P1-P2 | **S** | Gemma 4 31B | 10,00 | 0 | 0,012 | 99 % | Contrôle de format (une phrase / une ligne) et des valeurs numériques |
| Reformulation / extraction JSON | P3-P4 | **S** ⚠️ | Qwen 3.8 27B | 7,75 | 1 | 0,099 | 99 % | Contrôle automatique obligatoire. Validation JSON (schéma) et contrôle des faits conservés (nombre de mots, aucun fait ajouté) |
| Calcul multi-étapes | P5-P6 | **S** | Gemma 4 31B | 9,50 | 0 | 0,116 | 97 % | Recalcul automatique du résultat final (outil de calcul ou script) |
| Technique pointu | P7 | **S** | Gemma 4 31B | 8,00 | 0 | 0,670 | 95 % | Revue par un expert avant usage ; escalade vers XL si la vérification échoue |
| Jugement éthique / santé | P8 | **S** | Gemma 4 31B | 9,50 | 0 | 0,224 | 97 % | Relecture humaine obligatoire avant toute diffusion |

### Justifications

- **Question simple / conversion** : Plus petite classe ≥ 7/10 sans erreur critique (S 10,00/10, 0 EC ; L 8,25/10, 0 EC ; XL 10,00/10, 0 EC). Gemma 4 31B : 10,00/10, 0 EC, modèle suffisant le moins énergivore de la classe.
- **Reformulation / extraction JSON** : Aucune classe ne passe : chaque classe a au moins une erreur critique (S 7,75/10, 1 EC ; L 7,50/10, 1 EC ; XL 7,25/10, 1 EC ; erreurs : GLM 5.3, Gemma 4 31B, Mistral Small 4). Classe S retenue par défaut, car monter en taille n'élimine pas le risque. Qwen 3.8 27B : 9,50/10, 0 EC.
- **Calcul multi-étapes** : Plus petite classe ≥ 7/10 sans erreur critique (S 9,50/10, 0 EC ; L 8,00/10, 1 EC ; XL 10,00/10, 0 EC). Gemma 4 31B : 9,50/10, 0 EC, modèle suffisant le moins énergivore de la classe.
- **Technique pointu** : Plus petite classe ≥ 7/10 sans erreur critique (S 8,00/10, 0 EC ; L 6,00/10, 1 EC ; XL 9,50/10, 0 EC). Gemma 4 31B : 7,00/10, 0 EC, modèle suffisant le moins énergivore de la classe.
- **Jugement éthique / santé** : Plus petite classe ≥ 7/10 sans erreur critique (S 9,50/10, 0 EC ; L 7,50/10, 0 EC ; XL 10,00/10, 0 EC). Gemma 4 31B : 9,00/10, 0 EC, modèle suffisant le moins énergivore de la classe.

## Recommandation globale

- **Cascade avec la classe S par défaut** : sur l'ensemble du journal, une réponse S consomme en moyenne 0,168 Wh contre 5,559 Wh pour XL (économie 97 %).
- **Contrôle systématique en sortie** (validation JSON, recalcul, contrôle de format), et escalade vers XL quand le contrôle échoue.
- **Escalade XL pour le technique difficile** (type P7) quand la réponse engage une décision d'infrastructure. Sur P7, la classe S passe pourtant le seuil (8,00/10) : l'escalade est une mesure de prudence (un seul prompt, une seule exécution), pas un résultat mesuré.
- **Relecture humaine** pour tout contenu santé, juridique ou destiné à des publics vulnérables, quelle que soit la classe.

## À éviter

- Envoyer tout le trafic à XL : sur ce journal, aucun type de tâche n'en a besoin pour passer le seuil, pour une énergie 18 à 80 fois plus élevée que la classe recommandée.
- Croire que la taille élimine les erreurs critiques : sur reformulation / extraction, S, L et XL en ont chacune une.
- Diffuser une sortie JSON sans validation de schéma, ou un calcul sans recalcul.
- Se fier au vote à l'aveugle : il coïncide avec la grille sur 21 duels sur 24 (onglet 6 du tableau de bord).
- Généraliser à des langues peu dotées (wolof, arabe) ou au domaine médical sans évaluation dédiée.

## Limites

- 2 prompts par niveau (1 seul pour « technique pointu » et « jugement éthique / santé »), une seule exécution par modèle : une seule erreur critique fait basculer une classe.
- Énergie estimée par compar:IA (méthode EcoLogits), non mesurée ; nombre de GPU parfois inconnu.
- Notation par le même binôme, sans double correction ; règle stricte appliquée sur P4.
- Classes redéfinies (aucun modèle ≤ 4B disponible) : « S » désigne ici des modèles de 27 à 31B.
- L'erreur critique de GLM 5.3 sur P4 vient peut-être d'un bug d'affichage (réponse tronquée).
- Avant une décision de routage réelle : jeu de test plus large et représentatif du trafic, plusieurs exécutions, double notation, mesure de la latence et du coût réels.
