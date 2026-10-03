"""Recommandations de routage par type de tâche, à partir de lab1_log_complet.csv.

Règle : plus petite classe (S < L < XL) dont la note moyenne est >= 7 sans erreur critique.
Si aucune classe ne passe : classe S avec « contrôle automatique obligatoire ».

Lancement :
    python3 reco.py [journal.csv]
Génère recommandations.csv et recommandations.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import dashboard as db

THRESHOLD = 7.0
OUT_DIR = Path(__file__).parent

# Type de tâche -> prompts, condition d'usage
TASKS = [
    ("Question simple / conversion", ["P1", "P2"],
     "Contrôle de format (une phrase / une ligne) et des valeurs numériques"),
    ("Reformulation / extraction JSON", ["P3", "P4"],
     "Validation JSON (schéma) et contrôle des faits conservés (nombre de mots, aucun fait ajouté)"),
    ("Calcul multi-étapes", ["P5", "P6"],
     "Recalcul automatique du résultat final (outil de calcul ou script)"),
    ("Technique pointu", ["P7"],
     "Revue par un expert avant usage ; escalade vers XL si la vérification échoue"),
    ("Jugement éthique / santé", ["P8"],
     "Relecture humaine obligatoire avant toute diffusion"),
]


def _class_stats(sub: pd.DataFrame) -> pd.DataFrame:
    return sub.groupby("cls").agg(
        note=("score", "mean"), ec=("critical_error", "sum"), energie=("energy_wh", "mean")
    ).reindex([c for c in db.CLASSES if c in set(sub["cls"])])


def _pick_model(sub: pd.DataFrame, cls: str, threshold: float) -> pd.Series:
    """Dans la classe : modèle suffisant le moins énergivore, sinon le mieux noté."""
    m = sub[sub["cls"] == cls].groupby("model_name").agg(
        note=("score", "mean"), ec=("critical_error", "sum"), energie=("energy_wh", "mean"))
    m["ok"] = [db.is_sufficient(n, k, threshold) for n, k in zip(m["note"], m["ec"])]
    ok = m[m["ok"]]
    best = ok.sort_values("energie") if not ok.empty else m.sort_values(["note", "energie"], ascending=[False, True])
    row = best.iloc[0].copy()
    row["name"] = best.index[0]
    return row


def _fmt(x, nd=2):
    return db.fmt(x, nd)


def _pct(x):
    return f"{x:.0%}".replace("%", " %")


def build_recommendations(df: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    rows = []
    for task, prompts, condition in TASKS:
        sub = df[df["prompt_id"].isin(prompts)]
        if sub.empty:
            continue
        st = _class_stats(sub)
        st["ok"] = [db.is_sufficient(n, k, threshold) for n, k in zip(st["note"], st["ec"])]
        passing = [c for c in db.CLASSES if c in st.index and st.loc[c, "ok"]]
        xl_e = st.loc["XL", "energie"] if "XL" in st.index else np.nan
        detail = " ; ".join(f"{c} {_fmt(st.loc[c, 'note'])}/10, {int(st.loc[c, 'ec'])} EC"
                            for c in st.index)
        if passing:
            cls = passing[0]
            model = _pick_model(sub, cls, threshold)
            justification = (f"Plus petite classe ≥ {threshold:g}/10 sans erreur critique ({detail}). "
                             f"{model['name']} : {_fmt(model['note'])}/10, {int(model['ec'])} EC, "
                             f"modèle suffisant le moins énergivore de la classe.")
        else:
            cls = "S"
            model = _pick_model(sub, cls, threshold)
            ec_models = sorted(sub.loc[sub["critical_error"] == 1, "model_name"].unique())
            justification = (f"Aucune classe ne passe : chaque classe a au moins une erreur critique "
                             f"({detail} ; erreurs : {', '.join(ec_models)}). Classe S retenue par défaut, "
                             f"car monter en taille n'élimine pas le risque. {model['name']} : "
                             f"{_fmt(model['note'])}/10, {int(model['ec'])} EC.")
            condition = "Contrôle automatique obligatoire. " + condition
        rows.append({
            "type_tache": task,
            "prompts": "-".join(prompts),
            "classe_recommandee": cls,
            "modele_recommande": model["name"],
            "note_moyenne": round(float(st.loc[cls, "note"]), 2),
            "erreurs_critiques": int(st.loc[cls, "ec"]),
            "energie_moy_wh": round(float(st.loc[cls, "energie"]), 4),
            "economie_vs_xl": round(float(db.energy_saving(st.loc[cls, "energie"], xl_e)), 4),
            "classe_passe_le_seuil": bool(passing),
            "justification": justification,
            "condition": condition,
        })
    return pd.DataFrame(rows)


def to_markdown(reco: pd.DataFrame, df: pd.DataFrame, threshold: float = THRESHOLD) -> str:
    lines = [
        "# Recommandations de routage (Lab 1 compar:IA)",
        "",
        f"Source : `lab1_log_complet.csv` ({len(df)} réponses, {df['model_name'].nunique()} modèles). "
        f"Règle : plus petite classe avec une note moyenne ≥ {threshold:g}/10 sans erreur critique ; "
        "si aucune classe ne passe, classe S avec contrôle automatique obligatoire. "
        "Classes : S 15-60B, L 100-400B, XL > 400B (paramètres totaux).",
        "",
        "## Par type de tâche",
        "",
        "| Tâche | Prompts | Classe | Modèle | Note classe | EC | Énergie moy. (Wh) | Économie vs XL | Condition |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reco.itertuples():
        flag = "" if r.classe_passe_le_seuil else " ⚠️"
        lines.append(f"| {r.type_tache} | {r.prompts} | **{r.classe_recommandee}**{flag} | {r.modele_recommande} | "
                     f"{_fmt(r.note_moyenne)} | {r.erreurs_critiques} | {_fmt(r.energie_moy_wh, 3)} | "
                     f"{_pct(r.economie_vs_xl)} | {r.condition} |")
    lines += ["", "### Justifications", ""]
    lines += [f"- **{r.type_tache}** : {r.justification}" for r in reco.itertuples()]

    s = df[df["cls"] == "S"]["energy_wh"].mean()
    xl = df[df["cls"] == "XL"]["energy_wh"].mean()
    ratios = 1 / (1 - reco["economie_vs_xl"])
    no_xl_needed = (reco["classe_recommandee"] != "XL").all()
    duels = db.duel_outcomes(df)
    agree = int(duels["accord"].sum()) if not duels.empty else 0
    p7 = reco[reco["prompts"] == "P7"]
    p7_txt = ""
    if not p7.empty and p7.iloc[0]["classe_recommandee"] == "S":
        p7_txt = (f" Sur P7, la classe S passe pourtant le seuil ({_fmt(p7.iloc[0]['note_moyenne'])}/10) : "
                  "l'escalade est une mesure de prudence (un seul prompt, une seule exécution), "
                  "pas un résultat mesuré.")
    lines += [
        "",
        "## Recommandation globale",
        "",
        f"- **Cascade avec la classe S par défaut** : sur l'ensemble du journal, une réponse S consomme en moyenne "
        f"{_fmt(s, 3)} Wh contre {_fmt(xl, 3)} Wh pour XL (économie {_pct(1 - s / xl)}).",
        "- **Contrôle systématique en sortie** (validation JSON, recalcul, contrôle de format), "
        "et escalade vers XL quand le contrôle échoue.",
        "- **Escalade XL pour le technique difficile** (type P7) quand la réponse engage une décision "
        "d'infrastructure." + p7_txt,
        "- **Relecture humaine** pour tout contenu santé, juridique ou destiné à des publics vulnérables, "
        "quelle que soit la classe.",
        "",
        "## À éviter",
        "",
        "- Envoyer tout le trafic à XL : "
        + ("sur ce journal, aucun type de tâche n'en a besoin pour passer le seuil, " if no_xl_needed else "")
        + f"pour une énergie {ratios.min():.0f} à {ratios.max():.0f} fois plus élevée que la classe recommandée.",
        "- Croire que la taille élimine les erreurs critiques : sur reformulation / extraction, S, L et XL "
        "en ont chacune une.",
        "- Diffuser une sortie JSON sans validation de schéma, ou un calcul sans recalcul.",
        f"- Se fier au vote à l'aveugle : il coïncide avec la grille sur {agree} duels sur {len(duels)} "
        "(onglet 6 du tableau de bord).",
        "- Généraliser à des langues peu dotées (wolof, arabe) ou au domaine médical sans évaluation dédiée.",
        "",
        "## Limites",
        "",
        "- 2 prompts par niveau (1 seul pour « technique pointu » et « jugement éthique / santé »), "
        "une seule exécution par modèle : une seule erreur critique fait basculer une classe.",
        "- Énergie estimée par compar:IA (méthode EcoLogits), non mesurée ; nombre de GPU parfois inconnu.",
        "- Notation par le même binôme, sans double correction ; règle stricte appliquée sur P4.",
        "- Classes redéfinies (aucun modèle ≤ 4B disponible) : « S » désigne ici des modèles de 27 à 31B.",
        "- L'erreur critique de GLM 5.3 sur P4 vient peut-être d'un bug d'affichage (réponse tronquée).",
        "- Avant une décision de routage réelle : jeu de test plus large et représentatif du trafic, plusieurs "
        "exécutions, double notation, mesure de la latence et du coût réels.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    src = Path(argv[1]) if len(argv) > 1 else db.DEFAULT_LOG
    df = db.load_logs([src])
    reco = build_recommendations(df)
    reco.to_csv(OUT_DIR / "recommandations.csv", index=False, sep=";", decimal=",", encoding="utf-8-sig")
    (OUT_DIR / "recommandations.md").write_text(to_markdown(reco, df), encoding="utf-8")
    print(f"recommandations.csv et recommandations.md : {len(reco)} types de tâche (source {src.name})")


if __name__ == "__main__":
    main(sys.argv)
