"""Génère lab1_synthese.csv : synthèse niveau × classe, sans filtre, seuil 7 (+ seuils 6 et 8).

Lancement :
    python3 export_synthese.py [journal.csv] [sortie.csv]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import dashboard as db

THRESHOLD = 7.0
OTHER_THRESHOLDS = (6.0, 8.0)
DEFAULT_OUT = Path(__file__).with_name("lab1_synthese.csv")


def yes_no(v: bool) -> str:
    return "oui" if v else "non"


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    base = db.level_class_table(df, THRESHOLD)
    smallest = db.smallest_sufficient(base)
    out = pd.DataFrame({
        "niveau": base["niveau"],
        "classe": base["classe"],
        "n_reponses": base["n_reponses"],
        "note_moyenne": base["note_moyenne"].round(2),
        "erreurs_critiques": base["erreurs_critiques"],
        "energie_moy_wh": base["energie_moy_wh"].round(4),
        "suffisant_seuil_7": base["suffisante"].map(yes_no),
        "plus_petite_classe_suffisante_seuil_7": base["level"].map(lambda l: smallest.get(l) or "aucune"),
        "economie_vs_xl": base["economie_vs_xl"].round(4),
    })
    for th in OTHER_THRESHOLDS:
        t = db.level_class_table(df, th)
        s = db.smallest_sufficient(t)
        tag = int(th)
        out[f"suffisant_seuil_{tag}"] = t["suffisante"].map(yes_no).values
        out[f"plus_petite_classe_suffisante_seuil_{tag}"] = t["level"].map(lambda l: s.get(l) or "aucune").values
    return out


def main(argv: list[str]) -> None:
    src = Path(argv[1]) if len(argv) > 1 else db.DEFAULT_LOG
    dst = Path(argv[2]) if len(argv) > 2 else DEFAULT_OUT
    summary = build_summary(db.load_logs([src]))
    # Même format que le bouton du tableau de bord (Excel FR)
    summary.to_csv(dst, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"{dst.name} : {len(summary)} lignes (source {src.name}, seuil {THRESHOLD:g})")


if __name__ == "__main__":
    main(sys.argv)
