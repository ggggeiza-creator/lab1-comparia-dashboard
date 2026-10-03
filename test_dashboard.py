import numpy as np
import pandas as pd
import pytest

import dashboard as db
import export_synthese
import reco


# --- Journal de test intégré (48 lignes, plan de duels du CDC) ---------------

LOG_LEVELS = {"P1": "Easy", "P2": "Easy", "P3": "Medium", "P4": "Medium",
              "P5": "High", "P6": "High", "P7": "Hard", "P8": "Hard"}
LOG_MODELS = {  # nom -> (classe CDC, classe compar:IA, total, actifs, énergie Wh)
    "S-0": ("SLM", "S", 30, 30, 0.01), "S-1": ("SLM", "S", 27, 27, 0.02),
    "L-0": ("Small LLM", "L", 120, 6, 0.1), "L-1": ("Small LLM", "L", 117, 5, 0.2),
    "XL-0": ("Large LLM", "XL", 1600, 49, 1.0), "XL-1": ("Large LLM", "XL", 744, 40, 2.0),
}
LOG_DUELS = {"A": ("S-0", "XL-0"), "B": ("S-1", "L-0"), "C": ("L-1", "XL-1")}


def make_log(critical=(("P3", "S-0"),)):
    """48 réponses conformes ; les couples (prompt, modèle) de `critical` ont une erreur critique."""
    rows = []
    for pid, lvl in LOG_LEVELS.items():
        for duel, pair in LOG_DUELS.items():
            for pos, m in enumerate(pair):
                cdc, cls, tot, act, e = LOG_MODELS[m]
                crit = int((pid, m) in critical)
                rows.append({
                    "pair_id": "TEST", "prompt_id": pid, "level": lvl, "duel": duel, "model_name": m,
                    "size_class": cdc, "total_params_b": tot, "active_params_b": act, "energy_wh": e,
                    "co2e_g": None, "output_tokens": 100, "comparia_vote": "win" if pos else "loss",
                    "score_correctness": 4, "score_completeness": 2, "score_instruction": 1,
                    "score_clarity": 2, "critical_error": crit, "notes": "", "answer_text": "réponse",
                    "size_class_comparia": cls, "score_total_capped": 4 if crit else 9,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def log_file(tmp_path):
    path = tmp_path / "log.csv"
    make_log().to_csv(path, index=False)
    return path


# --- Plafonnement -----------------------------------------------------------

@pytest.mark.parametrize("total, crit, expected", [
    (10, 0, 10), (10, 1, 4), (7, 1, 4), (4, 1, 4), (3, 1, 3), (0, 1, 0), (6, 0, 6),
])
def test_capped_score(total, crit, expected):
    assert db.capped_score(total, crit) == expected


def test_capped_score_vectorised():
    out = db.capped_score([10, 9, 2], [1, 0, 1])
    assert list(out) == [4, 9, 2]


def test_log_score_total_capped_matches_rule(log_file):
    df = db.load_logs([log_file])
    assert len(df) == 48
    assert df.loc[df["critical_error"] == 1, "score"].tolist() == [4]
    assert (df["score"] == df["score_computed"]).all()
    assert (df.loc[df["critical_error"] == 1, "score"] <= 4).all()


# --- Classe suffisante ------------------------------------------------------

def test_is_sufficient_rule():
    assert db.is_sufficient(7.0, 0, 7.0)          # égalité au seuil
    assert db.is_sufficient(9.5, 0, 7.0)
    assert not db.is_sufficient(6.5, 0, 7.0)      # sous le seuil
    assert not db.is_sufficient(9.5, 1, 7.0)      # erreur critique
    assert not db.is_sufficient(np.nan, 0, 7.0)


def _toy():
    rows = []
    # niveau Easy : S suffisante ; niveau Hard : S a une erreur critique, L sous le seuil, XL ok
    spec = {
        ("Easy", "S"): [(8, 0), (9, 0)], ("Easy", "L"): [(9, 0), (9, 0)], ("Easy", "XL"): [(10, 0), (10, 0)],
        ("Hard", "S"): [(10, 0), (4, 1)], ("Hard", "L"): [(6, 0), (7, 0)], ("Hard", "XL"): [(8, 0), (9, 0)],
    }
    energy = {"S": 1.0, "L": 2.0, "XL": 10.0}
    for (lvl, c), items in spec.items():
        for i, (s, crit) in enumerate(items):
            rows.append({"level": lvl, "cls": c, "model_name": f"{c}-{i}", "score": s,
                         "critical_error": crit, "energy_wh": energy[c]})
    return pd.DataFrame(rows)


def test_smallest_sufficient_class():
    t = db.level_class_table(_toy(), 7.0)
    small = db.smallest_sufficient(t)
    assert small == {"Easy": "S", "Hard": "XL"}
    hard_s = t[(t.level == "Hard") & (t.classe == "S")].iloc[0]
    assert hard_s.note_moyenne == 7.0 and not hard_s.suffisante   # moyenne OK mais erreur critique


def test_no_class_passes():
    t = db.level_class_table(_toy(), 9.6)
    small = db.smallest_sufficient(t)
    assert small["Hard"] is None


def test_threshold_sensitivity():
    assert db.smallest_sufficient(db.level_class_table(_toy(), 6.0))["Hard"] == "L"
    assert db.smallest_sufficient(db.level_class_table(_toy(), 9.0))["Easy"] == "L"


def test_model_rule():
    m = db.level_model_table(_toy(), 7.0).set_index(["level", "model_name"])
    assert m.loc[("Hard", "S-0"), "suffisant"]
    assert not m.loc[("Hard", "S-1"), "suffisant"]


# --- Économie d'énergie ------------------------------------------------------

def test_energy_saving():
    assert db.energy_saving(1.0, 10.0) == pytest.approx(0.9)
    assert db.energy_saving(10.0, 10.0) == pytest.approx(0.0)
    assert db.energy_saving(20.0, 10.0) == pytest.approx(-1.0)
    assert np.isnan(db.energy_saving(1.0, 0.0))
    assert np.isnan(db.energy_saving(1.0, np.nan))


def test_energy_saving_in_table():
    t = db.level_class_table(_toy(), 7.0).set_index(["level", "classe"])
    assert t.loc[("Easy", "S"), "economie_vs_xl"] == pytest.approx(0.9)
    assert t.loc[("Easy", "L"), "economie_vs_xl"] == pytest.approx(0.8)
    assert t.loc[("Easy", "XL"), "economie_vs_xl"] == pytest.approx(0.0)


# --- Contrôles du CDC ----------------------------------------------------------

def test_log_checks_pass(log_file):
    checks = db.run_checks(db.load_logs([log_file]))
    assert checks["OK"].all(), checks[~checks["OK"]]


def test_log_checks_detect_problems(tmp_path):
    bad = make_log()
    bad.loc[0, "score_clarity"] = None                                  # réponse non notée
    bad.loc[bad["score_total_capped"] == 4, "score_total_capped"] = 9   # plafond oublié
    bad = bad.iloc[:-1]                                                 # 47 lignes
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    ko = db.run_checks(db.load_logs([path])).set_index("Contrôle")["OK"]
    assert not ko["48 réponses"]
    assert not ko["Toutes les réponses notées"]
    assert not ko["Note plafonnée cohérente (≤ 4 si erreur critique)"]


# --- Export de synthèse ---------------------------------------------------------

def test_export_summary(log_file):
    out = export_synthese.build_summary(db.load_logs([log_file])).set_index(["niveau", "classe"])
    assert len(out) == 12
    # Moyen : S a une erreur critique (P3) -> plus petite classe suffisante = L
    assert out.loc[("Moyen", "S"), "suffisant_seuil_7"] == "non"
    assert out.loc[("Moyen", "S"), "plus_petite_classe_suffisante_seuil_7"] == "L"
    assert out.loc[("Facile", "S"), "plus_petite_classe_suffisante_seuil_7"] == "S"
    # Toutes les notes valent 9 hors erreur critique : Facile passe aussi au seuil 6
    assert out.loc[("Facile", "S"), "suffisant_seuil_6"] == "oui"
    assert out.loc[("Facile", "S"), "economie_vs_xl"] == pytest.approx(1 - 0.015 / 1.5)
    for th in (6, 8):
        assert f"plus_petite_classe_suffisante_seuil_{th}" in out.columns


# --- Recommandations ---------------------------------------------------------

def test_recommendations(log_file):
    r = reco.build_recommendations(db.load_logs([log_file])).set_index("prompts")
    assert list(r.index) == ["P1-P2", "P3-P4", "P5-P6", "P7", "P8"]
    assert r.loc["P1-P2", "classe_recommandee"] == "S"
    assert r.loc["P1-P2", "modele_recommande"] == "S-0"          # suffisant le moins énergivore
    assert r.loc["P3-P4", "classe_recommandee"] == "L"           # S a une erreur critique
    assert r.loc["P3-P4", "classe_passe_le_seuil"]


def test_recommendations_fallback_when_no_class_passes(tmp_path):
    path = tmp_path / "log.csv"
    make_log(critical=(("P3", "S-0"), ("P3", "L-0"), ("P4", "XL-1"))).to_csv(path, index=False)
    df = db.load_logs([path])
    r = reco.build_recommendations(df).set_index("prompts")
    assert r.loc["P3-P4", "classe_recommandee"] == "S"
    assert not r.loc["P3-P4", "classe_passe_le_seuil"]
    assert r.loc["P3-P4", "condition"].startswith("Contrôle automatique obligatoire")
    assert r.loc["P3-P4", "modele_recommande"] == "S-1"          # le modèle S sans erreur
    md = reco.to_markdown(reco.build_recommendations(df), df)
    for section in ("## Recommandation globale", "## À éviter", "## Limites"):
        assert section in md


def test_pareto_front():
    df = pd.DataFrame({"energy_wh": [1, 2, 3, 0.5], "score": [8, 7, 10, 4]})
    pf = db.pareto_front(df)
    assert list(pf["energy_wh"]) == [0.5, 1, 3]
