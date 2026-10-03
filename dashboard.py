"""Lab 1 compar:IA : taille du modèle vs difficulté.

Tableau de bord Streamlit + Plotly. Lancement :
    streamlit run dashboard.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Référentiels
# ---------------------------------------------------------------------------

DEFAULT_LOG = Path(__file__).with_name("lab1_log_complet.csv")

CLASSES = ["S", "L", "XL"]
CLASS_LABELS = {"S": "S (15-60B)", "L": "L (100-400B)", "XL": "XL (>400B)"}
# Correspondance avec les libellés de l'annexe A du CDC
CDC_CLASS_TO_CODE = {"SLM": "S", "Small LLM": "L", "Large LLM": "XL"}

LEVELS = ["Easy", "Medium", "High", "Hard"]
LEVEL_FR = {"Easy": "Facile", "Medium": "Moyen", "High": "Élevé", "Hard": "Difficile"}
PROMPTS = [f"P{i}" for i in range(1, 9)]

# Casting final (ordre d'affichage)
ROSTER = {
    "S": ["Gemma 4 31B", "Qwen 3.8 27B"],
    "L": ["GPT OSS-120B", "Mistral Small 4"],
    "XL": ["DeepSeek V4 Pro", "GLM 5.3"],
}
ROSTER_DETAIL = {
    "Gemma 4 31B": "31B dense",
    "Qwen 3.8 27B": "27B dense",
    "GPT OSS-120B": "117B, 5,1B actifs",
    "Mistral Small 4": "119B, 6B actifs",
    "DeepSeek V4 Pro": "1600B, 49B actifs",
    "GLM 5.3": "744B, 40B actifs",
}

# Plan de duels du CDC (section 4) : duel -> paire de classes attendue
DUEL_PLAN = {"A": {"S", "XL"}, "B": {"S", "L"}, "C": {"L", "XL"}}

SCORE_COLS = ["score_correctness", "score_completeness", "score_instruction", "score_clarity"]
CRITICAL_CAP = 4.0

# Palette catégorielle validée (3 premiers créneaux : sûrs en « toutes paires »)
CLASS_COLORS = {"S": "#2a78d6", "L": "#eb6834", "XL": "#1baf7a"}
# Niveaux ordonnés : rampe séquentielle mono-teinte (clair -> foncé)
LEVEL_COLORS = {"Easy": "#8fbcf0", "Medium": "#4f8fdf", "High": "#2a64b8", "Hard": "#0f3a78"}
LEVEL_SYMBOLS = {"Easy": "circle", "Medium": "square", "High": "diamond", "Hard": "triangle-up"}

X_AXES = {
    "class": "Classe CDC (S < L < XL)",
    "total": "Paramètres totaux (Md, échelle log)",
    "active": "Paramètres actifs (Md, échelle log)",
}


# ---------------------------------------------------------------------------
# Règles de calcul (testées dans test_dashboard.py)
# ---------------------------------------------------------------------------

def capped_score(scores_sum, critical_error):
    """Note /10 = somme des 4 critères, plafonnée à 4 si erreur critique."""
    scores_sum = np.asarray(scores_sum, dtype=float)
    critical = np.asarray(critical_error, dtype=float) == 1
    out = np.where(critical, np.minimum(scores_sum, CRITICAL_CAP), scores_sum)
    return out if out.ndim else float(out)


def is_sufficient(mean_score, n_critical, threshold):
    """Suffisant si note moyenne >= seuil ET aucune erreur critique."""
    if mean_score is None or pd.isna(mean_score):
        return False
    return bool(mean_score >= threshold - 1e-9 and n_critical == 0)


def energy_saving(class_energy, xl_energy):
    """Économie d'énergie = 1 - énergie moyenne de la classe / énergie moyenne XL."""
    if xl_energy is None or pd.isna(xl_energy) or xl_energy <= 0 or pd.isna(class_energy):
        return np.nan
    return 1.0 - class_energy / xl_energy


def pareto_front(df, x="energy_wh", y="score"):
    """Points non dominés : énergie minimale, note maximale."""
    pts = df[[x, y]].dropna().sort_values([x, y], ascending=[True, False])
    keep, best = [], -np.inf
    for idx, row in pts.iterrows():
        if row[y] > best:
            keep.append(idx)
            best = row[y]
    return df.loc[keep].sort_values(x)


# ---------------------------------------------------------------------------
# Préparation des données
# ---------------------------------------------------------------------------

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "size_class_comparia" in df.columns:
        df["cls"] = df["size_class_comparia"].astype(str).str.strip()
    else:
        df["cls"] = df["size_class"].map(CDC_CLASS_TO_CODE)
    for c in SCORE_COLS + ["critical_error", "total_params_b", "active_params_b", "energy_wh",
                           "output_tokens", "tokens_total", "cost_eur", "gpus"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["tokens_total", "cost_eur", "gpus"]:
        if c not in df.columns:
            df[c] = np.nan
    if "energy_wh" not in df.columns and "energy_mwh" in df.columns:
        df["energy_wh"] = pd.to_numeric(df["energy_mwh"], errors="coerce") / 1000
    df["critical_error"] = df["critical_error"].fillna(0).astype(int)
    df["score_raw"] = df[SCORE_COLS].sum(axis=1, min_count=4)
    computed = capped_score(df["score_raw"], df["critical_error"])
    if "score_total_capped" in df.columns:
        df["score"] = pd.to_numeric(df["score_total_capped"], errors="coerce").fillna(
            pd.Series(computed, index=df.index))
    else:
        df["score"] = computed
    df["score_computed"] = computed
    df["tokens_mean_basis"] = df["tokens_total"].fillna(df["output_tokens"])
    df["energy_per_token_mwh"] = df["energy_wh"] * 1000 / df["output_tokens"]
    df["level_fr"] = df["level"].map(LEVEL_FR)
    df["cls_order"] = df["cls"].map({c: i for i, c in enumerate(CLASSES)})
    return df


def load_logs(files) -> pd.DataFrame:
    frames = []
    for f in files:
        d = pd.read_csv(f, dtype={"pair_id": str})
        d["source_file"] = getattr(f, "name", str(f))
        frames.append(d)
    return prepare(pd.concat(frames, ignore_index=True))


def level_class_table(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Une ligne par niveau × classe : note, erreurs critiques, énergie, suffisance, économie."""
    rows = []
    for lvl in LEVELS:
        sub = df[df["level"] == lvl]
        xl_e = sub.loc[sub["cls"] == "XL", "energy_wh"].mean()
        for c in CLASSES:
            s = sub[sub["cls"] == c]
            if s.empty:
                continue
            mean = s["score"].mean()
            ncrit = int(s["critical_error"].sum())
            rows.append({
                "level": lvl, "niveau": LEVEL_FR[lvl], "classe": c,
                "n_reponses": len(s), "note_moyenne": mean, "erreurs_critiques": ncrit,
                "energie_moy_wh": s["energy_wh"].mean(), "energie_tot_wh": s["energy_wh"].sum(),
                "suffisante": is_sufficient(mean, ncrit, threshold),
                "economie_vs_xl": energy_saving(s["energy_wh"].mean(), xl_e),
            })
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    smallest = smallest_sufficient(t)
    t["plus_petite_suffisante"] = [
        smallest.get(r.level) == r.classe for r in t.itertuples()]
    return t


def smallest_sufficient(table: pd.DataFrame) -> dict:
    """Niveau -> plus petite classe suffisante (ou None)."""
    out = {}
    for lvl in LEVELS:
        ok = table[(table["level"] == lvl) & table["suffisante"]]
        if (table["level"] == lvl).any():
            out[lvl] = min(ok["classe"], key=CLASSES.index) if not ok.empty else None
    return out


def level_model_table(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    g = df.groupby(["level", "cls", "model_name"]).agg(
        note_moyenne=("score", "mean"), erreurs_critiques=("critical_error", "sum"),
        energie_moy_wh=("energy_wh", "mean")).reset_index()
    g["suffisant"] = [is_sufficient(m, n, threshold)
                      for m, n in zip(g["note_moyenne"], g["erreurs_critiques"])]
    return g


def duel_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par duel : gagnant au vote vs gagnant à la grille."""
    rows = []
    keys = [k for k in ["source_file", "pair_id", "prompt_id", "duel"] if k in df.columns]
    for key, d in df.groupby(keys):
        if len(d) != 2:
            continue
        a, b = d.iloc[0], d.iloc[1]
        votes = set(d["comparia_vote"].astype(str).str.lower())
        if votes == {"tie"}:
            vote_w = "Égalité"
        elif "win" in votes:
            vote_w = d.loc[d["comparia_vote"].str.lower() == "win", "model_name"].iloc[0]
        else:
            vote_w = "?"
        if a["score"] == b["score"]:
            grid_w = "Égalité"
        else:
            grid_w = a["model_name"] if a["score"] > b["score"] else b["model_name"]
        rows.append({
            **dict(zip(keys, key if isinstance(key, tuple) else (key,))),
            "niveau": LEVEL_FR.get(a["level"], a["level"]),
            "modele_1": a["model_name"], "note_1": a["score"],
            "modele_2": b["model_name"], "note_2": b["score"],
            "gagnant_vote": vote_w, "gagnant_grille": grid_w,
            "accord": vote_w == grid_w,
            "vote_pour_le_plus_grand": (
                vote_w != "Égalité" and vote_w != "?" and
                d.loc[d["model_name"] == vote_w, "cls_order"].iloc[0] == d["cls_order"].max()),
        })
    return pd.DataFrame(rows)


def run_checks(df: pd.DataFrame) -> pd.DataFrame:
    checks = []
    group_key = "source_file" if "source_file" in df.columns else None
    groups = df.groupby(group_key) if group_key else [("log", df)]
    for name, d in groups:
        pfx = f"[{name}] " if df.get("source_file", pd.Series()).nunique() > 1 else ""
        checks.append((pfx + "48 réponses", len(d) == 48, f"{len(d)} lignes"))
        per_class = d.groupby("cls")["model_name"].nunique().reindex(CLASSES).fillna(0).astype(int)
        checks.append((pfx + "2 modèles par classe", bool((per_class == 2).all()),
                       ", ".join(f"{c} : {n}" for c, n in per_class.items())))
        unscored = int(d[SCORE_COLS].isna().any(axis=1).sum())
        checks.append((pfx + "Toutes les réponses notées", unscored == 0,
                       f"{unscored} réponse(s) sans note complète"))
        no_energy = int(d["energy_wh"].isna().sum())
        checks.append((pfx + "Énergie renseignée", no_energy == 0, f"{no_energy} valeur(s) manquante(s)"))
        mismatch = int((d["score"] - d["score_computed"]).abs().gt(1e-9).sum())
        checks.append((pfx + "Note plafonnée cohérente (≤ 4 si erreur critique)", mismatch == 0,
                       f"{mismatch} écart(s) entre score_total_capped et le recalcul"))
        bad = []
        for (pid, duel), g in d.groupby(["prompt_id", "duel"]):
            if set(g["cls"]) != DUEL_PLAN.get(duel, set()) or len(g) != 2:
                bad.append(f"{pid}/{duel}")
        n_duels = d.groupby(["prompt_id", "duel"]).ngroups
        checks.append((pfx + "Plan de duels respecté (A = S/XL, B = S/L, C = L/XL)",
                       not bad and n_duels == 24,
                       f"{n_duels} duels" + (f" ; non conformes : {', '.join(bad)}" if bad else "")))
        per_model = d.groupby("model_name")["prompt_id"].nunique()
        checks.append((pfx + "Chaque modèle répond aux 8 prompts", bool((per_model == 8).all()),
                       ", ".join(f"{m} : {n}" for m, n in per_model.items())))
    return pd.DataFrame(checks, columns=["Contrôle", "OK", "Détail"])


def ratio_table(df: pd.DataFrame) -> pd.DataFrame:
    by = df.groupby("cls").agg(e=("energy_wh", "mean"), tot=("total_params_b", "mean"),
                               act=("active_params_b", "mean"))
    rows = []
    for num, den in [("XL", "S"), ("XL", "L"), ("L", "S")]:
        if num in by.index and den in by.index:
            rows.append({
                "Rapport": f"{num} / {den}",
                "Énergie moyenne par réponse": by.loc[num, "e"] / by.loc[den, "e"],
                "Paramètres totaux": by.loc[num, "tot"] / by.loc[den, "tot"],
                "Paramètres actifs": by.loc[num, "act"] / by.loc[den, "act"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

def fmt(x, nd=2):
    if x is None or pd.isna(x):
        return "–"
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def main():
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="Lab 1 compar:IA", layout="wide")
    st.title("Lab 1 compar:IA : taille du modèle vs difficulté")

    # ---------------- Barre latérale ----------------
    sb = st.sidebar
    sb.header("Données")
    uploads = sb.file_uploader("Journaux CSV (vue classe : plusieurs fichiers)", type="csv",
                               accept_multiple_files=True)
    include_default = sb.checkbox("Inclure lab1_log_complet.csv", value=not uploads)
    files = list(uploads or [])
    if include_default and DEFAULT_LOG.exists():
        files.insert(0, DEFAULT_LOG)
    if not files:
        st.warning("Aucun fichier chargé.")
        st.stop()
    raw = load_logs(files)

    sb.header("Modèles")
    models_all = set(raw["model_name"].unique())
    selected = []
    for c in CLASSES:
        sb.markdown(f"**Classe {CLASS_LABELS[c]}**")
        names = [m for m in ROSTER[c] if m in models_all] + sorted(
            set(raw.loc[raw["cls"] == c, "model_name"]) - set(ROSTER[c]))
        for m in names:
            if sb.checkbox(f"{m} ({ROSTER_DETAIL.get(m, '')})", value=True, key=f"m_{m}"):
                selected.append(m)

    sb.header("Filtres")
    levels = sb.multiselect("Niveaux", LEVELS, default=LEVELS, format_func=LEVEL_FR.get)
    prompts = sb.multiselect("Prompts", PROMPTS, default=PROMPTS)
    threshold = sb.slider("Seuil de qualité (/10)", 0.0, 10.0, 7.0, 0.5)
    x_mode = sb.radio("Axe X", list(X_AXES), format_func=X_AXES.get)

    df = raw[raw["model_name"].isin(selected) & raw["level"].isin(levels)
             & raw["prompt_id"].isin(prompts)]
    if df.empty:
        st.warning("Aucune ligne ne correspond aux filtres.")
        st.stop()

    with st.container(border=True):
        st.markdown(
            "**Écarts au cahier des charges**\n"
            "- Aucun modèle ≤ 4B disponible sur compar:IA : classes redéfinies "
            "**S 15-60B**, **L 100-400B**, **XL > 400B** (paramètres totaux).\n"
            "- **Mistral 3 Large** remplacé par **DeepSeek V4 Pro** (substitué aléatoirement par la plateforme).\n"
            "- Règle stricte appliquée sur **P4** : balises `` ```json `` = texte hors JSON, −1 point de consignes.\n"
            "- Tableau de bord Streamlit à la place de `lab1_dashboard.html` (non fourni).\n"
            "- `lab1_log.csv` conforme à l'annexe A ; colonnes supplémentaires dans `lab1_log_complet.csv`.")

    n_src = raw["source_file"].nunique()
    st.caption(f"{len(df)} réponses affichées sur {len(raw)} chargées · {n_src} fichier(s) · "
               f"seuil {fmt(threshold, 1)}/10 · note = score_total_capped (plafonnée à 4 si erreur critique)")

    lc = level_class_table(df, threshold)
    lm = level_model_table(df, threshold)
    present_levels = [l for l in LEVELS if l in set(df["level"])]

    def x_of(frame):
        return {"class": "cls_order", "total": "total_params_b", "active": "active_params_b"}[x_mode]

    def style_x(fig):
        if x_mode == "class":
            fig.update_xaxes(tickvals=[0, 1, 2], ticktext=[CLASS_LABELS[c] for c in CLASSES],
                             range=[-0.4, 2.4], title=X_AXES["class"])
        else:
            fig.update_xaxes(type="log", title=X_AXES[x_mode])
        return fig

    def layout(fig, h=480):
        fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(orientation="h", y=-0.2), hovermode="closest")
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
        return fig

    tabs = st.tabs(["1. Synthèse", "2. Taille vs qualité", "3. Taille vs énergie",
                    "4. Quel modèle pour quel prompt", "5. Qualité vs énergie",
                    "6. Votes vs grille", "7. Détail par prompt", "8. Contrôles du CDC",
                    "9. Recommandations"])

    # ---------------- 1. Synthèse ----------------
    def indicators(frame, key):
        g = frame.groupby(key).agg(
            reponses=("score", "size"), note_moyenne=("score", "mean"),
            erreurs_critiques=("critical_error", "sum"), energie_totale_wh=("energy_wh", "sum"),
            cout_total_eur=("cost_eur", "sum"), tokens_moyens=("tokens_mean_basis", "mean"))
        return g

    with tabs[0]:
        by_cls = indicators(df, "cls").reindex([c for c in CLASSES if c in set(df["cls"])])
        cols = st.columns(len(by_cls))
        for col, (c, r) in zip(cols, by_cls.iterrows()):
            with col.container(border=True):
                st.markdown(f"**Classe {CLASS_LABELS[c]}**")
                st.metric("Note moyenne", f"{fmt(r.note_moyenne)}/10")
                st.metric("Erreurs critiques", int(r.erreurs_critiques))
                st.metric("Énergie totale", f"{fmt(r.energie_totale_wh, 2)} Wh")
                st.metric("Coût total", f"{fmt(r.cout_total_eur, 4)} €")
                st.metric("Tokens moyens", fmt(r.tokens_moyens, 0))
        st.subheader("Par modèle")
        by_m = indicators(df, ["cls", "model_name"]).reset_index()
        by_m["cls_order"] = by_m["cls"].map(CLASSES.index)
        by_m = by_m.sort_values(["cls_order", "model_name"]).drop(columns="cls_order")
        st.dataframe(by_m.rename(columns={
            "cls": "Classe", "model_name": "Modèle", "reponses": "Réponses",
            "note_moyenne": "Note moyenne", "erreurs_critiques": "Erreurs critiques",
            "energie_totale_wh": "Énergie totale (Wh)", "cout_total_eur": "Coût total (€)",
            "tokens_moyens": "Tokens moyens"}),
            hide_index=True, width="stretch",
            column_config={"Note moyenne": st.column_config.NumberColumn(format="%.2f"),
                           "Énergie totale (Wh)": st.column_config.NumberColumn(format="%.3f"),
                           "Coût total (€)": st.column_config.NumberColumn(format="%.5f"),
                           "Tokens moyens": st.column_config.NumberColumn(format="%.0f")})
        st.caption("Tokens moyens : tokens_total (entrée + sortie) affichés par compar:IA.")

    # ---------------- 2. Taille vs qualité ----------------
    with tabs[1]:
        if x_mode == "class":
            pts = df.groupby(["level", "cls"]).agg(
                score=("score", "mean"), x=("cls_order", "first"),
                label=("model_name", lambda s: " / ".join(sorted(s.unique())))).reset_index()
        else:
            pts = df.groupby(["level", "model_name", "cls"]).agg(
                score=("score", "mean"), x=(x_of(df), "first")).reset_index()
            pts["label"] = pts["model_name"]
        fig = go.Figure()
        for lvl in present_levels:
            p = pts[pts["level"] == lvl].sort_values("x")
            fig.add_trace(go.Scatter(
                x=p["x"], y=p["score"], mode="lines+markers+text", name=LEVEL_FR[lvl],
                text=p["label"], textposition="top center", textfont=dict(size=10),
                line=dict(color=LEVEL_COLORS[lvl], width=2),
                marker=dict(size=10, symbol=LEVEL_SYMBOLS[lvl], color=LEVEL_COLORS[lvl],
                            line=dict(width=2, color="white")),
                hovertemplate="%{text}<br>Note moyenne : %{y:.2f}<extra>" + LEVEL_FR[lvl] + "</extra>"))
        fig.add_hline(y=threshold, line_dash="dash", line_color="gray",
                      annotation_text=f"Seuil {fmt(threshold, 1)}", annotation_position="bottom right")
        fig.update_yaxes(title="Note moyenne /10", range=[0, 10.8])
        st.plotly_chart(layout(style_x(fig), 540), width="stretch")
        st.caption("Une courbe par niveau. Vue classe : moyenne des 2 modèles de la classe ; "
                   "vues paramètres : un point par modèle.")

    # ---------------- 3. Taille vs énergie ----------------
    with tabs[2]:
        e = df.groupby(["model_name", "cls"]).agg(
            energie=("energy_wh", "mean"), total=("total_params_b", "first"),
            active=("active_params_b", "first"), cls_order=("cls_order", "first"),
            gpus=("gpus", "first")).reset_index()
        if x_mode == "class":
            ec = df.groupby("cls").agg(energie=("energy_wh", "mean"), x=("cls_order", "first")).reset_index()
        xcol = {"class": "cls_order", "total": "total", "active": "active"}[x_mode]
        fig = go.Figure()
        for c in CLASSES:
            p = e[e["cls"] == c]
            if p.empty:
                continue
            fig.add_trace(go.Scatter(
                x=p[xcol], y=p["energie"], mode="markers+text", name=f"Classe {c}",
                text=p["model_name"], textposition="middle right", textfont=dict(size=10),
                marker=dict(size=12, color=CLASS_COLORS[c], line=dict(width=2, color="white")),
                hovertemplate="%{text}<br>Énergie moyenne : %{y:.3f} Wh<extra></extra>"))
        if x_mode == "class":
            fig.add_trace(go.Scatter(x=ec["x"], y=ec["energie"], mode="lines+markers",
                                     name="Moyenne de classe", line=dict(color="gray", width=2, dash="dot"),
                                     marker=dict(size=8, color="gray"),
                                     hovertemplate="Moyenne : %{y:.3f} Wh<extra></extra>"))
        fig.update_yaxes(type="log", title="Énergie moyenne par réponse (Wh, log)")
        st.plotly_chart(layout(style_x(fig)), width="stretch")

        st.subheader("Énergie par token de sortie vs paramètres actifs")
        t = df.dropna(subset=["energy_per_token_mwh"]).groupby(["model_name", "cls"]).agg(
            ept=("energy_per_token_mwh", "mean"), active=("active_params_b", "first"),
            gpus=("gpus", "first")).reset_index()
        t["gpus_txt"] = t["gpus"].map(lambda g: "inconnu" if pd.isna(g) else f"{int(g)}")
        t["size"] = t["gpus"].map(lambda g: 14 if pd.isna(g) else 10 + 6 * np.sqrt(g))
        fig = go.Figure()
        for c in CLASSES:
            p = t[t["cls"] == c]
            if p.empty:
                continue
            fig.add_trace(go.Scatter(
                x=p["active"], y=p["ept"], mode="markers+text", name=f"Classe {c}",
                text=p["model_name"], textposition="top center", textfont=dict(size=10),
                customdata=p[["gpus_txt"]],
                marker=dict(size=p["size"], color=CLASS_COLORS[c],
                            symbol=["circle-open" if g == "inconnu" else "circle" for g in p["gpus_txt"]],
                            line=dict(width=2, color=CLASS_COLORS[c])),
                hovertemplate="%{text}<br>%{y:.3f} mWh/token<br>Actifs : %{x} Md"
                              "<br>GPU : %{customdata[0]}<extra></extra>"))
        fig.update_xaxes(type="log", title="Paramètres actifs (Md, log)")
        fig.update_yaxes(type="log", title="Énergie par token de sortie (mWh, log)")
        st.plotly_chart(layout(fig), width="stretch")
        st.caption("Taille des points = nombre de GPU affiché par compar:IA ; cercle vide = GPU inconnu. "
                   f"{int(df['energy_per_token_mwh'].isna().sum())} réponse(s) sans tokens de sortie exclue(s).")

        st.subheader("Rapports d'énergie vs rapports de paramètres")
        rt = ratio_table(df)
        st.dataframe(rt, hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="× %.1f") for c in rt.columns if c != "Rapport"})
        st.caption("Moyennes par classe. Selon EcoLogits, l'énergie dépend des paramètres actifs "
                   "(et du nombre de GPU requis par les paramètres totaux) multipliés par les tokens générés.")

    # ---------------- 4. Quel modèle pour quel prompt ----------------
    with tabs[3]:
        if lc.empty:
            st.info("Pas de données.")
        else:
            cls_present = [c for c in CLASSES if c in set(lc["classe"])]
            z = lc.pivot(index="level", columns="classe", values="note_moyenne").reindex(
                index=present_levels, columns=cls_present)
            nc = lc.pivot(index="level", columns="classe", values="erreurs_critiques").reindex(
                index=present_levels, columns=cls_present)
            txt = [[f"{fmt(z.iloc[i, j])}<br>{int(nc.iloc[i, j])} EC" if not pd.isna(z.iloc[i, j]) else ""
                    for j in range(len(cls_present))] for i in range(len(present_levels))]
            fig = go.Figure(go.Heatmap(
                z=z.values, x=[CLASS_LABELS[c] for c in cls_present],
                y=[LEVEL_FR[l] for l in present_levels], text=txt, texttemplate="%{text}",
                colorscale=[[0, "#f1f6fd"], [1, "#0f3a78"]], zmin=0, zmax=10,
                colorbar=dict(title="Note"),
                hovertemplate="%{y} · %{x}<br>Note moyenne : %{z:.2f}<extra></extra>"))
            smallest = smallest_sufficient(lc)
            for i, lvl in enumerate(present_levels):
                c = smallest.get(lvl)
                if c in cls_present:
                    j = cls_present.index(c)
                    fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                                  line=dict(color="#eb6834", width=4))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(layout(fig, 420), width="stretch")
            st.caption("Annotation : note moyenne et nombre d'erreurs critiques (EC). "
                       "Cadre orange = plus petite classe suffisante du niveau.")

            for lvl in present_levels:
                if smallest.get(lvl) is None:
                    st.error(f"Niveau {LEVEL_FR[lvl]} : aucune classe ne passe le seuil de "
                             f"{fmt(threshold, 1)}/10 sans erreur critique.")

            st.subheader("Plus petite classe suffisante et économie d'énergie vs XL")
            rows = []
            for lvl in present_levels:
                c = smallest.get(lvl)
                r = lc[(lc["level"] == lvl) & (lc["classe"] == c)] if c else pd.DataFrame()
                rows.append({
                    "Niveau": LEVEL_FR[lvl], "Plus petite classe suffisante": c or "aucune",
                    "Note moyenne": r["note_moyenne"].iloc[0] if c else np.nan,
                    "Énergie moy. (Wh)": r["energie_moy_wh"].iloc[0] if c else np.nan,
                    "Économie vs XL": r["economie_vs_xl"].iloc[0] if c else np.nan})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                "Note moyenne": st.column_config.NumberColumn(format="%.2f"),
                "Énergie moy. (Wh)": st.column_config.NumberColumn(format="%.3f"),
                "Économie vs XL": st.column_config.NumberColumn(format="percent")})

            st.subheader("Sensibilité au seuil")
            sens = {}
            for th in (6.0, 7.0, 8.0):
                s = smallest_sufficient(level_class_table(df, th))
                sens[f"Seuil {int(th)}/10"] = [s.get(l) or "aucune" for l in present_levels]
            st.dataframe(pd.DataFrame(sens, index=[LEVEL_FR[l] for l in present_levels]),
                         width="stretch")

            st.subheader("Même règle par modèle")
            pm = lm.copy()
            pm["Niveau"] = pm["level"].map(LEVEL_FR)
            piv = pm.pivot(index="model_name", columns="Niveau", values="suffisant").reindex(
                columns=[LEVEL_FR[l] for l in present_levels])
            order = [m for c in CLASSES for m in ROSTER[c] if m in piv.index] + \
                    [m for m in piv.index if m not in sum(ROSTER.values(), [])]
            piv = piv.reindex(order).map(lambda v: "✅ suffisant" if v else "❌ insuffisant")
            notes = pm.pivot(index="model_name", columns="Niveau", values="note_moyenne").reindex(
                index=order, columns=[LEVEL_FR[l] for l in present_levels])
            ec = pm.pivot(index="model_name", columns="Niveau", values="erreurs_critiques").reindex(
                index=order, columns=[LEVEL_FR[l] for l in present_levels])
            disp = piv.copy()
            for col in disp.columns:
                disp[col] = [f"{v} ({fmt(n)} ; {int(k)} EC)" for v, n, k in zip(piv[col], notes[col], ec[col])]
            disp.index.name = "Modèle"
            st.dataframe(disp, width="stretch")

    # ---------------- 5. Qualité vs énergie ----------------
    with tabs[4]:
        fig = go.Figure()
        for c in CLASSES:
            for lvl in present_levels:
                p = df[(df["cls"] == c) & (df["level"] == lvl)]
                if p.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=p["energy_wh"], y=p["score"], mode="markers",
                    name=f"{c} · {LEVEL_FR[lvl]}", legendgroup=c,
                    customdata=p[["model_name", "prompt_id"]],
                    marker=dict(size=11, color=CLASS_COLORS[c], symbol=LEVEL_SYMBOLS[lvl],
                                line=dict(width=1.5, color="white")),
                    hovertemplate="%{customdata[0]} · %{customdata[1]}<br>Énergie : %{x:.3f} Wh"
                                  "<br>Note : %{y}<extra></extra>"))
        pf = pareto_front(df)
        fig.add_trace(go.Scatter(
            x=pf["energy_wh"], y=pf["score"], mode="lines", name="Frontière de Pareto",
            line=dict(color="gray", width=2, dash="dash", shape="hv"), hoverinfo="skip"))
        fig.update_xaxes(type="log", title="Énergie par réponse (Wh, log)")
        fig.update_yaxes(title="Note /10", range=[-0.3, 10.5])
        st.plotly_chart(layout(fig, 560), width="stretch")
        st.caption("Couleur = classe, symbole = niveau (● Facile, ■ Moyen, ◆ Élevé, ▲ Difficile). "
                   "Frontière : réponses non dominées (aucune autre n'a moins d'énergie et une note au moins égale).")
        st.dataframe(pf[["model_name", "cls", "prompt_id", "level_fr", "energy_wh", "score"]].rename(columns={
            "model_name": "Modèle", "cls": "Classe", "prompt_id": "Prompt", "level_fr": "Niveau",
            "energy_wh": "Énergie (Wh)", "score": "Note"}), hide_index=True, width="stretch")

    # ---------------- 6. Votes vs grille ----------------
    with tabs[5]:
        v = df.assign(vote=df["comparia_vote"].str.lower())

        def rates(key):
            g = v.groupby(key)["vote"].value_counts().unstack(fill_value=0)
            for k in ("win", "tie", "loss"):
                if k not in g:
                    g[k] = 0
            n = g[["win", "tie", "loss"]].sum(axis=1)
            return pd.DataFrame({"Duels": n, "Victoires": g["win"], "Nuls": g["tie"], "Défaites": g["loss"],
                                 "Taux de victoire": g["win"] / n, "Taux de nul": g["tie"] / n})

        pct = {"Taux de victoire": st.column_config.NumberColumn(format="percent"),
               "Taux de nul": st.column_config.NumberColumn(format="percent")}
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Par classe")
            st.dataframe(rates("cls").reindex([c for c in CLASSES if c in set(v["cls"])]),
                         width="stretch", column_config=pct)
        with c2:
            st.subheader("Par modèle")
            st.dataframe(rates("model_name"), width="stretch", column_config=pct)

        duels = duel_outcomes(df)
        if duels.empty:
            st.info("Aucun duel complet dans la sélection (les deux modèles doivent être cochés).")
        else:
            n, agree = len(duels), int(duels["accord"].sum())
            decisive = duels[duels["gagnant_vote"].isin(["Égalité", "?"]) == False]
            k1, k2, k3 = st.columns(3)
            k1.metric("Accord vote / grille", f"{agree}/{n}", f"{agree / n:.0%}", delta_color="off")
            k2.metric("Votes tranchés", f"{len(decisive)}/{n}")
            k3.metric("Votes tranchés en faveur du plus grand modèle",
                      f"{int(decisive['vote_pour_le_plus_grand'].sum())}/{len(decisive)}")
            st.caption("Accord = le gagnant du vote (ou l'égalité) coïncide avec la meilleure note "
                       "à la grille (ou l'égalité de notes).")
            st.subheader("Désaccords")
            dis = duels[~duels["accord"]].drop(columns=["accord", "vote_pour_le_plus_grand"], errors="ignore")
            dis = dis.drop(columns=[c for c in ["source_file"] if dis.get(c, pd.Series()).nunique() <= 1])
            st.dataframe(dis.rename(columns={
                "pair_id": "Binôme", "prompt_id": "Prompt", "duel": "Duel", "niveau": "Niveau",
                "modele_1": "Modèle 1", "note_1": "Note 1", "modele_2": "Modèle 2", "note_2": "Note 2",
                "gagnant_vote": "Gagnant vote", "gagnant_grille": "Gagnant grille"}),
                hide_index=True, width="stretch")

    # ---------------- 7. Détail par prompt ----------------
    with tabs[6]:
        pid = st.selectbox("Prompt", [p for p in PROMPTS if p in set(df["prompt_id"])])
        sub = df[df["prompt_id"] == pid]
        if "pair_id" in sub and sub["pair_id"].nunique() > 1:
            pair = st.selectbox("Binôme", sorted(sub["pair_id"].unique()))
            sub = sub[sub["pair_id"] == pair]
        st.markdown(f"**{pid} · niveau {LEVEL_FR.get(sub['level'].iloc[0], '')}**")
        for duel in ["A", "B", "C"]:
            d = sub[sub["duel"] == duel].sort_values("cls_order")
            if d.empty:
                continue
            st.markdown(f"#### Duel {duel}")
            cols = st.columns(2)
            for col, (_, r) in zip(cols, d.iterrows()):
                with col.container(border=True):
                    crit = " · ⚠️ erreur critique" if r["critical_error"] == 1 else ""
                    st.markdown(f"**{r['model_name']}** (classe {r['cls']}, {r.get('position', '')}) · "
                                f"vote : {r['comparia_vote']}{crit}")
                    st.markdown(
                        f"Note **{fmt(r['score'], 1)}/10** · exactitude {int(r['score_correctness'])}/4 · "
                        f"complétude {int(r['score_completeness'])}/2 · consignes {int(r['score_instruction'])}/2 · "
                        f"clarté {int(r['score_clarity'])}/2")
                    st.caption(f"Tokens : {fmt(r['tokens_total'], 0)} (sortie {fmt(r['output_tokens'], 0)}) · "
                               f"énergie {fmt(r['energy_wh'], 3)} Wh · coût {fmt(r['cost_eur'], 6)} €")
                    if isinstance(r.get("notes"), str) and r["notes"].strip():
                        st.info(r["notes"])
                    with st.container(height=320):
                        st.markdown(str(r["answer_text"]))

    # ---------------- 8. Contrôles du CDC ----------------
    with tabs[7]:
        st.caption("Contrôles sur les données chargées, hors filtres.")
        ch = run_checks(raw)
        ch["OK"] = ch["OK"].map(lambda b: "✅" if b else "❌")
        st.dataframe(ch, hide_index=True, width="stretch")
        if (ch["OK"] == "✅").all():
            st.success("Tous les contrôles sont validés.")
        else:
            st.error("Certains contrôles échouent : corriger le journal.")

        st.subheader("Synthèse niveau × classe")
        summary = lc.drop(columns=["level"]).rename(columns={"classe": "classe"})
        summary.insert(0, "seuil", threshold)
        st.dataframe(summary, hide_index=True, width="stretch")
        st.download_button("Télécharger le CSV de synthèse",
                           summary.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                           file_name="lab1_synthese.csv", mime="text/csv")
        st.caption("Synthèse calculée sur la sélection courante (filtres et seuil de la barre latérale). "
                   "Export de référence (sans filtre, seuils 6, 7 et 8) : `python3 export_synthese.py`.")

    # ---------------- 9. Recommandations ----------------
    with tabs[8]:
        import reco
        st.caption(f"Calculé sur toutes les données chargées, hors filtres, seuil fixe {reco.THRESHOLD:g}/10. "
                   "Règle : plus petite classe ≥ seuil sans erreur critique ; sinon classe S avec "
                   "contrôle automatique obligatoire (⚠️).")
        rc = reco.build_recommendations(raw)
        shown = rc.assign(classe_recommandee=[
            c if ok else f"{c} ⚠️" for c, ok in zip(rc["classe_recommandee"], rc["classe_passe_le_seuil"])])
        st.dataframe(shown.drop(columns=["classe_passe_le_seuil", "justification"]).rename(columns={
            "type_tache": "Tâche", "prompts": "Prompts", "classe_recommandee": "Classe",
            "modele_recommande": "Modèle", "note_moyenne": "Note classe", "erreurs_critiques": "EC",
            "energie_moy_wh": "Énergie moy. (Wh)", "economie_vs_xl": "Économie vs XL", "condition": "Condition"}),
            hide_index=True, width="stretch", column_config={
                "Note classe": st.column_config.NumberColumn(format="%.2f"),
                "Énergie moy. (Wh)": st.column_config.NumberColumn(format="%.3f"),
                "Économie vs XL": st.column_config.NumberColumn(format="percent")})
        for r in rc.itertuples():
            st.markdown(f"- **{r.type_tache}** : {r.justification}")
        md = reco.to_markdown(rc, raw)
        with st.expander("Recommandation globale, à éviter, limites"):
            st.markdown("## Recommandation globale" + md.split("## Recommandation globale", 1)[-1])
        c1, c2 = st.columns(2)
        c1.download_button("Télécharger recommandations.csv",
                           rc.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                           file_name="recommandations.csv", mime="text/csv")
        c2.download_button("Télécharger recommandations.md", md.encode("utf-8"),
                           file_name="recommandations.md", mime="text/markdown")


if __name__ == "__main__":
    main()
