# Ocean Genomes — Minimal, Insight-First Dashboard

from __future__ import annotations
import io
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------
# Config & Constants
# ----------------------------
st.set_page_config(
    page_title="Ocean Genomes — Research Progress",
    page_icon="🐟",
    layout="wide",
    menu_items={
        "Get Help": None,
        "Report a Bug": None,
        "About": None
    }
)

# Expected column names (update here if schema changes)
@dataclass(frozen=True)
class COLS:
    TAXON_ID: str = "ncbi_taxon_id"
    SPECIES: str = "species_canonical"
    GENUS: str = "Genus"
    FAMILY: str = "family_x"
    ORDER: str = "order"
    TARGET_LIST_STATUS: str = "target_list_status"
    SEQUENCING_STATUS: str = "sequencing_status"
    RANK: str = "rank"
    MODIFIED: str = "modified"
    DEMERS_PELAG: str = "DemersPelag"
    IS_MARINE: str = "isMarine"
    IS_BRACKISH: str = "isBrackish"
    IS_FRESHWATER: str = "isFreshwater"
    IS_TERRESTRIAL: str = "isTerrestrial"
    IS_EXTINCT: str = "isExtinct"
    DEPTH_MIN: str = "depth_min_in_m"
    DEPTH_MAX: str = "depth_max_in_m"
    LENGTH_MAX: str = "length_max_in_cm"

STATUS_ORDER: List[str] = [
    "not_started",
    "sample_acquired",
    "data_generation",
    "in_assembly",
    "insdc_open",
]

STATUS_LABELS = {
    "not_started": "Not started",
    "sample_acquired": "Sample acquired",
    "data_generation": "Data generation",
    "in_assembly": "In assembly",
    "insdc_open": "Open in INSDC",
}

HABITAT_PRIORITY = [
    (COLS.IS_MARINE, "marine"),
    (COLS.IS_BRACKISH, "brackish"),
    (COLS.IS_FRESHWATER, "freshwater"),
    (COLS.IS_TERRESTRIAL, "terrestrial"),
]

# ----------------------------
# Utilities
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data(default_path: str = "final_species.csv", uploaded_file=None) -> pd.DataFrame:
    """Load the dataset either from local path or an uploaded file."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(default_path)
    return df

def normalize_bool_series(s: pd.Series) -> pd.Series:
    """Return a clean boolean-int (0/1) series from mixed/NaN values."""
    return (
        s.fillna(0)
         .replace({"True": 1, "False": 0, True: 1, False: 0})
         .astype(int)
    )

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean columns, harmonize statuses, and add helpful features."""
    df = df.copy()

    # Normalize sequencing status
    if COLS.SEQUENCING_STATUS in df.columns:
        df[COLS.SEQUENCING_STATUS] = (
            df[COLS.SEQUENCING_STATUS]
            .fillna("-")
            .replace({"-": "not_started"})
            .str.strip()
            .str.lower()
        )

    # Normalize habitat flags
    for col in [COLS.IS_MARINE, COLS.IS_BRACKISH, COLS.IS_FRESHWATER, COLS.IS_TERRESTRIAL, COLS.IS_EXTINCT]:
        if col in df.columns:
            df[col] = normalize_bool_series(df[col])

    # Primary habitat label
    def primary_hab(row) -> str:
        present = [name for col, name in HABITAT_PRIORITY if row.get(col, 0) == 1]
        if len(present) == 0:
            return "unknown"
        if len(present) == 1:
            return present[0]
        for col, name in HABITAT_PRIORITY:
            if row.get(col, 0) == 1:
                return f"{name}_mixed"
        return "mixed"

    df["habitat_primary"] = df.apply(primary_hab, axis=1)

    # Depth & length helpers
    if COLS.DEPTH_MAX in df.columns and COLS.DEPTH_MIN in df.columns:
        df["depth_range_m"] = df[COLS.DEPTH_MAX] - df[COLS.DEPTH_MIN]

    if COLS.LENGTH_MAX in df.columns:
        bins = [0, 30, 100, 1000]
        labels = ["small (<30cm)", "medium (30–100cm)", "large (>100cm)"]
        df["length_class"] = pd.cut(df[COLS.LENGTH_MAX], bins=bins, labels=labels, include_lowest=True)

    # Taxonomic null-safety
    for col in [COLS.ORDER, COLS.FAMILY, COLS.GENUS]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Species name as string
    if COLS.SPECIES in df.columns:
        df[COLS.SPECIES] = df[COLS.SPECIES].astype(str)

    return df

def compute_kpis(df: pd.DataFrame) -> Tuple[int, int, int, float]:
    """Return (total, in_progress, completed, completion_pct)."""
    total = len(df)
    completed = int((df[COLS.SEQUENCING_STATUS] == "insdc_open").sum())
    in_progress = int(df[COLS.SEQUENCING_STATUS].isin(["sample_acquired", "data_generation", "in_assembly"]).sum())
    completion_pct = (completed / total * 100) if total else 0.0
    return total, in_progress, completed, completion_pct

def family_progress_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summary of progress by family; sorted by families with any activity."""
    fam = (
        df.groupby(COLS.FAMILY, dropna=False)
          .agg(
              total=(COLS.SPECIES, "count"),
              completed=(COLS.SEQUENCING_STATUS, lambda s: (s == "insdc_open").sum()),
              in_progress=(COLS.SEQUENCING_STATUS, lambda s: s.isin(["sample_acquired","data_generation","in_assembly"]).sum()),
          )
          .reset_index()
    )
    fam["coverage_pct"] = np.where(fam["total"] > 0, fam["completed"] / fam["total"] * 100, 0.0)
    fam_active = fam[(fam["completed"] > 0) | (fam["in_progress"] > 0)].copy()
    fam_active.sort_values(["completed", "in_progress", "total"], ascending=[False, False, True], inplace=True)
    return fam, fam_active

def plot_progress_funnel(df: pd.DataFrame) -> go.Figure:
    counts = (
        df[COLS.SEQUENCING_STATUS]
        .value_counts()
        .reindex(STATUS_ORDER)
        .fillna(0)
        .astype(int)
    )
    fig = px.bar(
        counts.reset_index(),
        x=counts.values,
        y=[STATUS_LABELS.get(s, s) for s in counts.index],
        orientation="h",
        text=counts.values,
    )
    fig.update_layout(
        xaxis_title="Number of species",
        yaxis_title="Research stage",
        showlegend=False,
        margin=dict(l=0, r=10, t=10, b=10),
        height=360,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig

def plot_family_activity(df_active: pd.DataFrame) -> go.Figure:
    if df_active.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No families with in-progress or completed sequencing in current filter.",
            height=80,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        return fig

    top = df_active.head(12)
    top = top.melt(id_vars=[COLS.FAMILY, "total", "coverage_pct"], value_vars=["in_progress", "completed"],
                   var_name="stage", value_name="count")

    fig = px.bar(
        top,
        x="count",
        y=COLS.FAMILY,
        color="stage",
        orientation="h",
        barmode="stack",
        category_orders={COLS.FAMILY: list(top.sort_values("total", ascending=True)[COLS.FAMILY])},
        labels={"count": "Species", COLS.FAMILY: "Family"},
    )
    fig.update_layout(
        legend_title="Stage",
        margin=dict(l=0, r=10, t=10, b=10),
        height=420,
    )
    return fig

def plot_habitat_progress_share(df: pd.DataFrame) -> go.Figure:
    if "habitat_primary" not in df.columns:
        return go.Figure()
    keep = ["marine", "marine_mixed", "brackish", "freshwater"]
    tmp = df.copy()
    tmp["habitat_display"] = np.where(tmp["habitat_primary"].isin(keep), tmp["habitat_primary"], "other/unknown")
    pivot = (
        tmp.pivot_table(index="habitat_display", columns=COLS.SEQUENCING_STATUS, values=COLS.SPECIES, aggfunc="count")
          .reindex(columns=STATUS_ORDER)
          .fillna(0)
    )
    share = pivot.div(pivot.sum(axis=1), axis=0).reset_index().melt(id_vars="habitat_display", var_name="stage", value_name="share")
    share["stage"] = share["stage"].map(STATUS_LABELS).fillna(share["stage"])
    fig = px.bar(
        share,
        x="habitat_display",
        y="share",
        color="stage",
        barmode="stack",
        labels={"habitat_display": "Habitat (primary)", "share": "Within-habitat share"},
    )
    fig.update_layout(
        yaxis_tickformat=".0%",
        margin=dict(l=0, r=10, t=10, b=10),
        height=360,
    )
    return fig

def to_csv_download(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def figure_to_png_bytes(fig: go.Figure) -> bytes | None:
    """Render plotly figure to PNG bytes (requires kaleido)."""
    try:
        import plotly.io as pio
        return pio.to_image(fig, format="png", width=1280, height=720, scale=2)
    except Exception:
        return None

# ---------- Extra visuals helpers (existing) ----------
def _ensure_midpoints(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if COLS.DEPTH_MIN in d.columns and COLS.DEPTH_MAX in d.columns:
        d["depth_mid_m"] = pd.to_numeric(d[COLS.DEPTH_MIN], errors="coerce") + pd.to_numeric(d[COLS.DEPTH_MAX], errors="coerce")
        d["depth_mid_m"] = d["depth_mid_m"] / 2.0
    return d

def plot_stage_share_by_category(df: pd.DataFrame, category: str, title: str, top_n: int = 8) -> go.Figure:
    if category not in df.columns:
        return go.Figure()
    tmp = df.copy()
    tmp[category] = tmp[category].fillna("Unknown")
    top = tmp[category].value_counts().head(top_n).index.tolist()
    tmp[category] = np.where(tmp[category].isin(top), tmp[category], "Other")
    mix = (tmp.pivot_table(index=category, columns=COLS.SEQUENCING_STATUS, values=COLS.SPECIES, aggfunc="count")
                 .reindex(columns=STATUS_ORDER).fillna(0))
    share = (mix.div(mix.sum(axis=1), axis=0)
                  .reset_index()
                  .melt(id_vars=category, var_name="stage", value_name="share"))
    share["stage_lbl"] = share["stage"].map(STATUS_LABELS).fillna(share["stage"])
    fig = px.bar(share, x=category, y="share", color="stage_lbl", barmode="stack",
                 title=title, labels={"share":"Within-category share"})
    fig.update_layout(yaxis_tickformat=".0%", margin=dict(l=0, r=10, t=40, b=10), height=360, legend_title="Stage")
    return fig

def plot_flag_habitat_overview(df: pd.DataFrame) -> go.Figure:
    flags = [(COLS.IS_MARINE,"Marine"), (COLS.IS_BRACKISH,"Brackish"),
             (COLS.IS_FRESHWATER,"Freshwater"), (COLS.IS_TERRESTRIAL,"Terrestrial")]
    rows = []
    for col,label in flags:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            rows.append({"Habitat": label, "Count": int(s.sum())})
    if not rows:
        return go.Figure()
    dat = pd.DataFrame(rows)
    fig = px.bar(dat, x="Habitat", y="Count", title="Primary habitat flags (count of taxa with flag = 1)")
    fig.update_layout(margin=dict(l=0, r=10, t=40, b=10), height=320, showlegend=False)
    return fig

def plot_length_vs_depth_density(df: pd.DataFrame) -> go.Figure:
    d = _ensure_midpoints(df)
    if ("depth_mid_m" not in d.columns) or (COLS.LENGTH_MAX not in d.columns):
        return go.Figure()
    d = d[["depth_mid_m", COLS.LENGTH_MAX, COLS.SPECIES, COLS.ORDER, COLS.FAMILY, COLS.GENUS]].dropna()
    if d.empty:
        return go.Figure()
    fig = px.density_heatmap(d, x="depth_mid_m", y=COLS.LENGTH_MAX, nbinsx=40, nbinsy=40,
                             labels={"depth_mid_m":"Depth midpoint (m)", COLS.LENGTH_MAX:"Max length (cm)"},
                             hover_data=[COLS.SPECIES, COLS.ORDER, COLS.FAMILY, COLS.GENUS])
    fig.update_layout(margin=dict(l=0, r=10, t=10, b=10), height=420)
    return fig

def plot_trait_histograms(df: pd.DataFrame) -> go.Figure:
    has_len = COLS.LENGTH_MAX in df.columns
    has_range = "depth_range_m" in df.columns
    if not (has_len or has_range):
        return go.Figure()
    from plotly.subplots import make_subplots
    cols = (1 if has_len else 0) + (1 if has_range else 0)
    fig = make_subplots(rows=1, cols=cols, subplot_titles=[
        "Max length (cm)" if has_len else None,
        "Depth range (m)" if has_range else None
    ])
    c = 1
    if has_len:
        s = pd.to_numeric(df[COLS.LENGTH_MAX], errors="coerce").dropna()
        if len(s):
            fig.add_trace(go.Histogram(x=s, nbinsx=40, name="Max length (cm)"), row=1, col=c)
        c += 1
    if has_range:
        s2 = pd.to_numeric(df.get("depth_range_m"), errors="coerce").dropna()
        if len(s2):
            fig.add_trace(go.Histogram(x=s2, nbinsx=40, name="Depth range (m)"), row=1, col=c)
    fig.update_layout(height=360, margin=dict(l=0, r=10, t=40, b=10), showlegend=False)
    return fig

# ----------------------------
# NEW: Temperature-bins helpers (ported & adapted)
# ----------------------------
def _norm(s: str) -> str:
    """Normalize a column name for fuzzy matching."""
    return (
        s.strip()
         .lower()
         .replace(" ", "_")
         .replace("/", "_")
         .replace("-", "_")
    )

def _find_first(df: pd.DataFrame, candidates: List[str]) -> str | None:
    """Return the first matching column (case/sep insensitive) or None."""
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        if _norm(cand) in norm_map:
            return norm_map[_norm(cand)]
    return None

@st.cache_data(show_spinner=False)
def prep_temp_stats(df_in: pd.DataFrame) -> pd.DataFrame | None:
    """
    Prepare per-AphiaID (or species fallback) temperature stats for binned charts.
    Returns None if required columns are unavailable.
    """
    df = df_in.copy()

    # Candidate columns across different schemas
    col_tmin = _find_first(df, ["tempmin","temp_min","temperature_min","tmin"])
    col_tmax = _find_first(df, ["tempmax","temp_max","temperature_max","tmax"])
    col_species = _find_first(df, [COLS.SPECIES, "species","species_canonical","scientificname","scientific_name"])
    col_status = _find_first(df, [COLS.SEQUENCING_STATUS,"sequencial_status","sequence_status","status_sequencing","status"])
    col_aphia = _find_first(df, ["aphia_id","aphiaid", COLS.TAXON_ID, "taxon_id"])

    # Must have at least one temp column and a species/ID
    if (col_species is None) or (col_aphia is None) or (col_tmin is None and col_tmax is None):
        return None

    tmin = pd.to_numeric(df[col_tmin], errors="coerce") if col_tmin else None
    tmax = pd.to_numeric(df[col_tmax], errors="coerce") if col_tmax else None

    if tmin is not None and tmax is not None:
        temp = np.where(~tmin.isna() & ~tmax.isna(), (tmin + tmax)/2.0,
                        np.where(~tmin.isna(), tmin, tmax))
    elif tmin is not None:
        temp = tmin.to_numpy()
    else:
        temp = (tmax.to_numpy() if tmax is not None else np.full(len(df), np.nan))

    base = pd.DataFrame({
        "entity_id": df[col_aphia].astype(str),
        "species": df[col_species].astype(str),
        "tavg": pd.to_numeric(temp, errors="coerce")
    })
    # Status if available; else "All"
    if col_status:
        base["status"] = (
            df[col_status].astype(str)
              .replace(["nan","None",""," "], np.nan)
              .fillna("unknown")
        )
    else:
        base["status"] = "All"

    base = base.dropna(subset=["tavg"])
    # Reasonable bounds; tweak if needed
    base = base[(base["tavg"] > -5) & (base["tavg"] < 60)]

    stats = (
        base.groupby(["status","entity_id","species"])["tavg"]
            .agg(tmin="min", tavg="mean", tmax="max", n="count")
            .reset_index()
    )
    return stats

def _make_bins(series: pd.Series, width: float = 2.0) -> np.ndarray:
    lo = np.floor(series.min() / width) * width
    hi = np.ceil(series.max() / width) * width
    return np.arange(lo, hi + width + 1e-9, width)

def _binned_counts(stats: pd.DataFrame, bin_edges: np.ndarray, status_filter: str = "All", entity_filter: str | None = None) -> pd.DataFrame:
    df = stats if status_filter == "All" else stats[stats["status"] == status_filter]
    if entity_filter:
        df = df[df["entity_id"].astype(str) == str(entity_filter).strip()]
    df = df.copy()
    df["bin_idx"] = np.digitize(df["tavg"], bin_edges, right=True) - 1
    df["bin_idx"] = df["bin_idx"].clip(0, len(bin_edges)-2)
    counts = df.groupby("bin_idx")["entity_id"].nunique()
    counts = counts.reindex(range(len(bin_edges)-1), fill_value=0)
    out = []
    for i in range(len(bin_edges)-1):
        out.append({
            "bin": f"{int(bin_edges[i])}–{int(bin_edges[i+1])}",
            "left": float(bin_edges[i]),
            "right": float(bin_edges[i+1]),
            "count": int(counts.iloc[i])
        })
    return pd.DataFrame(out)

def make_temp_chart(stats: pd.DataFrame, bin_edges: np.ndarray, status_sel: str = "All", entity_sel: str | None = None) -> go.Figure:
    data = _binned_counts(stats, bin_edges, status_sel, entity_sel)
    fig = go.Figure(go.Bar(
        x=data["bin"],
        y=data["count"],
        customdata=np.c_[data["left"], data["right"]],
        hovertemplate=(
            "Temperature bin: %{customdata[0]:.0f}–%{customdata[1]:.0f} °C<br>"
            "Distinct IDs: %{y}"
        )
    ))
    title_suffix = f"Status={status_sel}, ID={entity_sel or 'All'}"
    fig.update_layout(
        title=f"Species per 2 °C bin — {title_suffix}",
        xaxis_title="Average temperature bin (°C)",
        yaxis_title="Distinct entities",
        margin=dict(l=10, r=10, t=40, b=10),
        height=380
    )
    return fig

# ----------------------------
# Sidebar — Controls
# ----------------------------
st.sidebar.header("⚙️ Data & Filters")

uploaded = st.sidebar.file_uploader("Upload client dataset CSV", type=["csv"], accept_multiple_files=False)
path_hint = st.sidebar.text_input("…or path to CSV", value="final_species.csv", help="If left as default, the app will look for final_species.csv in the working directory.")

# Safe loader
data = None
load_error = None
try:
    data = load_data(default_path=path_hint, uploaded_file=uploaded)
except FileNotFoundError as e:
    load_error = f"File not found: {e}"
except Exception as e:
    load_error = f"Failed to load CSV: {e}"

if load_error:
    st.error(load_error)
    st.stop()

df = add_features(data)

# Filters
with st.sidebar:
    st.subheader("Filter records")

    # Taxonomy filters
    orders = sorted(df[COLS.ORDER].dropna().unique().tolist()) if COLS.ORDER in df.columns else []
    families = sorted(df[COLS.FAMILY].dropna().unique().tolist()) if COLS.FAMILY in df.columns else []
    genera = sorted(df[COLS.GENUS].dropna().unique().tolist()) if COLS.GENUS in df.columns else []

    sel_orders = st.multiselect("Order", options=orders)
    sel_families = st.multiselect("Family", options=families)
    sel_genera = st.multiselect("Genus", options=genera)

    # Status filters
    status_options = [STATUS_LABELS.get(s, s) for s in STATUS_ORDER]
    sel_status = st.multiselect(
        "Research stage",
        options=status_options,
        default=[s for s in status_options if s != "Not started"]
    )

    # Target list filter
    tl_values = sorted(df[COLS.TARGET_LIST_STATUS].dropna().unique().tolist()) if COLS.TARGET_LIST_STATUS in df.columns else []
    sel_target_list = st.multiselect("Target list status", options=tl_values, default=tl_values)

    # Habitat filter
    habitats = ["marine", "marine_mixed", "brackish", "freshwater", "terrestrial", "unknown"]
    sel_habitats = st.multiselect("Primary habitat", options=habitats, default=habitats)

    # Free text search (species)
    q = st.text_input("Search species name", value="", placeholder="e.g., Lethrinus nebulosus")

# Apply filters
mask = pd.Series(True, index=df.index)
if sel_orders:
    mask &= df[COLS.ORDER].isin(sel_orders)
if sel_families:
    mask &= df[COLS.FAMILY].isin(sel_families)
if sel_genera:
    mask &= df[COLS.GENUS].isin(sel_genera)
if sel_status:
    reverse_map = {v: k for k, v in STATUS_LABELS.items()}
    stages = [reverse_map.get(v, v) for v in sel_status]
    mask &= df[COLS.SEQUENCING_STATUS].isin(stages)
if sel_target_list:
    mask &= df[COLS.TARGET_LIST_STATUS].isin(sel_target_list)
if sel_habitats:
    mask &= df["habitat_primary"].isin(sel_habitats)
if q.strip():
    mask &= df[COLS.SPECIES].str.contains(q.strip(), case=False, na=False)

df_f = df[mask].copy()

# ----------------------------
# Header / KPIs
# ----------------------------
st.title("🐟 Ocean Genomes — Research Progress Dashboard")
st.caption("Fewer, clearer visuals. Designed for quick, defensible decisions.")

total, in_prog, completed, pct = compute_kpis(df_f)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Species (filtered)", f"{total:,}")
k2.metric("In progress", f"{in_prog:,}")
k3.metric("Completed (INSDC)", f"{completed:,}")
k4.metric("Completion rate", f"{pct:.2f}%")

# ----------------------------
# Plot 1 — Research Progress Funnel
# ----------------------------
st.subheader("1) Research progress to date")
fig1 = plot_progress_funnel(df_f)
st.plotly_chart(fig1, use_container_width=True)
st.caption("Counts of species at each stage. Ordered from not started → INSDC open.")
png1 = figure_to_png_bytes(fig1)
if png1:
    st.download_button("⬇️ Download this figure (PNG)", data=png1, file_name="progress_funnel.png", mime="image/png")

st.markdown("---")

# ----------------------------
# Plot 2 — Families with activity
# ----------------------------
st.subheader("2) Families with activity (in progress or completed)")
fam_all, fam_active = family_progress_table(df_f)
fig2 = plot_family_activity(fam_active)
st.plotly_chart(fig2, use_container_width=True)
st.caption("Top families showing any in-progress or completed species in the current filter.")
png2 = figure_to_png_bytes(fig2)
if png2:
    st.download_button("⬇️ Download this figure (PNG)", data=png2, file_name="family_activity.png", mime="image/png")

st.markdown("---")

# ----------------------------
# Completed species table (+ download)
# ----------------------------
st.subheader("Completed species (INSDC open)")
done = df_f[df_f[COLS.SEQUENCING_STATUS] == "insdc_open"].sort_values(COLS.FAMILY) if COLS.SEQUENCING_STATUS in df_f.columns else pd.DataFrame()
if done.empty:
    st.info("No completed species in the current filter.")
else:
    show_cols = [c for c in [COLS.SPECIES, COLS.GENUS, COLS.FAMILY, COLS.ORDER] if c in done.columns]
    st.dataframe(done[show_cols], use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download completed species (CSV)",
        data=to_csv_download(done[show_cols]),
        file_name="completed_species.csv",
        mime="text/csv"
    )

# ----------------------------
# Optional: Download filtered data
# ----------------------------
with st.expander("Optional: Download filtered dataset"):
    st.download_button(
        "⬇️ Download filtered CSV",
        data=to_csv_download(df_f),
        file_name="filtered_ocean_genomes.csv",
        mime="text/csv"
    )

# ----------------------------
# Plot 3 — Taxonomic Explorer (sunburst)
# ----------------------------
st.subheader("3) Taxonomic Explorer — click to drill into taxa")
st.caption("Interactive, circle-like view of taxonomy. Click a sector to zoom; breadcrumb to go back.")

def build_taxonomy_sunburst(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title="No records in the current filter.", height=120, margin=dict(l=0, r=0, t=30, b=0))
        return fig

    levels = [c for c in [COLS.ORDER, COLS.FAMILY, COLS.GENUS, COLS.SPECIES] if c in df.columns]
    if len(levels) < 2:
        fig = go.Figure()
        fig.update_layout(title="Need at least two taxonomy levels (e.g., Order and Family).", height=120, margin=dict(l=0, r=0, t=30, b=0))
        return fig

    def completed_mask(s: pd.Series) -> pd.Series:
        return s.eq("insdc_open")

    def inprog_mask(s: pd.Series) -> pd.Series:
        return s.isin(["sample_acquired", "data_generation", "in_assembly"])

    parts = []
    for depth in range(1, len(levels)+1):
        group_cols = levels[:depth]
        grp = (
            df.groupby(group_cols, dropna=False)[COLS.SEQUENCING_STATUS]
              .agg(
                  total="size",
                  completed=lambda s: int(completed_mask(s).sum()),
                  in_progress=lambda s: int(inprog_mask(s).sum()),
              )
              .reset_index()
        )
        grp["level"] = depth
        parts.append(grp)

    nodes = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if nodes.empty:
        fig = go.Figure()
        fig.update_layout(title="No taxonomy nodes to display for the current filter.", height=120, margin=dict(l=0, r=0, t=30, b=0))
        return fig

    def make_id(row):
        keys = [str(row.get(col, "Unknown")) for col in levels[:int(row["level"])]]
        return " / ".join(keys)

    def make_parent(row):
        if int(row["level"]) == 1:
            return ""
        keys = [str(row.get(col, "Unknown")) for col in levels[:int(row["level"]) - 1]]
        return " / ".join(keys)

    nodes = nodes.assign(
        id=nodes.apply(make_id, axis=1).astype(str).to_numpy(),
        parent=nodes.apply(make_parent, axis=1).astype(str).to_numpy()
    )

    nodes["coverage_pct"] = (
        nodes["completed"] / nodes["total"].replace(0, np.nan) * 100
    ).fillna(0).round(2)

    nodes["label"] = nodes.apply(lambda r: str(r[levels[int(r["level"]) - 1]]), axis=1)
    nodes["value"] = nodes["total"]
    nodes["hover"] = (
        nodes["id"].astype(str)
        + "<br><b>Total</b>: " + nodes["total"].astype(int).astype(str)
        + "<br><b>Completed</b>: " + nodes["completed"].astype(int).astype(str)
        + "<br><b>In progress</b>: " + nodes["in_progress"].astype(int).astype(str)
        + "<br><b>Completion</b>: " + nodes["coverage_pct"].astype(str) + "%"
    )

    fig = go.Figure(go.Sunburst(
        ids=nodes["id"],
        labels=nodes["label"],
        parents=nodes["parent"],
        values=nodes["value"],
        branchvalues="total",
        hovertext=nodes["hover"],
        hoverinfo="text",
        maxdepth=None,
        insidetextorientation="radial",
        marker=dict(line=dict(width=0.5))
    ))
    fig.update_traces(marker=dict(colors=nodes["coverage_pct"], colorbar=dict(title="Completion %")), selector=dict(type="sunburst"))
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=10), height=520)
    return fig

fig4 = build_taxonomy_sunburst(df_f)
st.plotly_chart(fig4, use_container_width=True)
png4 = figure_to_png_bytes(fig4)
if png4:
    st.download_button("⬇️ Download taxonomy sunburst (PNG)", data=png4, file_name="taxonomy_sunburst.png", mime="image/png")

st.markdown("---")

# ----------------------------
# Plot 4 — Species by Average Temperature (2 °C bins)  <-- NEW
# ----------------------------
st.subheader("4) Species by Average Temperature (2 °C bins)")
st.caption("Counts of distinct IDs per 2 °C bin. Appears when temperature columns are available.")
temp_stats = prep_temp_stats(df_f)

if temp_stats is None or temp_stats.empty:
    st.info("Temperature columns not detected (e.g., temp_min / temp_max). Add them to enable this view.")
else:
    # Status control (include 'All' option)
    statuses = ["All"] + sorted([s for s in temp_stats["status"].unique() if s != "All"])
    status_sel = st.selectbox("Status", statuses, index=0, key="temp_status")

    # Entity control (AphiaID / taxon_id / fallback)
    ent_options = ["All"] + sorted(
        temp_stats[temp_stats["status"].eq(status_sel) | (status_sel == "All")]["entity_id"].astype(str).unique(),
        key=lambda x: (len(x), x)
    )
    ent_sel = st.selectbox("Entity ID", ent_options, index=0, key="temp_entity")
    selected_entity = None if ent_sel == "All" else ent_sel

    # Build chart
    bin_edges = _make_bins(temp_stats["tavg"], width=2.0)
    fig_temp = make_temp_chart(temp_stats, bin_edges, status_sel, selected_entity)
    st.plotly_chart(fig_temp, use_container_width=True)

    png_temp = figure_to_png_bytes(fig_temp)
    if png_temp:
        st.download_button("⬇️ Download temperature bins (PNG)", data=png_temp, file_name="temperature_bins.png", mime="image/png")

# ----------------------------
# Notes / Footers
# ----------------------------
with st.expander("About this dashboard & data assumptions"):
    st.markdown("""
    - **Status definitions** are interpreted from the `sequencing_status` column:  
      `not_started`, `sample_acquired`, `data_generation`, `in_assembly`, `insdc_open`.
    - **Primary habitat** is prioritized as marine → brackish → freshwater → terrestrial; species with multiple habitats are labeled `"*_mixed"`.
    - **Length classes** are derived from `length_max_in_cm` and used only for exploratory filters in future iterations.
    - Temperature bins view appears when temperature columns (e.g., `temp_min`/`temp_max`) exist.
    - The visuals are intentionally minimal to prioritize **clarity**.
    """)
