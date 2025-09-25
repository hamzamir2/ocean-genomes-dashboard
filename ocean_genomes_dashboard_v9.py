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
    menu_items={  # 👈 hides "Fork this repo" and other default links
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

    # Primary habitat label (prioritize marine > brackish > freshwater > terrestrial)
    def primary_hab(row) -> str:
        present = [name for col, name in HABITAT_PRIORITY if row.get(col, 0) == 1]
        if len(present) == 0:
            return "unknown"
        if len(present) == 1:
            return present[0]
        # Multiple habitats → pick first by priority & label as mixed
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

    # Species name as string (for search/filter)
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
    # Focus on families with some activity (in_progress or completed)
    fam_active = fam[(fam["completed"] > 0) | (fam["in_progress"] > 0)].copy()
    fam_active.sort_values(["completed", "in_progress", "total"], ascending=[False, False, True], inplace=True)
    return fam, fam_active


def plot_progress_funnel(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar showing species counts at each progress stage."""
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
    """Top families where any progress has been made (in_progress or completed)."""
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
    """100% stacked bar chart: share of species by research stage within each primary habitat."""
    if "habitat_primary" not in df.columns:
        return go.Figure()

    # Limit to most common habitats to keep it readable
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



# ---------- Extra visuals helpers (new) ----------
def _ensure_midpoints(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Depth midpoint and range already exist or get recomputed
    if COLS.DEPTH_MIN in d.columns and COLS.DEPTH_MAX in d.columns:
        d["depth_mid_m"] = pd.to_numeric(d[COLS.DEPTH_MIN], errors="coerce") + pd.to_numeric(d[COLS.DEPTH_MAX], errors="coerce")
        d["depth_mid_m"] = d["depth_mid_m"] / 2.0
    return d

def plot_stage_share_by_category(df: pd.DataFrame, category: str, title: str, top_n: int = 8) -> go.Figure:
    """For a categorical feature, show a 100% stacked bar of stage mix for top-N categories (rest grouped as 'Other')."""
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
                 title=title, labels={"share":"Within‑category share"})
    fig.update_layout(yaxis_tickformat=".0%", margin=dict(l=0, r=10, t=40, b=10), height=360, legend_title="Stage")
    return fig

def plot_flag_habitat_overview(df: pd.DataFrame) -> go.Figure:
    """Compact bar of counts by boolean habitat flags (isMarine, isBrackish, isFreshwater, isTerrestrial)."""
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
    """Smoothed density heatmap of max length vs. depth midpoint to avoid clutter."""
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
    """Side-by-side histograms: length_max_in_cm and depth_range_m (if available)."""
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
    orders = sorted(df[COLS.ORDER].dropna().unique().tolist())
    families = sorted(df[COLS.FAMILY].dropna().unique().tolist())
    genera = sorted(df[COLS.GENUS].dropna().unique().tolist())

    sel_orders = st.multiselect("Order", options=orders)
    sel_families = st.multiselect("Family", options=families)
    sel_genera = st.multiselect("Genus", options=genera)

    # Status filters
    status_options = [STATUS_LABELS.get(s, s) for s in STATUS_ORDER]
    sel_status = st.multiselect(
        "Research stage",
        options=status_options,
        default=[s for s in status_options if s != "Not started"]  # exclude Not started
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

# Optional download
png1 = figure_to_png_bytes(fig1)
if png1:
    st.download_button("⬇️ Download this figure (PNG)", data=png1, file_name="progress_funnel.png", mime="image/png")

st.markdown("---")

# ----------------------------
# Plot 2 — Where is progress happening? (Families with activity)
# ----------------------------
st.subheader("2) Families with activity (in progress or completed)")
fam_all, fam_active = family_progress_table(df_f)
fig2 = plot_family_activity(fam_active)
st.plotly_chart(fig2, use_container_width=True)
st.caption("Top families showing any in-progress or completed species in the current filter. Helps highlight where work is concentrated.")

png2 = figure_to_png_bytes(fig2)
if png2:
    st.download_button("⬇️ Download this figure (PNG)", data=png2, file_name="family_activity.png", mime="image/png")

st.markdown("---")

# ----------------------------
# Plot 3 — Progress by habitat (share within habitat)
# ----------------------------
st.subheader("3) Habitat context — progress share within each primary habitat")
fig3 = plot_habitat_progress_share(df_f)
st.plotly_chart(fig3, use_container_width=True)
st.caption("Within each habitat, the proportion of species at each stage (100% stacked). Useful for planning fieldwork or focusing curation efforts.")

png3 = figure_to_png_bytes(fig3)
if png3:
    st.download_button("⬇️ Download this figure (PNG)", data=png3, file_name="habitat_progress_share.png", mime="image/png")

st.markdown("---")

# ----------------------------
# Completed species table (+ download)
# ----------------------------
st.subheader("Completed species (INSDC open)")
done = df_f[df_f[COLS.SEQUENCING_STATUS] == "insdc_open"].sort_values(COLS.FAMILY)
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
# Plot 4 — Taxonomic Explorer (click to drill)
# ----------------------------
st.subheader("4) Taxonomic Explorer — click to drill into taxa")
st.caption("Interactive, circle-like view of taxonomy. Click a sector to zoom in; use the upper-left breadcrumb to go back.")

def build_taxonomy_sunburst(df: pd.DataFrame) -> go.Figure:
    """
    Constructs a sunburst with path [Order → Family → Genus → Species].
    Node color encodes completion rate (INSDC open / total in node).
    Hover shows totals, in-progress, and completed counts.
    """
    # If nothing after filters, return a friendly placeholder
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No records in the current filter.",
            height=120,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        return fig

    # Determine available taxonomy columns in the right order
    levels = [c for c in [COLS.ORDER, COLS.FAMILY, COLS.GENUS, COLS.SPECIES] if c in df.columns]
    if len(levels) < 2:
        fig = go.Figure()
        fig.update_layout(
            title="Need at least two taxonomy levels (e.g., Order and Family).",
            height=120, margin=dict(l=0, r=0, t=30, b=0),
        )
        return fig

    # Define helpers
    def completed_mask(s: pd.Series) -> pd.Series:
        return s.eq("insdc_open")

    def inprog_mask(s: pd.Series) -> pd.Series:
        return s.isin(["sample_acquired", "data_generation", "in_assembly"])

    # Build table with totals/completed/in_progress for each node at each depth
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

    # If still empty (e.g., weird filter), bail out gracefully
    if nodes.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No taxonomy nodes to display for the current filter.",
            height=120, margin=dict(l=0, r=0, t=30, b=0),
        )
        return fig

    # Safe ID/parent builders
    def make_id(row):
        keys = [str(row.get(col, "Unknown")) for col in levels[:int(row["level"])]]
        return " / ".join(keys)

    def make_parent(row):
        if int(row["level"]) == 1:
            return ""
        keys = [str(row.get(col, "Unknown")) for col in levels[:int(row["level"]) - 1]]
        return " / ".join(keys)

    # Force 1-D numpy arrays to avoid the "set_item_frame_value" ValueError
    nodes_ids = nodes.apply(make_id, axis=1).astype(str).to_numpy()
    nodes_parents = nodes.apply(make_parent, axis=1).astype(str).to_numpy()
    nodes = nodes.assign(id=nodes_ids, parent=nodes_parents)

    # Coverage percentage per node
    nodes["coverage_pct"] = (
        nodes["completed"] / nodes["total"].replace(0, np.nan) * 100
    ).fillna(0).round(2)

    # Display label = last taxonomy part at that depth
    nodes["label"] = nodes.apply(lambda r: str(r[levels[int(r["level"]) - 1]]), axis=1)
    nodes["value"] = nodes["total"]

    # Custom hover info
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

    fig.update_traces(
        marker=dict(colors=nodes["coverage_pct"], colorbar=dict(title="Completion %")),
        selector=dict(type="sunburst")
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=10),
        height=520,
    )
    return fig

    # Define helpers
    def completed_mask(s: pd.Series) -> pd.Series:
        return s.eq("insdc_open")

    def inprog_mask(s: pd.Series) -> pd.Series:
        return s.isin(["sample_acquired", "data_generation", "in_assembly"])

    # Build a table with totals/completed/in_progress for each node at each depth
    parts = []
    for depth in range(1, len(levels)+1):
        group_cols = levels[:depth]
        grp = df.groupby(group_cols, dropna=False)[COLS.SEQUENCING_STATUS].agg(
            total="size",
            completed=lambda s: completed_mask(s).sum(),
            in_progress=lambda s: inprog_mask(s).sum(),
        ).reset_index()
        grp["level"] = depth
        parts.append(grp)

    nodes = pd.concat(parts, ignore_index=True)

    # Create unique node ids and parent ids for sunburst
    def make_id(row):
        keys = [str(row.get(col, "Unknown")) for col in levels[:row["level"]]]
        return " / ".join(keys)

    def make_parent(row):
        if row["level"] == 1:
            return ""
        keys = [str(row.get(col, "Unknown")) for col in levels[:row["level"]-1]]
        return " / ".join(keys)

    nodes["id"] = nodes.apply(make_id, axis=1)
    nodes["parent"] = nodes.apply(make_parent, axis=1)

    # Coverage percentage per node
    nodes["coverage_pct"] = (nodes["completed"] / nodes["total"].replace(0, np.nan) * 100).fillna(0).round(2)

    # Display label is the last taxonomy part
    last = levels[-1]
    nodes["label"] = nodes.apply(lambda r: str(r[levels[r["level"]-1]]), axis=1)

    # Values for area
    nodes["value"] = nodes["total"]

    # Custom hover info
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
        marker=dict(
            line=dict(width=0.5)
        )
    ))

    # Color by coverage pct with a continuous colorscale
    fig.update_traces(
        marker=dict(colors=nodes["coverage_pct"], colorbar=dict(title="Completion %")),
        selector=dict(type="sunburst")
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=10),
        height=520,
    )
    return fig

fig4 = build_taxonomy_sunburst(df_f)
st.plotly_chart(fig4, use_container_width=True)

png4 = figure_to_png_bytes(fig4)
if png4:
    st.download_button("⬇️ Download taxonomy sunburst (PNG)", data=png4, file_name="taxonomy_sunburst.png", mime="image/png")

# ----------------------------
# Notes / Footers
# ----------------------------
with st.expander("About this dashboard & data assumptions"):
    st.markdown("""
    - **Status definitions** are interpreted from the `sequencing_status` column:  
      `not_started`, `sample_acquired`, `data_generation`, `in_assembly`, `insdc_open`.
    - **Primary habitat** is prioritized as marine → brackish → freshwater → terrestrial; species with multiple habitats are labeled `"*_mixed"`.
    - **Length classes** are derived from `length_max_in_cm` and used only for exploratory filters in future iterations.
    - The visuals are intentionally minimal to prioritize **clarity**.
    """)
