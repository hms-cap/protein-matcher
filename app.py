"""Main Streamlit Interactive 3D Molecular Puzzle Workbench (app.py)."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import numpy as np

from core.pipeline import LockKeyAnalysisBundle, analyze_classroom_complex
from core.scorer import PuzzleScorecard
from core.viewer_3d import build_interactive_3d_html
from core.what_if_explorer import EDUCATIONAL_SWAP_OPTIONS, run_what_if_experiment


st.set_page_config(
    page_title="LockKey-Explorer | 3D Protein Puzzle Workbench",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject PWA manifest & Science-Fair modern dark UI styling
st.markdown(
    """
    <link rel="manifest" href="/app/static/manifest.json">
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 1.5rem; }
      .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
      }
      .metric-title { font-size: 0.82rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.04em; }
      .metric-value { font-size: 1.75rem; font-weight: 800; color: #f8fafc; margin: 4px 0; }
      .metric-sub { font-size: 0.78rem; color: #38bdf8; }
      .delta-badge-red {
        background: rgba(239, 68, 68, 0.18); border: 1px solid #ef4444; color: #fca5a5;
        padding: 10px 14px; border-radius: 10px; font-weight: 600; margin-top: 8px;
      }
      .delta-badge-green {
        background: rgba(34, 197, 94, 0.18); border: 1px solid #22c55e; color: #86efac;
        padding: 10px 14px; border-radius: 10px; font-weight: 600; margin-top: 8px;
      }
      .delta-badge-neutral {
        background: rgba(56, 189, 248, 0.14); border: 1px solid #38bdf8; color: #bae6fd;
        padding: 10px 14px; border-radius: 10px; font-weight: 600; margin-top: 8px;
      }
      .story-banner {
        background: rgba(139, 92, 246, 0.12); border-left: 4px solid #8b5cf6;
        padding: 10px 16px; border-radius: 0 10px 10px 0; margin-bottom: 12px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_analyze_complex(preset_id: str, min_plddt: float, pocket_id: int) -> LockKeyAnalysisBundle:
    return analyze_classroom_complex(
        preset_or_id=preset_id,
        min_plddt=min_plddt,
        selected_pocket_id=pocket_id,
    )


def build_radar_chart(baseline_sc: PuzzleScorecard, active_sc: PuzzleScorecard, is_swapped: bool) -> go.Figure:
    categories = [
        "3D Shape Lock",
        "Charge Attraction",
        "Hydrophobic Match",
        "Pocket Depth",
        "AlphaFold Confidence",
    ]
    base_vals = [
        baseline_sc.shape_score,
        baseline_sc.charge_score,
        baseline_sc.hydrophobic_score,
        baseline_sc.pocket_depth_score,
        baseline_sc.plddt_confidence,
    ]
    base_vals_closed = base_vals + [base_vals[0]]
    cats_closed = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=base_vals_closed,
            theta=cats_closed,
            fill="toself",
            name="Natural Protein",
            line=dict(color="#8b5cf6", width=2.5),
            fillcolor="rgba(139, 92, 246, 0.28)",
        )
    )

    if is_swapped:
        swap_vals = [
            active_sc.shape_score,
            active_sc.charge_score,
            active_sc.hydrophobic_score,
            active_sc.pocket_depth_score,
            active_sc.plddt_confidence,
        ]
        swap_vals_closed = swap_vals + [swap_vals[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=swap_vals_closed,
                theta=cats_closed,
                fill="toself",
                name="After 'What-If' Swap",
                line=dict(color="#ef4444", width=2.5, dash="dot"),
                fillcolor="rgba(239, 68, 68, 0.20)",
            )
        )

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(15, 23, 42, 0.6)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#334155", tickfont=dict(color="#94a3b8")),
            angularaxis=dict(gridcolor="#334155", tickfont=dict(color="#e2e8f0", size=11)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=25, b=25),
        height=310,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(color="#f8fafc")),
    )
    return fig


def build_touch_map_heatmap(scorecard: PuzzleScorecard) -> go.Figure:
    contacts = scorecard.contacts[:14]
    if not contacts:
        fig = go.Figure()
        fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)")
        return fig

    lock_labels = []
    key_labels = []
    for c in contacts:
        l_lbl = f"Lock {c.lock_resname}{c.lock_resnum}"
        k_lbl = f"Key {c.key_resname}{c.key_resnum}"
        if l_lbl not in lock_labels:
            lock_labels.append(l_lbl)
        if k_lbl not in key_labels:
            key_labels.append(k_lbl)

    z_mat = np.zeros((len(key_labels), len(lock_labels)), dtype=float)
    hover_mat = [
        ["No direct contact (< 4.8 Å)" for _ in range(len(lock_labels))]
        for _ in range(len(key_labels))
    ]

    for c in contacts:
        li = lock_labels.index(f"Lock {c.lock_resname}{c.lock_resnum}")
        ki = key_labels.index(f"Key {c.key_resname}{c.key_resnum}")
        z_mat[ki, li] = c.strength_score
        hover_mat[ki][li] = f"<b>{c.interaction_type}</b><br>{c.tooltip}"

    fig = go.Figure(
        data=go.Heatmap(
            z=z_mat,
            x=lock_labels,
            y=key_labels,
            text=hover_mat,
            hoverinfo="text",
            zmin=-1.0,
            zmax=1.0,
            colorscale=[
                [0.0, "#ef4444"],   # Charge repulsion (Red)
                [0.5, "#1e293b"],   # No contact (Dark Slate)
                [0.75, "#38bdf8"],  # H-bond / Hydrophobic (Sky Blue)
                [1.0, "#facc15"],   # Salt Bridge (Gold)
            ],
            colorbar=dict(
                title=dict(text="Force", font=dict(color="#cbd5e1")),
                tickvals=[-0.9, 0.0, 0.65, 0.95],
                ticktext=["Repel", "None", "Sticky", "Salt Bridge"],
                tickfont=dict(color="#cbd5e1"),
            ),
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.8)",
        margin=dict(l=10, r=10, t=25, b=20),
        height=310,
        xaxis=dict(title="Lock Protein Amino Acids", tickfont=dict(color="#e2e8f0", size=11)),
        yaxis=dict(title="Key Partner Amino Acids", tickfont=dict(color="#e2e8f0", size=11)),
    )
    return fig


def main() -> None:
    if "selected_preset" not in st.session_state:
        st.session_state["selected_preset"] = "3OGO"

    # ---------------- LEFT SIDEBAR ----------------
    with st.sidebar:
        st.markdown("## 🧩 Choose Your Protein Puzzle")
        st.caption("100% Classroom-Safe High School Biology Presets")

        if st.button("🟢 Glowing Jellyfish GFP + Enhancer (PDB: 3OGO)", use_container_width=True):
            st.session_state["selected_preset"] = "3OGO"
        if st.button("🥚 Egg-White Lysozyme + Partner VHH (PDB: 1MEL)", use_container_width=True):
            st.session_state["selected_preset"] = "1MEL"
        if st.button("♻️ Plastic-Eating Enzyme PETase (PDB: 6EQE)", use_container_width=True):
            st.session_state["selected_preset"] = "6EQE"
        if st.button("🩸 Human Hemoglobin Alpha + Beta (PDB: 1A3N)", use_container_width=True):
            st.session_state["selected_preset"] = "1A3N"

        preset_id = st.session_state["selected_preset"]
        st.divider()

        st.markdown("### 🔬 AlphaFold Confidence Filter")
        min_plddt = st.slider(
            "Trim Uncertain Loops (Min pLDDT)",
            min_value=0,
            max_value=90,
            value=65,
            step=5,
            help="Slide from 0 to 90 to trim floppy, low-confidence 'spaghetti tails' and reveal the solid 3D active site!",
        )

        st.markdown("### 🗝️ Keyhole Selector")
        pocket_labels = {
            "Pocket #1 (Main Active Groove)": 1,
            "Pocket #2 (Secondary Surface Groove)": 2,
            "Pocket #3 (Auxiliary Surface Pocket)": 3,
        }
        chosen_pocket_label = st.selectbox("Select Candidate Keyhole on Lock:", list(pocket_labels.keys()))
        pocket_id = pocket_labels[chosen_pocket_label]

        # Run or retrieve cached analysis
        bundle = cached_analyze_complex(preset_id, float(min_plddt), pocket_id)

        st.divider()
        st.markdown("### ⚡ 'What-If' Salt-Bridge Tester")
        feat = bundle.metadata.get("featured_contact", {})
        contact_options = []
        default_label = (
            f"Chain {feat.get('key_chain', 'B')}: {feat.get('key_resname', 'ARG')}{feat.get('key_resnum', 35)} "
            f"(Featured: {feat.get('description', '')[:32]}...)"
        )
        contact_map: dict[str, tuple[str, int]] = {
            default_label: (feat.get("key_chain", "B"), int(feat.get("key_resnum", 35)))
        }
        contact_options.append(default_label)

        for c in bundle.scorecard.contacts[:8]:
            lbl = f"Chain {c.key_chain}: {c.key_resname}{c.key_resnum} ↔ Lock {c.lock_resname}{c.lock_resnum} ({c.interaction_type})"
            if lbl not in contact_map:
                contact_map[lbl] = (c.key_chain, c.key_resnum)
                contact_options.append(lbl)

        selected_contact_lbl = st.selectbox("1. Pick Interface Amino Acid:", contact_options)
        target_chain, target_resnum = contact_map[selected_contact_lbl]

        swap_choice = st.selectbox(
            "2. Swap Amino Acid Property To:",
            list(EDUCATIONAL_SWAP_OPTIONS.keys()),
        )

    # Run What-If Experiment instantaneously
    what_if_result = run_what_if_experiment(
        lock_atoms=bundle.lock_atoms,
        key_atoms=bundle.key_atoms,
        lock_fingerprint=bundle.lock_fingerprint,
        key_fingerprint=bundle.key_fingerprint,
        pocket=bundle.selected_pocket,
        ghost_mold=bundle.ghost_mold,
        chain_id=target_chain,
        residue_num=target_resnum,
        new_amino_acid=swap_choice,
    )
    active_sc = what_if_result.swapped_scorecard
    is_swapped = what_if_result.new_resname != what_if_result.original_resname

    # ---------------- HEADER & STORY BANNER ----------------
    st.markdown(f"## 🧬 `LockKey-Explorer` — {bundle.metadata['title']}")
    st.markdown(
        f"""<div class="story-banner">
        <b>Everyday Biology Story:</b> {bundle.metadata['story']}<br/>
        <span style="color:#cbd5e1; font-size:0.92rem;"><b>What to Look For in 3D:</b> {bundle.metadata['what_students_see']}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # ---------------- CENTER STAGE: ONE-CLICK PAINT MODES & 3D VIEWER ----------------
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2.6, 1.1, 1.1])
    with ctrl_col1:
        paint_mode = st.radio(
            "🎨 One-Click 3D Paint Mode:",
            [
                "Puzzle Shape Mode",
                "Magnet Charge Mode",
                "Oil-vs-Water Mode",
                "AlphaFold Confidence Mode",
            ],
            horizontal=True,
        )
    with ctrl_col2:
        show_ghost = st.toggle("👻 Show 'Ghost Key' Mold", value=True)
    with ctrl_col3:
        show_partner = st.toggle("🗝️ Show Partner 'Key' Protein", value=True)

    viewer_html = build_interactive_3d_html(
        bundle=bundle,
        paint_mode=paint_mode,
        show_ghost_mold=show_ghost,
        show_partner_key=show_partner,
        highlight_residue=(target_chain, target_resnum),
        height=500,
    )
    components.html(viewer_html, height=515, scrolling=False)

    # Display live What-If Experiment Banner
    badge_cls = (
        "delta-badge-red"
        if what_if_result.badge_status == "degraded"
        else "delta-badge-green"
        if what_if_result.badge_status == "improved"
        else "delta-badge-neutral"
    )
    st.markdown(
        f"""<div class="{badge_cls}">
        🔬 <b>What-If Classroom Experiment:</b> {what_if_result.badge_text}<br/>
        <span style="font-weight:400; font-size:0.92rem;">{what_if_result.scientific_explanation}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    st.write("")

    # ---------------- 4 BIG SCIENCE FAIR METRIC BADGES ----------------
    m1, m2, m3, m4 = st.columns(4)
    delta_str = f"({what_if_result.score_delta:+.1f} vs Natural)" if is_swapped else "Optimal Classroom Lock"
    with m1:
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-title">Overall Puzzle Fit (0–100)</div>
            <div class="metric-value">{active_sc.total_score:.1f} / 100</div>
            <div class="metric-sub">{delta_str}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-title">Contact Surface Area</div>
            <div class="metric-value">{active_sc.contact_area_angstrom2:.0f} Å²</div>
            <div class="metric-sub">Buried Keyhole Skin</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-title">Salt Bridges Formed</div>
            <div class="metric-value">{active_sc.salt_bridges_count}</div>
            <div class="metric-sub">+ {active_sc.hydrogen_bonds_count} Polar H-Bonds</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m4:
        clean_lbl = "Clean Fit (No Crash!)" if active_sc.atomic_overlaps == 0 else "Minor Steric Bump"
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-title">Atomic Overlaps (0 = Clean)</div>
            <div class="metric-value">{active_sc.atomic_overlaps}</div>
            <div class="metric-sub">{clean_lbl} | Trimmed {bundle.trim_stats['trimmed_residues_count']} Low-pLDDT Res</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.write("")

    # ---------------- CHARTS: 5-AXIS RADAR & 2D TOUCH-MAP HEATMAP ----------------
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("#### 📊 5-Axis Biophysical Fit Radar")
        radar_fig = build_radar_chart(bundle.scorecard, active_sc, is_swapped)
        st.plotly_chart(radar_fig, use_container_width=True)

    with chart_col2:
        st.markdown("#### 🤝 Interactive 2D Amino Acid Touch-Map")
        heatmap_fig = build_touch_map_heatmap(active_sc)
        st.plotly_chart(heatmap_fig, use_container_width=True)

    # ---------------- SCIENCE FAIR BOOTH EXPORT BUTTONS ----------------
    st.divider()
    exp_col1, exp_col2, exp_col3 = st.columns([1.5, 1.5, 2.0])
    with exp_col1:
        st.download_button(
            label="📥 Export Standalone 3D Model (.html)",
            data=viewer_html,
            file_name=f"LockKey_{bundle.preset_id}_{paint_mode.replace(' ', '_')}.html",
            mime="text/html",
            use_container_width=True,
        )
    with exp_col2:
        st.download_button(
            label="🧬 Export Trimmed Coordinates (.pdb)",
            data=bundle.trimmed_pdb_str,
            file_name=f"LockKey_{bundle.preset_id}_pLDDT{int(min_plddt)}.pdb",
            mime="chemical/x-pdb",
            use_container_width=True,
        )
    with exp_col3:
        st.info(
            "💡 **Science Fair Booth Tip:** Click **Export Standalone 3D Model (.html)** to save an offline interactive 3D hologram you can open on any tablet or school Chromebook!"
        )


if __name__ == "__main__":
    main()
