"""Demo 2: Egg-White Lysozyme + Camelid VHH Partner (PDB: 1MEL)."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.pipeline import analyze_classroom_complex
from core.viewer_3d import build_interactive_3d_html
from core.what_if_explorer import run_what_if_experiment


def run_demo() -> Path:
    print("=" * 72)
    print("DEMO 2: Egg-White Defense — Lysozyme + Camelid VHH Partner (PDB: 1MEL)")
    print("=" * 72)
    bundle = analyze_classroom_complex("1MEL", min_plddt=65.0, selected_pocket_id=1)
    sc = bundle.scorecard

    print(f"Lock Protein : {bundle.metadata['lock_name']}")
    print(f"Key Partner  : {bundle.metadata['key_name']}")
    print(f"Active Pocket: {bundle.selected_pocket.name} (Depth Curvature: {bundle.selected_pocket.mean_curvature:.3f})")
    print(f"Overall Fit  : {sc.total_score:.1f} / 100")
    print(f"  - 3D Shape Lock       : {sc.shape_score:.1f}%")
    print(f"  - Charge Attraction   : {sc.charge_score:.1f}%")
    print(f"  - Hydrophobic Match   : {sc.hydrophobic_score:.1f}%")

    swap = run_what_if_experiment(
        bundle.lock_atoms,
        bundle.key_atoms,
        bundle.lock_fingerprint,
        bundle.key_fingerprint,
        bundle.selected_pocket,
        bundle.ghost_mold,
        chain_id="M",
        residue_num=103,
        new_amino_acid="ALA",
    )
    print("\n[What-If Classroom Experiment: Key Loop Tyr103 -> Tiny Alanine (ALA)]")
    print(f"  -> {swap.badge_text}")

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / "demo_2_lysozyme_1MEL.html"
    out_html.write_text(
        build_interactive_3d_html(bundle, paint_mode="Puzzle Shape Mode", highlight_residue=("M", 103)),
        encoding="utf-8",
    )
    print(f"\nSaved standalone 3D HTML viewer to: {out_html}")
    return out_html


if __name__ == "__main__":
    run_demo()
