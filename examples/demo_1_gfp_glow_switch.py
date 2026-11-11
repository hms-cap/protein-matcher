"""Demo 1: Jellyfish GFP + Enhancer Nanobody (PDB: 3OGO)."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.pipeline import analyze_classroom_complex
from core.viewer_3d import build_interactive_3d_html
from core.what_if_explorer import run_what_if_experiment


def run_demo() -> Path:
    print("=" * 72)
    print("DEMO 1: The Jellyfish Glow Switch — GFP + Enhancer (PDB: 3OGO)")
    print("=" * 72)
    bundle = analyze_classroom_complex("3OGO", min_plddt=65.0, selected_pocket_id=1)
    sc = bundle.scorecard

    print(f"Lock Protein : {bundle.metadata['lock_name']}")
    print(f"Key Partner  : {bundle.metadata['key_name']}")
    print(f"Active Pocket: {bundle.selected_pocket.name} (Area: {bundle.selected_pocket.surface_area:.1f} Å²)")
    print(f"Overall Fit  : {sc.total_score:.1f} / 100")
    print(f"  - 3D Shape Lock       : {sc.shape_score:.1f}%")
    print(f"  - Charge Attraction   : {sc.charge_score:.1f}% ({sc.salt_bridges_count} Salt Bridges)")
    print(f"  - Hydrophobic Match   : {sc.hydrophobic_score:.1f}%")
    print(f"  - Buried Contact Area : {sc.contact_area_angstrom2:.1f} Å²")

    # Test the What-If Salt-Bridge Swap on Key Arg35 -> Asp35 (-)
    swap = run_what_if_experiment(
        bundle.lock_atoms,
        bundle.key_atoms,
        bundle.lock_fingerprint,
        bundle.key_fingerprint,
        bundle.selected_pocket,
        bundle.ghost_mold,
        chain_id="B",
        residue_num=35,
        new_amino_acid="ASP",
    )
    print("\n[What-If Classroom Experiment: Key Arg35 (+) -> Asp35 (-)]")
    print(f"  -> {swap.badge_text}")
    print(f"  -> {swap.scientific_explanation}")

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / "demo_1_gfp_3OGO.html"
    out_html.write_text(
        build_interactive_3d_html(bundle, paint_mode="Magnet Charge Mode", highlight_residue=("B", 35)),
        encoding="utf-8",
    )
    print(f"\nSaved standalone 3D HTML viewer to: {out_html}")
    return out_html


if __name__ == "__main__":
    run_demo()
