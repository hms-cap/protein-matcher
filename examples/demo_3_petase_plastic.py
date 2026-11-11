"""Demo 3: AlphaFold Plastic-Recycling PETase Pocket Explorer (PDB: 6EQE / AF-A0A0K8P6T7)."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.pipeline import analyze_classroom_complex
from core.viewer_3d import build_interactive_3d_html


def run_demo() -> Path:
    print("=" * 72)
    print("DEMO 3: Plastic-Recycling Enzyme PETase Pocket Explorer (PDB: 6EQE)")
    print("=" * 72)
    bundle = analyze_classroom_complex("6EQE", min_plddt=65.0, selected_pocket_id=1)
    sc = bundle.scorecard

    print(f"Enzyme       : {bundle.metadata['lock_name']}")
    print(f"Trimmed Res  : {bundle.trim_stats['trimmed_residues_count']} low-confidence residues removed")
    print(f"Active Cleft : {bundle.selected_pocket.name} (Lining: {', '.join(bundle.selected_pocket.key_residues[:6])})")
    print(f"Overall Fit  : {sc.total_score:.1f} / 100 | Hydrophobic Match: {sc.hydrophobic_score:.1f}%")

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / "demo_3_petase_6EQE.html"
    out_html.write_text(
        build_interactive_3d_html(bundle, paint_mode="Oil-vs-Water Mode", highlight_residue=("A", 185)),
        encoding="utf-8",
    )
    print(f"\nSaved standalone 3D HTML viewer to: {out_html}")
    return out_html


if __name__ == "__main__":
    run_demo()
