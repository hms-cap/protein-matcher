"""Model Context Protocol (mcp_server.py) Server for LockKey-Explorer."""

from __future__ import annotations

from typing import Any

try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    from fastmcp import FastMCP  # type: ignore

from core.fetcher import fetch_structure_file, load_biopython_structure, trim_low_confidence_tails
from core.fingerprint import pocket_to_dict
from core.pipeline import analyze_classroom_complex
from core.scorer import scorecard_to_dict
from core.what_if_explorer import run_what_if_experiment, what_if_to_dict

mcp = FastMCP("LockKey-Explorer")


@mcp.tool()
def load_classroom_protein(preset_or_id: str, min_plddt: float = 65.0) -> dict[str, Any]:
    """Loads an approved classroom PDB or AlphaFold model, trims uncertain tails (pLDDT < min_plddt),
    and summarizes the protein's biological function and Lock-and-Key chains.
    """
    pdb_path, meta = fetch_structure_file(preset_or_id)
    structure = load_biopython_structure(pdb_path, structure_id=meta["id"])
    _, trim_stats = trim_low_confidence_tails(structure, min_plddt=min_plddt)

    return {
        "protein_id": meta["id"],
        "title": meta["title"],
        "pdb_path": str(pdb_path),
        "lock_protein": meta["lock_name"],
        "key_partner": meta["key_name"],
        "lock_chain": meta["lock_chain"],
        "key_chain": meta["key_chain"],
        "biology_story": meta["story"],
        "what_students_see": meta["what_students_see"],
        "confidence_trimming": trim_stats,
        "biosecurity_status": "VERIFIED_SAFE_CLASSROOM_PROTEIN",
    }


@mcp.tool()
def find_surface_pockets(pdb_path: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Returns the Top-K surface pockets on the Lock protein with their 3D coordinates,
    valley depth (curvature), electric charge, and greasiness (Kyte-Doolittle hydropathy).
    """
    bundle = analyze_classroom_complex(preset_or_id=pdb_path, min_plddt=65.0, selected_pocket_id=1)
    pockets = bundle.pockets[: max(1, int(top_k))]
    return [pocket_to_dict(p) for p in pockets]


@mcp.tool()
def match_partner_key(lock_pdb: str, pocket_id: int = 1) -> dict[str, Any]:
    """Aligns the partner protein against the pocket's inverted Ghost Key Mold (2.8 Å projection)
    using Kabsch + ICP and returns the Lock-and-Key coupling scorecard and amino acid contact list.
    """
    bundle = analyze_classroom_complex(
        preset_or_id=lock_pdb,
        min_plddt=65.0,
        selected_pocket_id=pocket_id,
    )
    return {
        "protein_id": bundle.preset_id,
        "selected_pocket": pocket_to_dict(bundle.selected_pocket),
        "ghost_mold_points_count": len(bundle.ghost_mold.points),
        "rmsd_to_ghost_mold_angstrom": round(bundle.alignment.rmsd_to_mold, 2),
        "scorecard": scorecard_to_dict(bundle.scorecard),
    }


@mcp.tool()
def simulate_what_if_swap(
    complex_pdb: str,
    chain_id: str,
    residue_num: int,
    new_amino_acid: str,
) -> dict[str, Any]:
    """Simulates a single educational amino acid swap at the Lock-Key interface (e.g. swapping
    Positive ARG to Negative ASP or tiny ALA) and explains why the fit score went up or down.
    """
    bundle = analyze_classroom_complex(preset_or_id=complex_pdb, min_plddt=65.0, selected_pocket_id=1)
    swap_res = run_what_if_experiment(
        lock_atoms=bundle.lock_atoms,
        key_atoms=bundle.key_atoms,
        lock_fingerprint=bundle.lock_fingerprint,
        key_fingerprint=bundle.key_fingerprint,
        pocket=bundle.selected_pocket,
        ghost_mold=bundle.ghost_mold,
        chain_id=chain_id,
        residue_num=int(residue_num),
        new_amino_acid=new_amino_acid,
    )
    return what_if_to_dict(swap_res)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
