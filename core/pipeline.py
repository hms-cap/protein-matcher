"""High-level orchestration pipeline for LockKey-Explorer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

from core.fetcher import (
    AtomArrayData,
    extract_atom_data,
    fetch_structure_file,
    load_biopython_structure,
    structure_to_pdb_string,
    trim_low_confidence_tails,
)
from core.fingerprint import (
    PocketRegion,
    SurfaceFingerprint,
    compute_surface_fingerprint,
    discover_top_pockets,
)
from core.matcher import (
    AlignmentResult,
    GhostKeyMold,
    align_partner_to_ghost_mold,
    create_ideal_key_ghost_mold,
)
from core.scorer import PuzzleScorecard, evaluate_puzzle_fit
from core.surface import compute_molecular_surface


@dataclass
class LockKeyAnalysisBundle:
    """Complete precomputed bundle for a classroom protein Lock-and-Key puzzle."""

    preset_id: str
    metadata: dict[str, Any]
    pdb_path: Path
    trimmed_pdb_str: str
    trim_stats: dict[str, Any]
    lock_atoms: AtomArrayData
    key_atoms: AtomArrayData
    lock_fingerprint: SurfaceFingerprint
    key_fingerprint: SurfaceFingerprint
    pockets: list[PocketRegion]
    selected_pocket: PocketRegion
    ghost_mold: GhostKeyMold
    alignment: AlignmentResult
    scorecard: PuzzleScorecard


def analyze_classroom_complex(
    preset_or_id: str = "3OGO",
    min_plddt: float = 65.0,
    selected_pocket_id: int = 1,
) -> LockKeyAnalysisBundle:
    """Runs the complete biophysical pipeline on a safe classroom preset in <1.5 seconds on CPU."""
    pdb_path, meta = fetch_structure_file(preset_or_id)
    structure = load_biopython_structure(pdb_path, structure_id=meta["id"])
    structure, trim_stats = trim_low_confidence_tails(structure, min_plddt=min_plddt)

    lock_chain = meta["lock_chain"]
    key_chain = meta["key_chain"]

    if lock_chain != key_chain:
        lock_atoms = extract_atom_data(structure, chain_id=lock_chain)
        key_atoms = extract_atom_data(structure, chain_id=key_chain)
        trimmed_pdb_str = structure_to_pdb_string(structure, chains=[lock_chain, key_chain])
    else:
        # Single-chain enzyme (e.g., 6EQE / AF-A0A0K8P6T7 PETase):
        # Split into the catalytic core ("The Lock", residues <= 175 or >= 210)
        # and the substrate-clamping Trp185/Ile208 mobile lid domain ("The Key Partner Patch", residues 176..209)
        lock_atoms = _extract_petase_split(structure, lock_chain, is_lock=True)
        key_atoms = _extract_petase_split(structure, lock_chain, is_lock=False)
        trimmed_pdb_str = structure_to_pdb_string(structure, chains=[lock_chain])

    lock_surf = compute_molecular_surface(lock_atoms, subsample_max_points=1400)
    key_surf = compute_molecular_surface(key_atoms, subsample_max_points=1200)

    lock_fp = compute_surface_fingerprint(lock_surf, lock_atoms)
    key_fp = compute_surface_fingerprint(key_surf, key_atoms)

    # Compute the true biological contact center between Lock and Key so Pocket #1 is anchored to the main groove
    pref_center = None
    if len(lock_atoms.coords) > 0 and len(key_atoms.coords) > 0:
        feat = meta.get("featured_contact", {})
        target_res = feat.get("lock_resnum")
        if target_res is not None:
            m = lock_atoms.res_nums == int(target_res)
            if np.any(m):
                pref_center = np.mean(lock_atoms.coords[m], axis=0)
        if pref_center is None:
            pref_center = np.mean(key_atoms.coords, axis=0)

    pockets = discover_top_pockets(
        lock_fp,
        lock_atoms,
        top_k=3,
        preferred_contact_center=pref_center,
    )
    if not pockets:
        # Fallback single pocket covering the center of the Lock surface
        pockets = [
            PocketRegion(
                pocket_id=1,
                name="Pocket #1 (Main Active Groove)",
                center=np.mean(lock_surf.points, axis=0) if len(lock_surf.points) else np.zeros(3),
                point_indices=np.arange(min(80, len(lock_surf.points))),
                mean_curvature=-0.25,
                mean_charge=0.0,
                mean_hydropathy=0.2,
                surface_area=420.0,
                key_residues=["SER160", "TRP185", "HIS237"],
            )
        ]

    idx_choice = max(0, min(len(pockets) - 1, int(selected_pocket_id) - 1))
    chosen_pocket = pockets[idx_choice]

    ghost_mold = create_ideal_key_ghost_mold(lock_fp, chosen_pocket, projection_distance=2.8)
    alignment = align_partner_to_ghost_mold(
        ghost_mold=ghost_mold,
        lock_atoms=lock_atoms,
        key_atoms=key_atoms,
        key_fingerprint=key_fp,
        n_icp_steps=15,
        refine_from_current_pose=(idx_choice == 0),
    )

    scorecard = evaluate_puzzle_fit(
        lock_atoms=lock_atoms,
        key_atoms=alignment.aligned_key_atoms,
        lock_fingerprint=lock_fp,
        key_fingerprint=key_fp,
        pocket=chosen_pocket,
        ghost_mold=ghost_mold,
        aligned_key_surface_pts=alignment.aligned_key_surface_pts,
    )

    return LockKeyAnalysisBundle(
        preset_id=meta["id"],
        metadata=meta,
        pdb_path=pdb_path,
        trimmed_pdb_str=trimmed_pdb_str,
        trim_stats=trim_stats,
        lock_atoms=lock_atoms,
        key_atoms=alignment.aligned_key_atoms,
        lock_fingerprint=lock_fp,
        key_fingerprint=key_fp,
        pockets=pockets,
        selected_pocket=chosen_pocket,
        ghost_mold=ghost_mold,
        alignment=alignment,
        scorecard=scorecard,
    )


def _extract_petase_split(structure, chain_id: str, is_lock: bool) -> AtomArrayData:
    """Splits a single-chain enzyme (6EQE / AF-A0A0K8P6T7) into its main catalytic scaffold ('The Lock')
    and its substrate-binding Trp185 clamp loop ('The Key') so all Lock-and-Key tools work seamlessly.
    """
    full = extract_atom_data(structure, chain_id=chain_id)
    if len(full.coords) == 0:
        return full
    key_loop_mask = (full.res_nums >= 178) & (full.res_nums <= 212)
    mask = ~key_loop_mask if is_lock else key_loop_mask
    if not np.any(mask):
        mask = np.ones(len(full.coords), dtype=bool)
    idx = np.where(mask)[0]
    return AtomArrayData(
        coords=full.coords[idx],
        radii=full.radii[idx],
        charges=full.charges[idx],
        hydropathy=full.hydropathy[idx],
        bfactors=full.bfactors[idx],
        res_nums=full.res_nums[idx],
        res_names=[full.res_names[i] for i in idx],
        atom_names=[full.atom_names[i] for i in idx],
        chain_ids=[full.chain_ids[i] for i in idx],
    )
