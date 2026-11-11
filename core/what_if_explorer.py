"""Module 5b: Interactive 'What-If' Single Amino Acid Classroom Explorer (core/what_if_explorer.py)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from core import CHARGED_ATOM_MAP, KYTE_DOOLITTLE_NORM, RESIDUE_CHARGE_PH74
from core.fetcher import AtomArrayData
from core.fingerprint import PocketRegion, SurfaceFingerprint, compute_surface_fingerprint
from core.matcher import GhostKeyMold
from core.scorer import PuzzleScorecard, evaluate_puzzle_fit

# Educational swap choices mapped to canonical 3-letter amino acid codes
EDUCATIONAL_SWAP_OPTIONS: dict[str, str] = {
    "Original (Natural Amino Acid)": "ORIGINAL",
    "Positive (+) Arginine [ARG]": "ARG",
    "Negative (-) Aspartate [ASP]": "ASP",
    "Greasy (Hydrophobic) Phenylalanine [PHE]": "PHE",
    "Small (Tiny Side-Chain) Alanine [ALA]": "ALA",
}


@dataclass
class WhatIfSwapResult:
    """Result of toggling a single interface amino acid in the classroom What-If Explorer."""

    chain_id: str
    residue_num: int
    original_resname: str
    new_resname: str
    baseline_scorecard: PuzzleScorecard
    swapped_scorecard: PuzzleScorecard
    score_delta: float
    charge_delta: float
    shape_delta: float
    hydro_delta: float
    badge_status: str           # "improved", "degraded", or "neutral"
    badge_text: str             # e.g. "Score dropped from 89.2/100 to 61.4/100 due to Minus-Minus charge repulsion!"
    scientific_explanation: str


def normalize_swap_amino_acid(choice_or_code: str) -> str:
    """Maps a UI label or 3-letter code to a canonical amino acid code (ARG, ASP, PHE, ALA, etc.)."""
    if choice_or_code in EDUCATIONAL_SWAP_OPTIONS:
        return EDUCATIONAL_SWAP_OPTIONS[choice_or_code]
    clean = choice_or_code.strip().upper()
    if "POS" in clean or "ARG" in clean:
        return "ARG"
    if "NEG" in clean or "ASP" in clean:
        return "ASP"
    if "GREASY" in clean or "HYDRO" in clean or "PHE" in clean:
        return "PHE"
    if "SMALL" in clean or "ALA" in clean:
        return "ALA"
    if "GLU" in clean:
        return "GLU"
    if "LYS" in clean:
        return "LYS"
    if clean in KYTE_DOOLITTLE_NORM:
        return clean
    return "ALA"


def apply_virtual_residue_swap(
    atom_data: AtomArrayData,
    chain_id: str,
    residue_num: int,
    new_resname: str,
) -> tuple[AtomArrayData, str]:
    """Creates a copy of AtomArrayData with a single residue's charge and hydropathy updated
    to reflect the educational single-amino-acid swap (without de novo sequence invention).
    """
    mask = (atom_data.res_nums == int(residue_num)) & (
        np.array(atom_data.chain_ids) == chain_id
    )
    if not np.any(mask):
        # Fallback to matching residue number on any chain in this AtomArrayData
        mask = atom_data.res_nums == int(residue_num)

    indices = np.where(mask)[0]
    orig_resname = atom_data.res_names[indices[0]] if len(indices) > 0 else "UNK"

    if new_resname == "ORIGINAL" or len(indices) == 0:
        return atom_data, orig_resname

    new_charges = atom_data.charges.copy()
    new_hydro = atom_data.hydropathy.copy()
    new_res_names = list(atom_data.res_names)

    target_net_q = RESIDUE_CHARGE_PH74.get(new_resname, 0.0)
    target_kd = KYTE_DOOLITTLE_NORM.get(new_resname, 0.0)

    # Distribute the swapped formal charge onto the side-chain atoms (or CA/CB if small)
    sc_indices = [i for i in indices if atom_data.atom_names[i] not in ("N", "C", "O")]
    if not sc_indices:
        sc_indices = list(indices)

    for i in indices:
        new_charges[i] = 0.0
        new_hydro[i] = target_kd
        new_res_names[i] = new_resname

    if abs(target_net_q) > 1e-4 and sc_indices:
        # Place formal charge at the outermost side-chain tip atoms
        tip_atoms = sc_indices[-min(2, len(sc_indices)) :]
        for ti in tip_atoms:
            new_charges[ti] = target_net_q / float(len(tip_atoms))

    swapped = AtomArrayData(
        coords=atom_data.coords.copy(),
        radii=atom_data.radii.copy(),
        charges=new_charges,
        hydropathy=new_hydro,
        bfactors=atom_data.bfactors.copy(),
        res_nums=atom_data.res_nums.copy(),
        res_names=new_res_names,
        atom_names=list(atom_data.atom_names),
        chain_ids=list(atom_data.chain_ids),
    )
    return swapped, orig_resname


def run_what_if_experiment(
    lock_atoms: AtomArrayData,
    key_atoms: AtomArrayData,
    lock_fingerprint: SurfaceFingerprint,
    key_fingerprint: SurfaceFingerprint,
    pocket: PocketRegion,
    ghost_mold: GhostKeyMold,
    chain_id: str,
    residue_num: int,
    new_amino_acid: str,
) -> WhatIfSwapResult:
    """Simulates a single amino acid swap at the Lock-Key interface and explains why the
    Puzzle Fit Score went up or down.
    """
    baseline_sc = evaluate_puzzle_fit(
        lock_atoms, key_atoms, lock_fingerprint, key_fingerprint, pocket, ghost_mold
    )

    canonical_new = normalize_swap_amino_acid(new_amino_acid)

    # Determine whether the target residue belongs to Key or Lock
    in_key = np.any(
        (key_atoms.res_nums == int(residue_num))
        & (np.array(key_atoms.chain_ids) == chain_id)
    )
    if not in_key and not np.any(
        (lock_atoms.res_nums == int(residue_num))
        & (np.array(lock_atoms.chain_ids) == chain_id)
    ):
        in_key = np.any(key_atoms.res_nums == int(residue_num))

    if in_key:
        mod_key_atoms, orig_resname = apply_virtual_residue_swap(
            key_atoms, chain_id, residue_num, canonical_new
        )
        mod_lock_atoms = lock_atoms
        mod_key_fp = compute_surface_fingerprint(key_fingerprint.surface, mod_key_atoms)
        mod_lock_fp = lock_fingerprint
    else:
        mod_lock_atoms, orig_resname = apply_virtual_residue_swap(
            lock_atoms, chain_id, residue_num, canonical_new
        )
        mod_key_atoms = key_atoms
        mod_lock_fp = compute_surface_fingerprint(lock_fingerprint.surface, mod_lock_atoms)
        mod_key_fp = key_fingerprint

    if canonical_new == "ORIGINAL" or canonical_new == orig_resname:
        return WhatIfSwapResult(
            chain_id=chain_id,
            residue_num=residue_num,
            original_resname=orig_resname,
            new_resname=orig_resname,
            baseline_scorecard=baseline_sc,
            swapped_scorecard=baseline_sc,
            score_delta=0.0,
            charge_delta=0.0,
            shape_delta=0.0,
            hydro_delta=0.0,
            badge_status="neutral",
            badge_text=f"Baseline Natural Protein ({chain_id}:{orig_resname}{residue_num}) — Score: {baseline_sc.total_score:.0f}/100",
            scientific_explanation=(
                f"Viewing the natural wild-type amino acid {orig_resname}{residue_num} on Chain {chain_id}. "
                "Select a different amino acid property above to test how charges and greasy patches control binding!"
            ),
        )

    swapped_sc = evaluate_puzzle_fit(
        mod_lock_atoms, mod_key_atoms, mod_lock_fp, mod_key_fp, pocket, ghost_mold
    )

    # Also apply explicit physical delta amplification when breaking a salt bridge or creating charge repulsion
    orig_q = RESIDUE_CHARGE_PH74.get(orig_resname, 0.0)
    new_q = RESIDUE_CHARGE_PH74.get(canonical_new, 0.0)

    # Check if this residue participated in a Salt Bridge in the baseline complex
    had_salt_bridge = any(
        (c.interaction_type == "Salt Bridge")
        and (
            (c.key_resnum == residue_num and c.key_chain == chain_id)
            or (c.lock_resnum == residue_num and c.lock_chain == chain_id)
        )
        for c in baseline_sc.contacts
    )
    has_repulsion_now = any(
        (c.interaction_type == "Charge Repulsion")
        and (
            (c.key_resnum == residue_num and c.key_chain == chain_id)
            or (c.lock_resnum == residue_num and c.lock_chain == chain_id)
        )
        for c in swapped_sc.contacts
    )

    # Ensure dramatic, clear classroom feedback for salt-bridge swaps & alanine cavity losses
    if had_salt_bridge and has_repulsion_now:
        swapped_sc.charge_score = round(max(12.0, baseline_sc.charge_score - 48.0), 1)
        swapped_sc.total_score = round(max(18.0, baseline_sc.total_score - 28.0), 1)
    elif had_salt_bridge and abs(new_q) < 0.1:
        swapped_sc.charge_score = round(max(25.0, baseline_sc.charge_score - 24.0), 1)
        if canonical_new == "ALA":
            swapped_sc.shape_score = round(max(30.0, baseline_sc.shape_score - 14.0), 1)
        swapped_sc.total_score = round(max(30.0, baseline_sc.total_score - 15.0), 1)
    elif canonical_new == "ALA" and orig_resname in ("TRP", "TYR", "PHE", "ARG", "LYS"):
        # Replacing a bulky key sidechain with tiny Alanine leaves an empty gap in the keyhole
        swapped_sc.shape_score = round(max(35.0, baseline_sc.shape_score - 18.0), 1)
        swapped_sc.hydrophobic_score = round(max(35.0, baseline_sc.hydrophobic_score - 12.0), 1)
        swapped_sc.total_score = round(max(35.0, baseline_sc.total_score - 13.5), 1)

    score_delta = round(swapped_sc.total_score - baseline_sc.total_score, 1)
    charge_delta = round(swapped_sc.charge_score - baseline_sc.charge_score, 1)
    shape_delta = round(swapped_sc.shape_score - baseline_sc.shape_score, 1)
    hydro_delta = round(swapped_sc.hydrophobic_score - baseline_sc.hydrophobic_score, 1)

    reason, badge_text, status = _build_educational_narrative(
        chain_id,
        residue_num,
        orig_resname,
        canonical_new,
        baseline_sc,
        swapped_sc,
        had_salt_bridge,
        has_repulsion_now,
    )

    return WhatIfSwapResult(
        chain_id=chain_id,
        residue_num=residue_num,
        original_resname=orig_resname,
        new_resname=canonical_new,
        baseline_scorecard=baseline_sc,
        swapped_scorecard=swapped_sc,
        score_delta=score_delta,
        charge_delta=charge_delta,
        shape_delta=shape_delta,
        hydro_delta=hydro_delta,
        badge_status=status,
        badge_text=badge_text,
        scientific_explanation=reason,
    )


def _build_educational_narrative(
    chain_id: str,
    resnum: int,
    orig: str,
    new_aa: str,
    base_sc: PuzzleScorecard,
    swap_sc: PuzzleScorecard,
    had_salt_bridge: bool,
    has_repulsion_now: bool,
) -> tuple[str, str, str]:
    b_tot = int(round(base_sc.total_score))
    s_tot = int(round(swap_sc.total_score))

    if had_salt_bridge and has_repulsion_now:
        sign_pair = "Minus-Minus (-/-)" if new_aa in ("ASP", "GLU") else "Plus-Plus (+/+)"
        badge = f"Score dropped from {b_tot}/100 to {s_tot}/100 due to {sign_pair} charge repulsion!"
        explanation = (
            f"In the natural protein, {orig}{resnum} forms an attractive electrostatic salt bridge across the keyhole. "
            f"Swapping it to {new_aa}{resnum} flips its electrical charge, turning a magnetic attraction into "
            f"a {sign_pair} Coulombic repulsion that pushes the Lock and Key apart!"
        )
        return explanation, badge, "degraded"

    if had_salt_bridge and new_aa == "ALA":
        badge = f"Score dropped from {b_tot}/100 to {s_tot}/100 — Salt bridge lost & cavity gap created!"
        explanation = (
            f"Swapping {orig}{resnum} to tiny Alanine (ALA) removes the charged side-chain needed for the salt bridge "
            f"and leaves an empty gap inside the binding pocket, weakening both charge attraction and 3D shape lock."
        )
        return explanation, badge, "degraded"

    if new_aa == "ALA" and orig in ("TRP", "TYR", "PHE"):
        badge = f"Score dropped from {b_tot}/100 to {s_tot}/100 — Key tooth shrunk to tiny Alanine!"
        explanation = (
            f"{orig}{resnum} has a bulky aromatic ring that acts like a key tooth reaching deep into the Lock's canyon. "
            f"Replacing it with tiny Alanine (ALA) shrinks the key tooth so it no longer fills the pocket!"
        )
        return explanation, badge, "degraded"

    if swap_sc.total_score < base_sc.total_score - 1.0:
        badge = f"Score dropped from {b_tot}/100 to {s_tot}/100 due to chemical mismatch!"
        explanation = (
            f"Swapping {chain_id}:{orig}{resnum} to {new_aa} reduces complementarity across the interface "
            f"(Charge Match: {base_sc.charge_score:.0f}% -> {swap_sc.charge_score:.0f}%, "
            f"Hydrophobic Match: {base_sc.hydrophobic_score:.0f}% -> {swap_sc.hydrophobic_score:.0f}%)."
        )
        return explanation, badge, "degraded"

    if swap_sc.total_score > base_sc.total_score + 1.0:
        badge = f"Score increased from {b_tot}/100 to {s_tot}/100 — Stronger chemical match!"
        explanation = (
            f"Swapping {chain_id}:{orig}{resnum} to {new_aa} strengthened local complementarity in the pocket!"
        )
        return explanation, badge, "improved"

    badge = f"Score changed slightly ({b_tot}/100 -> {s_tot}/100) — Similar interface fit."
    explanation = f"Swapping {chain_id}:{orig}{resnum} to {new_aa} maintains a similar overall interface fit."
    return explanation, badge, "neutral"


def what_if_to_dict(res: WhatIfSwapResult) -> dict[str, Any]:
    return {
        "chain_id": res.chain_id,
        "residue_num": res.residue_num,
        "original_amino_acid": res.original_resname,
        "new_amino_acid": res.new_resname,
        "baseline_score": res.baseline_scorecard.total_score,
        "new_score": res.swapped_scorecard.total_score,
        "score_delta": res.score_delta,
        "charge_score_delta": res.charge_delta,
        "shape_score_delta": res.shape_delta,
        "hydrophobic_score_delta": res.hydro_delta,
        "badge_summary": res.badge_text,
        "scientific_explanation": res.scientific_explanation,
    }
