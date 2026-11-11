"""Module 5a: Puzzle Fit Scorer & Contact Analyzer (core/scorer.py)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from core import RESIDUE_CHARGE_PH74, KYTE_DOOLITTLE_NORM
from core.fetcher import AtomArrayData
from core.fingerprint import PocketRegion, SurfaceFingerprint
from core.matcher import GhostKeyMold, count_backbone_clashes
from core.spatial_ops import FastKDTree as cKDTree


@dataclass
class ResidueContact:
    """Represents an atomic/residue-level interaction across the Lock-Key interface."""

    lock_chain: str
    lock_resname: str
    lock_resnum: int
    key_chain: str
    key_resname: str
    key_resnum: int
    min_distance: float
    interaction_type: str       # "Salt Bridge", "Hydrogen Bond / Polar", "Greasy Hydrophobic", "Shape Contact", "Charge Repulsion"
    strength_score: float       # positive = favorable, negative = repulsive
    tooltip: str


@dataclass
class PuzzleScorecard:
    """Complete Science Fair Scorecard for a Lock-and-Key coupling."""

    total_score: float          # S_total in [0, 100]
    shape_score: float          # S_shape in [0, 100]
    charge_score: float         # S_charge in [0, 100]
    hydrophobic_score: float    # S_hydrophobic in [0, 100]
    pocket_depth_score: float   # 0-100 for 5-axis radar chart
    plddt_confidence: float     # 0-100 for 5-axis radar chart
    overlap_penalty: float      # P_overlap
    atomic_overlaps: int        # Count of steric clashes
    contact_area_angstrom2: float
    salt_bridges_count: int
    hydrogen_bonds_count: int
    contacts: list[ResidueContact]


def evaluate_puzzle_fit(
    lock_atoms: AtomArrayData,
    key_atoms: AtomArrayData,
    lock_fingerprint: SurfaceFingerprint,
    key_fingerprint: SurfaceFingerprint,
    pocket: PocketRegion,
    ghost_mold: GhostKeyMold,
    aligned_key_surface_pts: np.ndarray | None = None,
) -> PuzzleScorecard:
    """Computes Total Puzzle Fit Score (S_total in [0, 100]):
        S_total = 0.40 * S_shape + 0.30 * S_charge + 0.30 * S_hydrophobic - P_overlap
    Along with Contact Surface Area (Å^2), Salt Bridges, and 2D Touch-Map contacts.
    """
    key_pts = (
        aligned_key_surface_pts
        if aligned_key_surface_pts is not None
        else key_fingerprint.surface.points
    )

    contacts = find_residue_contacts(lock_atoms, key_atoms)
    salt_bridges = [c for c in contacts if c.interaction_type == "Salt Bridge"]
    hbonds = [c for c in contacts if c.interaction_type == "Hydrogen Bond / Polar"]
    repulsions = [c for c in contacts if c.interaction_type == "Charge Repulsion"]

    # Find surface points in the Lock pocket that are matched by Key surface points
    if len(ghost_mold.points) > 0 and len(key_pts) > 0:
        key_surf_tree = cKDTree(key_pts)
        dists, key_nn_idx = key_surf_tree.query(ghost_mold.points)
        contact_mask = dists < 5.5
        if np.sum(contact_mask) < 6:
            # Take closest 25% of mold points if pocket is partially matched
            cutoff = float(np.percentile(dists, 35))
            contact_mask = dists <= max(cutoff, 6.0)
    else:
        contact_mask = np.zeros(0, dtype=bool)
        key_nn_idx = np.zeros(0, dtype=np.int32)
        dists = np.zeros(0, dtype=np.float64)

    if np.any(contact_mask):
        matched_key_idx = key_nn_idx[contact_mask]
        matched_dists = dists[contact_mask]

        # 1. 3D Shape Match (S_shape): proximity to Ghost Mold + curvature complementarity
        k_ideal = ghost_mold.ideal_curvature[contact_mask]
        k_actual = key_fingerprint.surface.curvature[matched_key_idx]
        prox_factor = np.exp(-0.08 * (matched_dists**2))
        curv_diff = np.abs(k_ideal - k_actual)
        shape_raw = float(np.mean(prox_factor * (1.0 - 0.35 * curv_diff)))
        s_shape = float(np.clip(45.0 + 55.0 * shape_raw, 15.0, 99.0))

        # 2. Plus/Minus Charge Match (S_charge): surface potential complementarity + explicit salt bridges
        q_ideal = ghost_mold.ideal_charge[contact_mask]
        q_actual = key_fingerprint.charge[matched_key_idx]
        elec_corr = float(np.mean(1.0 - 0.5 * np.abs(q_ideal - q_actual)))
        sb_bonus = min(18.0, 7.5 * len(salt_bridges))
        # Only penalize non-native charge repulsions (or cap gentle adjacent crystal sidechain proximity)
        rep_penalty = 6.0 * max(0, len(repulsions) - 2)
        s_charge = float(np.clip(48.0 + 42.0 * elec_corr + sb_bonus - rep_penalty, 15.0, 99.0))

        # 3. Hydrophobic Stickiness Match (S_hydrophobic): matching greasy-to-greasy & polar-to-polar
        h_ideal = ghost_mold.ideal_hydropathy[contact_mask]
        h_actual = key_fingerprint.hydropathy[matched_key_idx]
        hydro_match = float(np.mean(1.0 - 0.45 * np.abs(h_ideal - h_actual)))
        greasy_contacts = sum(1 for c in contacts if c.interaction_type == "Greasy Hydrophobic")
        hydro_bonus = min(14.0, 2.5 * greasy_contacts)
        s_hydro = float(np.clip(42.0 + 46.0 * hydro_match + hydro_bonus, 10.0, 99.0))

        # Buried contact surface area (both sides of the interface)
        lock_pocket_areas = lock_fingerprint.surface.area_per_point[pocket.point_indices]
        contact_area = float(np.sum(lock_pocket_areas[contact_mask]) * 1.85)
    else:
        s_shape, s_charge, s_hydro, contact_area = 30.0, 30.0, 30.0, 120.0

    # Ensure contact area reflects residue contacts when full interface is larger than a single sub-pocket
    if contacts:
        contact_area = max(contact_area, float(len(contacts) * 28.5))

    clashes = count_backbone_clashes(lock_atoms, key_atoms)
    p_overlap = float(min(40.0, clashes * 4.5))

    s_total = float(
        np.clip(
            0.40 * s_shape + 0.30 * s_charge + 0.30 * s_hydro - p_overlap,
            0.0,
            100.0,
        )
    )

    pocket_depth_score = float(np.clip(50.0 + abs(pocket.mean_curvature) * 110.0, 25.0, 98.0))
    all_plddt = np.concatenate([lock_atoms.bfactors, key_atoms.bfactors])
    mean_plddt = float(np.mean(all_plddt)) if len(all_plddt) > 0 else 90.0

    return PuzzleScorecard(
        total_score=round(s_total, 1),
        shape_score=round(s_shape, 1),
        charge_score=round(s_charge, 1),
        hydrophobic_score=round(s_hydro, 1),
        pocket_depth_score=round(pocket_depth_score, 1),
        plddt_confidence=round(mean_plddt, 1),
        overlap_penalty=round(p_overlap, 1),
        atomic_overlaps=clashes,
        contact_area_angstrom2=round(contact_area, 1),
        salt_bridges_count=len(salt_bridges),
        hydrogen_bonds_count=len(hbonds),
        contacts=contacts,
    )


def find_residue_contacts(
    lock_atoms: AtomArrayData,
    key_atoms: AtomArrayData,
    contact_cutoff: float = 4.8,
) -> list[ResidueContact]:
    """Finds all residue pairs across the Lock and Key within contact_cutoff Å and classifies
    their physical interaction type with plain-English student tooltips.
    """
    if len(lock_atoms.coords) == 0 or len(key_atoms.coords) == 0:
        return []

    same_chain = (
        len(set(lock_atoms.chain_ids)) == 1
        and lock_atoms.chain_ids[0] == key_atoms.chain_ids[0]
    )

    lock_tree = cKDTree(lock_atoms.coords)
    pairs = lock_tree.query_ball_point(key_atoms.coords, r=contact_cutoff)

    best_pair_dist: dict[tuple[int, int], float] = {}
    lock_meta: dict[int, tuple[str, str]] = {}
    key_meta: dict[int, tuple[str, str]] = {}

    for k_idx, l_indices in enumerate(pairs):
        if not l_indices:
            continue
        k_res = int(key_atoms.res_nums[k_idx])
        k_name = key_atoms.res_names[k_idx]
        k_chain = key_atoms.chain_ids[k_idx]
        key_meta[k_res] = (k_chain, k_name)

        k_coord = key_atoms.coords[k_idx]
        for l_idx in l_indices:
            l_res = int(lock_atoms.res_nums[l_idx])
            if same_chain and abs(k_res - l_res) <= 4:
                continue
            l_name = lock_atoms.res_names[l_idx]
            l_chain = lock_atoms.chain_ids[l_idx]
            lock_meta[l_res] = (l_chain, l_name)

            d = float(np.linalg.norm(k_coord - lock_atoms.coords[l_idx]))
            pair_key = (l_res, k_res)
            if pair_key not in best_pair_dist or d < best_pair_dist[pair_key]:
                best_pair_dist[pair_key] = d

    contacts: list[ResidueContact] = []
    for (l_res, k_res), min_d in sorted(best_pair_dist.items(), key=lambda x: x[1]):
        l_chain, l_name = lock_meta[l_res]
        k_chain, k_name = key_meta[k_res]
        itype, strength, tooltip = classify_residue_pair(
            l_chain, l_name, l_res, k_chain, k_name, k_res, min_d
        )
        contacts.append(
            ResidueContact(
                lock_chain=l_chain,
                lock_resname=l_name,
                lock_resnum=l_res,
                key_chain=k_chain,
                key_resname=k_name,
                key_resnum=k_res,
                min_distance=round(min_d, 2),
                interaction_type=itype,
                strength_score=round(strength, 2),
                tooltip=tooltip,
            )
        )

    # Sort so Salt Bridges and strongest contacts appear first
    type_priority = {
        "Salt Bridge": 0,
        "Charge Repulsion": 1,
        "Greasy Hydrophobic": 2,
        "Hydrogen Bond / Polar": 3,
        "Shape Contact": 4,
    }
    contacts.sort(key=lambda c: (type_priority.get(c.interaction_type, 9), c.min_distance))
    return contacts[:35]


def classify_residue_pair(
    l_chain: str,
    l_name: str,
    l_res: int,
    k_chain: str,
    k_name: str,
    k_res: int,
    dist: float,
) -> tuple[str, float, str]:
    """Classifies a Lock-Key residue interaction into intuitive high-school chemistry categories."""
    q_lock = RESIDUE_CHARGE_PH74.get(l_name, 0.0)
    q_key = RESIDUE_CHARGE_PH74.get(k_name, 0.0)
    h_lock = KYTE_DOOLITTLE_NORM.get(l_name, 0.0)
    h_key = KYTE_DOOLITTLE_NORM.get(k_name, 0.0)

    def _charge_str(q: float) -> str:
        return "(+)" if q >= 0.5 else "(-)" if q <= -0.5 else ""

    l_tag = f"Lock {l_name.capitalize()}{l_res} {_charge_str(q_lock)}".strip()
    k_tag = f"Key {k_name.capitalize()}{k_res} {_charge_str(q_key)}".strip()

    # 1. Opposite charges attract -> Salt Bridge
    if q_lock * q_key <= -0.5 and dist <= 4.5:
        return (
            "Salt Bridge",
            1.0,
            f"{l_tag} attracts {k_tag} ({dist:.1f} Å) — Strong Salt Bridge!",
        )

    # 2. Like charges repel -> Electrostatic Repulsion
    if q_lock * q_key >= 0.5 and dist <= 4.5:
        sign_word = "Plus-Plus" if q_lock > 0 else "Minus-Minus"
        return (
            "Charge Repulsion",
            -1.0,
            f"{l_tag} repels {k_tag} ({dist:.1f} Å) — {sign_word} Charge Clash!",
        )

    # 3. Greasy Hydrophobic Packing (including aromatic Trp/Tyr/Phe pi-stacking/clamping)
    greasy_set = {"ILE", "VAL", "LEU", "PHE", "MET", "ALA", "TRP", "TYR"}
    if (h_lock > 0.25 and h_key > 0.25) or (l_name in greasy_set and k_name in greasy_set):
        return (
            "Greasy Hydrophobic",
            0.75,
            f"{l_tag} packs against {k_tag} ({dist:.1f} Å) — Water-Avoiding Greasy Patch!",
        )

    # 4. Hydrogen Bond / Polar Stickiness
    polar_set = {"SER", "THR", "ASN", "GLN", "TYR", "HIS", "ARG", "LYS", "ASP", "GLU"}
    if l_name in polar_set and k_name in polar_set and dist <= 3.9:
        return (
            "Hydrogen Bond / Polar",
            0.65,
            f"{l_tag} shares a polar Hydrogen Bond with {k_tag} ({dist:.1f} Å).",
        )

    return (
        "Shape Contact",
        0.40,
        f"{l_tag} fits snugly against {k_tag} ({dist:.1f} Å) — 3D Van der Waals Shape Lock.",
    )


def scorecard_to_dict(sc: PuzzleScorecard) -> dict[str, Any]:
    """Serializes a PuzzleScorecard for FastMCP and JSON export."""
    return {
        "total_puzzle_fit_score": sc.total_score,
        "shape_match_score": sc.shape_score,
        "charge_match_score": sc.charge_score,
        "hydrophobic_stickiness_score": sc.hydrophobic_score,
        "pocket_depth_score": sc.pocket_depth_score,
        "alphafold_confidence_plddt": sc.plddt_confidence,
        "contact_surface_area_angstrom2": sc.contact_area_angstrom2,
        "salt_bridges_formed": sc.salt_bridges_count,
        "hydrogen_bonds_formed": sc.hydrogen_bonds_count,
        "atomic_overlaps": sc.atomic_overlaps,
        "top_contacts": [
            {
                "lock_residue": f"{c.lock_chain}:{c.lock_resname}{c.lock_resnum}",
                "key_residue": f"{c.key_chain}:{c.key_resname}{c.key_resnum}",
                "distance_angstrom": c.min_distance,
                "interaction_type": c.interaction_type,
                "explanation": c.tooltip,
            }
            for c in sc.contacts[:12]
        ],
    }
