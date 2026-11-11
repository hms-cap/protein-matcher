"""Module 3: Chemical Puzzle Rules & Pocket Discovery (core/fingerprint.py)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from core.fetcher import AtomArrayData
from core.spatial_ops import cluster_points_by_distance
from core.surface import MolecularSurface


@dataclass
class SurfaceFingerprint:
    """Stores the 3D Molecular Skin and its 3 chemical puzzle features f_i = [kappa_i, q_i, h_i]."""

    surface: MolecularSurface
    charge: np.ndarray          # q_i in [-1, +1]
    hydropathy: np.ndarray      # h_i in [-1, +1]
    plddt: np.ndarray           # (N,) confidence from parent atom B-factor
    features: np.ndarray        # (N, 3) [kappa_i, q_i, h_i]


@dataclass
class PocketRegion:
    """Represents a discovered concave keyhole pocket on the Lock protein."""

    pocket_id: int
    name: str
    center: np.ndarray          # (3,) xyz coordinates
    point_indices: np.ndarray   # indices into SurfaceFingerprint
    mean_curvature: float       # negative = concave valley depth
    mean_charge: float          # in [-1, +1]
    mean_hydropathy: float      # in [-1, +1]
    surface_area: float         # in Å^2
    key_residues: list[str]     # e.g., ["GLU142", "TYR145", "HIS148"]


def compute_electrostatic_potential(
    surface_points: np.ndarray,
    atom_data: AtomArrayData,
) -> np.ndarray:
    """Calculates the local electric field at each surface point using Coulomb's law:
    q_i = tanh( sum_{k in atoms} (332.0 * z_k) / (4 * ||p_i - a_k||^2) )
    Scaled with a realistic distance softening (+ 4.0 Å^2) so surface potentials span [-1, +1] smoothly.
    """
    n_pts = len(surface_points)
    if n_pts == 0:
        return np.zeros(0, dtype=np.float64)

    charged_mask = np.abs(atom_data.charges) > 1e-4
    if not np.any(charged_mask):
        return np.zeros(n_pts, dtype=np.float64)

    c_coords = atom_data.coords[charged_mask]    # (K, 3)
    c_charges = atom_data.charges[charged_mask]  # (K,)

    # Compute pairwise squared distances in chunks to keep memory tiny on laptops
    potentials = np.zeros(n_pts, dtype=np.float64)
    chunk_size = 512
    for start in range(0, n_pts, chunk_size):
        end = min(start + chunk_size, n_pts)
        pts_chunk = surface_points[start:end]  # (B, 3)
        diff = pts_chunk[:, None, :] - c_coords[None, :, :]  # (B, K, 3)
        dist_sq = np.einsum("bkc,bkc->bk", diff, diff)
        # Apply distance cutoff (15 Å) and softening so tanh spans [-1, +1] clearly
        within_cutoff = dist_sq < 225.0
        inv_r2 = np.where(within_cutoff, 1.0 / np.maximum(dist_sq, 4.0), 0.0)
        # Normalize constant so single nearby charged sidechain (~3.5 Å) gives ~0.65-0.85
        coulomb_sum = np.sum((332.0 * c_charges[None, :] * inv_r2) / (4.0 * 18.0), axis=1)
        potentials[start:end] = np.tanh(coulomb_sum)

    return np.clip(potentials, -1.0, 1.0)


def compute_surface_fingerprint(
    surface: MolecularSurface,
    atom_data: AtomArrayData,
) -> SurfaceFingerprint:
    """Computes f_i = [kappa_i, q_i, h_i] for every point on the protein's 3D skin."""
    if len(surface.points) == 0:
        empty = np.zeros(0, dtype=np.float64)
        return SurfaceFingerprint(
            surface=surface,
            charge=empty,
            hydropathy=empty,
            plddt=empty,
            features=np.zeros((0, 3), dtype=np.float64),
        )

    q_i = compute_electrostatic_potential(surface.points, atom_data)
    h_i = np.clip(atom_data.hydropathy[surface.atom_indices], -1.0, 1.0)
    plddt_i = atom_data.bfactors[surface.atom_indices]
    features = np.column_stack([surface.curvature, q_i, h_i])

    return SurfaceFingerprint(
        surface=surface,
        charge=q_i,
        hydropathy=h_i,
        plddt=plddt_i,
        features=features,
    )


def discover_top_pockets(
    fingerprint: SurfaceFingerprint,
    atom_data: AtomArrayData,
    top_k: int = 3,
    concavity_threshold: float = -0.15,
    preferred_contact_center: np.ndarray | None = None,
) -> list[PocketRegion]:
    """Groups concave valley points (kappa_i < -0.15) into the Top-K Candidate Keyholes
    using scipy.cluster.hierarchy.fcluster.
    If preferred_contact_center is provided (e.g., from a known partner protein or active site),
    Pocket #1 is anchored to the primary biologically active binding groove!
    """
    pts = fingerprint.surface.points
    kappa = fingerprint.surface.curvature
    if len(pts) == 0:
        return []

    valley_idx = np.where(kappa < concavity_threshold)[0]
    if len(valley_idx) < 12:
        # Fallback to the most concave 15% of points
        cutoff = float(np.percentile(kappa, 15))
        valley_idx = np.where(kappa <= cutoff)[0]

    valley_pts = pts[valley_idx]
    if len(valley_pts) >= 4:
        labels = cluster_points_by_distance(valley_pts, cutoff=12.5)
    else:
        labels = np.ones(len(valley_idx), dtype=np.int32)

    clusters: list[tuple[float, np.ndarray]] = []
    for cid in np.unique(labels):
        member_mask = labels == cid
        member_global_idx = valley_idx[member_mask]
        if len(member_global_idx) < 5:
            continue
        #Expand pocket slightly to include immediate 6.0 Å rim points for complete keyhole representation
        cluster_center = np.mean(pts[member_global_idx], axis=0)
        rim_dists = np.linalg.norm(pts - cluster_center, axis=1)
        expanded_idx = np.where((rim_dists <= 9.5) & (kappa < 0.25))[0]
        if len(expanded_idx) < len(member_global_idx):
            expanded_idx = member_global_idx

        depth_score = float(len(member_global_idx)) * abs(float(np.mean(kappa[member_global_idx])))
        if preferred_contact_center is not None:
            dist_to_pref = float(np.linalg.norm(cluster_center - preferred_contact_center))
            # Boost cluster closest to the true biological interface so Pocket #1 is always the Main Active Groove
            if dist_to_pref < 14.0:
                depth_score += 500.0 / (1.0 + 0.1 * dist_to_pref)

        clusters.append((depth_score, expanded_idx))

    clusters.sort(key=lambda x: x[0], reverse=True)

    pockets: list[PocketRegion] = []
    used_centers: list[np.ndarray] = []

    for _, idx_arr in clusters:
        c_center = np.mean(pts[idx_arr], axis=0)
        if any(np.linalg.norm(c_center - uc) < 9.0 for uc in used_centers):
            continue
        used_centers.append(c_center)

        p_num = len(pockets) + 1
        label_name = (
            "Pocket #1 (Main Active Groove)"
            if p_num == 1
            else f"Pocket #{p_num} (Secondary Surface Groove)"
        )

        # Identify top residues lining this pocket
        parent_atoms = fingerprint.surface.atom_indices[idx_arr]
        res_labels = []
        seen_res = set()
        for a_idx in parent_atoms:
            r_tag = f"{atom_data.res_names[a_idx]}{atom_data.res_nums[a_idx]}"
            if r_tag not in seen_res:
                seen_res.add(r_tag)
                res_labels.append(r_tag)

        pockets.append(
            PocketRegion(
                pocket_id=p_num,
                name=label_name,
                center=c_center,
                point_indices=idx_arr,
                mean_curvature=float(np.mean(kappa[idx_arr])),
                mean_charge=float(np.mean(fingerprint.charge[idx_arr])),
                mean_hydropathy=float(np.mean(fingerprint.hydropathy[idx_arr])),
                surface_area=float(np.sum(fingerprint.surface.area_per_point[idx_arr])),
                key_residues=res_labels[:10],
            )
        )
        if len(pockets) >= top_k:
            break

    return pockets


def pocket_to_dict(pocket: PocketRegion) -> dict[str, Any]:
    """Serializes a PocketRegion into a JSON-friendly dictionary for MCP & UI."""
    charge_desc = (
        "Positively Charged (+)"
        if pocket.mean_charge > 0.15
        else "Negatively Charged (-)"
        if pocket.mean_charge < -0.15
        else "Mostly Neutral / Mixed"
    )
    hydro_desc = (
        "Greasy / Hydrophobic"
        if pocket.mean_hydropathy > 0.15
        else "Water-Loving / Polar"
        if pocket.mean_hydropathy < -0.15
        else "Balanced Polar/Apolar"
    )
    return {
        "pocket_id": pocket.pocket_id,
        "name": pocket.name,
        "center_xyz": [round(float(x), 2) for x in pocket.center],
        "valley_depth_curvature": round(pocket.mean_curvature, 3),
        "electric_charge": round(pocket.mean_charge, 3),
        "charge_character": charge_desc,
        "greasiness_hydropathy": round(pocket.mean_hydropathy, 3),
        "hydropathy_character": hydro_desc,
        "surface_area_angstrom2": round(pocket.surface_area, 1),
        "lining_residues": pocket.key_residues,
    }
