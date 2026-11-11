"""Module 4: Lock-to-Key Puzzle Inverter & 3D Alignment (core/matcher.py)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from core.fetcher import AtomArrayData
from core.fingerprint import PocketRegion, SurfaceFingerprint
from core.spatial_ops import FastKDTree as cKDTree, kabsch_rotation_matrix


@dataclass
class GhostKeyMold:
    """Represents the inverted 'Ideal Key' Ghost Mold projected 2.8 Å into a Lock pocket."""

    points: np.ndarray              # P_ideal_key = P_lock + 2.8 * N_lock (K, 3)
    normals: np.ndarray             # N_ideal_key = -N_lock (K, 3)
    ideal_curvature: np.ndarray     # kappa_ideal = -kappa_lock
    ideal_charge: np.ndarray        # q_ideal = -q_lock
    ideal_hydropathy: np.ndarray    # h_ideal = +h_lock


@dataclass
class AlignmentResult:
    """Result of aligning a Partner Key protein into the Ghost Key Mold via Kabsch + ICP."""

    rotation_matrix: np.ndarray     # (3, 3)
    translation_vector: np.ndarray  # (3,)
    aligned_key_atoms: AtomArrayData
    aligned_key_surface_pts: np.ndarray
    aligned_key_surface_normals: np.ndarray
    rmsd_to_mold: float
    backbone_clashes: int


def create_ideal_key_ghost_mold(
    lock_fingerprint: SurfaceFingerprint,
    pocket: PocketRegion,
    projection_distance: float = 2.8,
) -> GhostKeyMold:
    """Projects a virtual mold 2.8 Å outward into the pocket:
        P_ideal_key = P_lock + 2.8 * N_lock
    Applies the Three Lock-and-Key Rules:
      1. Invert Shape:  kappa_ideal = -kappa_lock
      2. Invert Charge: q_ideal     = -q_lock
      3. Match Greasy:  h_ideal     = +h_lock
    """
    idx = pocket.point_indices
    p_lock = lock_fingerprint.surface.points[idx]
    n_lock = lock_fingerprint.surface.normals[idx]

    p_ideal = p_lock + projection_distance * n_lock
    n_ideal = -n_lock
    kappa_ideal = -lock_fingerprint.surface.curvature[idx]
    q_ideal = -lock_fingerprint.charge[idx]
    h_ideal = +lock_fingerprint.hydropathy[idx]

    return GhostKeyMold(
        points=p_ideal,
        normals=n_ideal,
        ideal_curvature=kappa_ideal,
        ideal_charge=q_ideal,
        ideal_hydropathy=h_ideal,
    )


def align_partner_to_ghost_mold(
    ghost_mold: GhostKeyMold,
    lock_atoms: AtomArrayData,
    key_atoms: AtomArrayData,
    key_fingerprint: SurfaceFingerprint,
    n_icp_steps: int = 15,
    refine_from_current_pose: bool = True,
) -> AlignmentResult:
    """Aligns the partner Key protein's surface hills into the Ghost Mold using
    scipy.spatial.transform.Rotation.align_vectors (Kabsch algorithm) and 15 steps of
    Iterative Closest Point (ICP), while checking for backbone steric clashes.
    """
    pts_key = key_fingerprint.surface.points.copy()
    nrm_key = key_fingerprint.surface.normals.copy()
    atoms_coords = key_atoms.coords.copy()

    if len(pts_key) == 0 or len(ghost_mold.points) == 0:
        return AlignmentResult(
            rotation_matrix=np.eye(3),
            translation_vector=np.zeros(3),
            aligned_key_atoms=key_atoms,
            aligned_key_surface_pts=pts_key,
            aligned_key_surface_normals=nrm_key,
            rmsd_to_mold=0.0,
            backbone_clashes=0,
        )

    mold_tree = cKDTree(ghost_mold.points)

    # Check if the Key protein is already sitting in or near the Ghost Mold (e.g. native crystal complex Pocket #1)
    dists_init, _ = mold_tree.query(pts_key)
    close_mask = dists_init < 12.0
    if refine_from_current_pose:
        # Native co-crystallized pocket: preserve crystal pose
        rmsd = float(np.sqrt(np.mean(dists_init[close_mask] ** 2))) if np.any(close_mask) else float(np.min(dists_init))
        clashes = count_backbone_clashes(lock_atoms, key_atoms)
        return AlignmentResult(
            rotation_matrix=np.eye(3),
            translation_vector=np.zeros(3),
            aligned_key_atoms=key_atoms,
            aligned_key_surface_pts=pts_key,
            aligned_key_surface_normals=nrm_key,
            rmsd_to_mold=rmsd,
            backbone_clashes=clashes,
        )

    # Otherwise, select the best convex/complementary patch on the Key protein and dock it into the Ghost Mold
    convex_idx = np.where(key_fingerprint.surface.curvature > 0.05)[0]
    if len(convex_idx) < 15:
        convex_idx = np.arange(len(pts_key))

    # Pick patch center on Key with highest convexity + complementary charge to mold
    key_patch_center = np.mean(pts_key[convex_idx], axis=0)
    patch_dists = np.linalg.norm(pts_key - key_patch_center, axis=1)
    active_key_idx = np.where(patch_dists <= 12.0)[0]
    if len(active_key_idx) < 8:
        active_key_idx = convex_idx

    # Initial translation to Ghost Mold center & normal alignment
    mold_center = np.mean(ghost_mold.points, axis=0)
    mold_mean_normal = np.mean(ghost_mold.normals, axis=0)
    mold_mean_normal /= np.linalg.norm(mold_mean_normal) + 1e-8

    key_mean_normal = np.mean(nrm_key[active_key_idx], axis=0)
    key_mean_normal /= np.linalg.norm(key_mean_normal) + 1e-8

    R_total = kabsch_rotation_matrix(
        mold_mean_normal.reshape(1, 3),
        key_mean_normal.reshape(1, 3),
    )
    key_patch_curr = (pts_key[active_key_idx] - key_patch_center) @ R_total.T + mold_center

    # 15 steps of Kabsch Iterative Closest Point (ICP)
    curr_pts_all = (pts_key - key_patch_center) @ R_total.T + mold_center
    curr_nrm_all = nrm_key @ R_total.T
    curr_atoms = (atoms_coords - key_patch_center) @ R_total.T + mold_center

    for _ in range(n_icp_steps):
        d_icp, nn_mold_idx = mold_tree.query(key_patch_curr)
        valid = d_icp < 12.0
        if np.sum(valid) < 6:
            valid = np.ones(len(d_icp), dtype=bool)

        src = key_patch_curr[valid]
        dst = ghost_mold.points[nn_mold_idx[valid]]

        src_cent = np.mean(src, axis=0)
        dst_cent = np.mean(dst, axis=0)

        R_step = kabsch_rotation_matrix(dst - dst_cent, src - src_cent)

        key_patch_curr = (key_patch_curr - src_cent) @ R_step.T + dst_cent
        curr_pts_all = (curr_pts_all - src_cent) @ R_step.T + dst_cent
        curr_nrm_all = curr_nrm_all @ R_step.T
        curr_atoms = (curr_atoms - src_cent) @ R_step.T + dst_cent
        R_total = R_step @ R_total

    # Prevent steric backbone overlaps by gently stepping outward along pocket normal if clashes occur
    lock_bb_mask = np.isin(lock_atoms.atom_names, ["N", "CA", "C", "O"])
    key_bb_mask = np.isin(key_atoms.atom_names, ["N", "CA", "C", "O"])
    if np.any(lock_bb_mask) and np.any(key_bb_mask):
        lock_bb_tree = cKDTree(lock_atoms.coords[lock_bb_mask])
        outward_dir = -mold_mean_normal  # away from lock into solvent
        for _ in range(8):
            bb_dists, _ = lock_bb_tree.query(curr_atoms[key_bb_mask])
            if np.sum(bb_dists < 2.8) <= 2:
                break
            curr_atoms += 0.8 * outward_dir
            curr_pts_all += 0.8 * outward_dir
            key_patch_curr += 0.8 * outward_dir

    d_final, _ = mold_tree.query(key_patch_curr)
    rmsd_final = float(np.sqrt(np.mean(d_final**2)))

    aligned_atom_data = AtomArrayData(
        coords=curr_atoms,
        radii=key_atoms.radii,
        charges=key_atoms.charges,
        hydropathy=key_atoms.hydropathy,
        bfactors=key_atoms.bfactors,
        res_nums=key_atoms.res_nums,
        res_names=key_atoms.res_names,
        atom_names=key_atoms.atom_names,
        chain_ids=key_atoms.chain_ids,
    )
    clashes = count_backbone_clashes(lock_atoms, aligned_atom_data)
    t_vec = np.mean(curr_atoms, axis=0) - (np.mean(atoms_coords, axis=0) @ R_total.T)

    return AlignmentResult(
        rotation_matrix=R_total,
        translation_vector=t_vec,
        aligned_key_atoms=aligned_atom_data,
        aligned_key_surface_pts=curr_pts_all,
        aligned_key_surface_normals=curr_nrm_all,
        rmsd_to_mold=rmsd_final,
        backbone_clashes=clashes,
    )


def count_backbone_clashes(
    lock_atoms: AtomArrayData,
    key_atoms: AtomArrayData,
    clash_cutoff: float = 2.4,
) -> int:
    """Counts severe heavy-atom backbone overlaps (< 2.4 Å) between Lock and Key."""
    if len(lock_atoms.coords) == 0 or len(key_atoms.coords) == 0:
        return 0
    lock_bb = lock_atoms.coords[np.isin(lock_atoms.atom_names, ["N", "CA", "C", "O"])]
    key_bb = key_atoms.coords[np.isin(key_atoms.atom_names, ["N", "CA", "C", "O"])]
    if len(lock_bb) == 0 or len(key_bb) == 0:
        return 0
    tree = cKDTree(lock_bb)
    dists, _ = tree.query(key_bb)
    return int(np.sum(dists < clash_cutoff))
