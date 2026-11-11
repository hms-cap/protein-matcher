"""Module 2: 3D Molecular Surface & Puzzle Curvature Extractor (core/surface.py)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from core.fetcher import AtomArrayData
from core.spatial_ops import FastKDTree as cKDTree

WATER_PROBE_RADIUS = 1.4  # Angstroms (Å)


@dataclass
class MolecularSurface:
    """Represents the 3D Molecular Skin (Solvent-Accessible Surface) with outward normals & curvature."""

    points: np.ndarray          # (N, 3) float64
    normals: np.ndarray         # (N, 3) float64 outward unit vectors
    curvature: np.ndarray       # (N,) float64 in [-1, +1] (< -0.2 concave valley, > +0.2 convex hill)
    atom_indices: np.ndarray    # (N,) int32 parent atom index in AtomArrayData
    area_per_point: np.ndarray  # (N,) float64 approximate surface area patch (Å^2)


def generate_fibonacci_sphere(n_points: int = 64) -> np.ndarray:
    """Generates n_points evenly distributed on the unit sphere using the golden angle."""
    indices = np.arange(n_points, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / n_points)
    theta = np.pi * (1.0 + 5.0**0.5) * indices

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


_UNIT_SPHERE_64 = generate_fibonacci_sphere(64)


def compute_molecular_surface(
    atom_data: AtomArrayData,
    n_sphere_points: int = 64,
    probe_radius: float = WATER_PROBE_RADIUS,
    subsample_max_points: int = 3500,
) -> MolecularSurface:
    """Rolls a water molecule (r = 1.4 Å) over all atoms using a 64-point Fibonacci sphere
    and cKDTree pruning, then computes local normal-tilt curvature kappa_i in [-1, +1].
    """
    n_atoms = len(atom_data.coords)
    if n_atoms == 0:
        return MolecularSurface(
            points=np.zeros((0, 3)),
            normals=np.zeros((0, 3)),
            curvature=np.zeros(0),
            atom_indices=np.zeros(0, dtype=np.int32),
            area_per_point=np.zeros(0),
        )

    unit_sphere = (
        _UNIT_SPHERE_64 if n_sphere_points == 64 else generate_fibonacci_sphere(n_sphere_points)
    )
    expanded_radii = atom_data.radii + probe_radius  # (M,)
    max_radius = float(np.max(expanded_radii))

    # Build KD-tree of atom centers to find neighbor atoms within 2 * max_radius
    atom_tree = cKDTree(atom_data.coords)

    surviving_points = []
    surviving_normals = []
    surviving_atom_idx = []
    surviving_areas = []

    # Query atom neighbors once for all atoms
    neighbor_lists = atom_tree.query_ball_point(atom_data.coords, r=2.0 * max_radius + 0.1)

    for i in range(n_atoms):
        r_i = expanded_radii[i]
        center_i = atom_data.coords[i]
        # Candidate points on sphere i: (K, 3)
        cand_pts = center_i + r_i * unit_sphere
        neighbors = [j for j in neighbor_lists[i] if j != i]

        if neighbors:
            nbr_coords = atom_data.coords[neighbors]          # (L, 3)
            nbr_radii_sq = (expanded_radii[neighbors] - 1e-3) ** 2  # (L,)
            # Compute squared distances from cand_pts (K, 1, 3) to nbr_coords (1, L, 3)
            diff = cand_pts[:, None, :] - nbr_coords[None, :, :]
            dist_sq = np.einsum("klc,klc->kl", diff, diff)
            is_buried = np.any(dist_sq < nbr_radii_sq[None, :], axis=1)
            keep_mask = ~is_buried
        else:
            keep_mask = np.ones(len(unit_sphere), dtype=bool)

        if np.any(keep_mask):
            kept_pts = cand_pts[keep_mask]
            kept_normals = unit_sphere[keep_mask]
            k_kept = len(kept_pts)
            patch_area = (4.0 * np.pi * (r_i**2)) / float(len(unit_sphere))

            surviving_points.append(kept_pts)
            surviving_normals.append(kept_normals)
            surviving_atom_idx.append(np.full(k_kept, i, dtype=np.int32))
            surviving_areas.append(np.full(k_kept, patch_area, dtype=np.float64))

    if not surviving_points:
        return MolecularSurface(
            points=np.zeros((0, 3)),
            normals=np.zeros((0, 3)),
            curvature=np.zeros(0),
            atom_indices=np.zeros(0, dtype=np.int32),
            area_per_point=np.zeros(0),
        )

    points = np.vstack(surviving_points)
    normals = np.vstack(surviving_normals)
    atom_indices = np.concatenate(surviving_atom_idx)
    area_per_point = np.concatenate(surviving_areas)

    # Deterministic stride subsampling if a large protein generates > subsample_max_points
    if len(points) > subsample_max_points:
        step = len(points) / float(subsample_max_points)
        idx = np.floor(np.arange(subsample_max_points) * step).astype(np.int32)
        scale_factor = len(points) / float(subsample_max_points)
        points = points[idx]
        normals = normals[idx]
        atom_indices = atom_indices[idx]
        area_per_point = area_per_point[idx] * scale_factor

    curvature = compute_surface_curvature(points, normals, radius=8.0)
    return MolecularSurface(
        points=points,
        normals=normals,
        curvature=curvature,
        atom_indices=atom_indices,
        area_per_point=area_per_point,
    )


def compute_surface_curvature(
    points: np.ndarray,
    normals: np.ndarray,
    radius: float = 8.0,
) -> np.ndarray:
    """Computes Pocket Valley vs. Hill Curvature kappa_i in [-1, +1]:
    kappa_i = clip( (1/|M_i|) * sum_{j in M_i} ((p_j - p_i) . (n_j - n_i)) / ||p_j - p_i||^2, -1.0, 1.0 )
    Negative kappa_i < -0.2 = Concave Valley / Keyhole ("The Lock Pocket")
    Positive kappa_i > +0.2 = Convex Hill / Bump ("The Key Tooth")
    """
    n_pts = len(points)
    if n_pts == 0:
        return np.zeros(0, dtype=np.float64)

    surf_tree = cKDTree(points)
    neighbors_list = surf_tree.query_ball_point(points, r=radius)
    kappa = np.zeros(n_pts, dtype=np.float64)

    # Scale factor 2.2 maps typical 8 Å protein surface normal divergences cleanly into [-1, +1]
    for i, nbrs in enumerate(neighbors_list):
        if len(nbrs) <= 1:
            continue
        nbr_idx = np.asarray([j for j in nbrs if j != i], dtype=np.int32)
        if len(nbr_idx) == 0:
            continue
        dp = points[nbr_idx] - points[i]       # (M, 3)
        dn = normals[nbr_idx] - normals[i]     # (M, 3)
        dist_sq = np.einsum("mc,mc->m", dp, dp)
        valid = dist_sq > 1e-4
        if not np.any(valid):
            continue
        dot_val = np.einsum("mc,mc->m", dp[valid], dn[valid]) / dist_sq[valid]
        # Amplify slightly by radius/4.0 so concave pockets reach < -0.2 and convex hills > +0.2
        raw_k = float(np.mean(dot_val)) * 2.2
        kappa[i] = np.clip(raw_k, -1.0, 1.0)

    return kappa
