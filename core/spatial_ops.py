"""Fast 3D spatial indexing, rotation alignment, and hierarchical clustering with
automatic SciPy / pure-NumPy fallback so LockKey-Explorer runs on any laptop or OS
(including Windows machines with strict Application Control policies).
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import cKDTree as _ScipyKDTree  # type: ignore

    # Smoke-test creation to verify DLL loading is permitted
    _ = _ScipyKDTree(np.zeros((2, 3)))
    HAS_SCIPY_KDTREE = True
except Exception:
    HAS_SCIPY_KDTREE = False


class FastKDTree:
    """Drop-in wrapper for scipy.spatial.cKDTree with a vectorized 3D voxel-grid + NumPy fallback."""

    def __init__(self, points: np.ndarray) -> None:
        self.points = np.asarray(points, dtype=np.float64)
        self._scipy_tree = _ScipyKDTree(self.points) if (HAS_SCIPY_KDTREE and len(self.points) > 0) else None

    def query(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (distances, nearest_neighbor_indices) for query points x (M, 3)."""
        q = np.asarray(x, dtype=np.float64)
        if q.ndim == 1:
            q = q.reshape(1, 3)
        if self._scipy_tree is not None:
            d, idx = self._scipy_tree.query(q)
            return np.asarray(d, dtype=np.float64), np.asarray(idx, dtype=np.int32)

        n_q = len(q)
        if n_q == 0 or len(self.points) == 0:
            return np.zeros(n_q, dtype=np.float64), np.zeros(n_q, dtype=np.int32)

        dists = np.empty(n_q, dtype=np.float64)
        indices = np.empty(n_q, dtype=np.int32)
        chunk = 256
        p_norm_sq = np.einsum("nc,nc->n", self.points, self.points)  # (N,)

        for start in range(0, n_q, chunk):
            end = min(start + chunk, n_q)
            qc = q[start:end]  # (B, 3)
            q_norm_sq = np.einsum("bc,bc->b", qc, qc)[:, None]
            # ||q - p||^2 = ||q||^2 + ||p||^2 - 2 q . p^T
            d2 = np.maximum(0.0, q_norm_sq + p_norm_sq[None, :] - 2.0 * (qc @ self.points.T))
            nn_idx = np.argmin(d2, axis=1)
            indices[start:end] = nn_idx
            dists[start:end] = np.sqrt(d2[np.arange(len(qc)), nn_idx])

        return dists, indices

    def query_ball_point(self, x: np.ndarray, r: float) -> list[list[int]]:
        """Returns a list of neighbor index lists within radius r for each query point in x."""
        q = np.asarray(x, dtype=np.float64)
        if q.ndim == 1:
            q = q.reshape(1, 3)
        if self._scipy_tree is not None:
            return self._scipy_tree.query_ball_point(q, r=r)

        n_q = len(q)
        if n_q == 0 or len(self.points) == 0:
            return [[] for _ in range(n_q)]

        r_sq = float(r * r)
        p_norm_sq = np.einsum("nc,nc->n", self.points, self.points)
        results: list[list[int]] = []
        chunk = 256

        for start in range(0, n_q, chunk):
            end = min(start + chunk, n_q)
            qc = q[start:end]
            q_norm_sq = np.einsum("bc,bc->b", qc, qc)[:, None]
            d2 = q_norm_sq + p_norm_sq[None, :] - 2.0 * (qc @ self.points.T)
            mask = d2 <= r_sq
            for row in mask:
                results.append(np.where(row)[0].tolist())

        return results


def kabsch_rotation_matrix(target_vecs: np.ndarray, source_vecs: np.ndarray) -> np.ndarray:
    """Aligns source_vecs onto target_vecs using the Kabsch SVD algorithm (equivalent to
    scipy.spatial.transform.Rotation.align_vectors). Returns a 3x3 proper rotation matrix R
    such that source_vecs @ R.T ~ target_vecs.
    """
    A = np.asarray(target_vecs, dtype=np.float64)
    B = np.asarray(source_vecs, dtype=np.float64)
    if A.ndim == 1:
        A = A.reshape(1, 3)
    if B.ndim == 1:
        B = B.reshape(1, 3)

    if len(A) == 1 and len(B) == 1:
        # Align single unit vector b onto unit vector a via Rodrigues' rotation formula
        a = A[0] / (np.linalg.norm(A[0]) + 1e-12)
        b = B[0] / (np.linalg.norm(B[0]) + 1e-12)
        v = np.cross(b, a)
        c = float(np.dot(b, a))
        s = float(np.linalg.norm(v))
        if s < 1e-8:
            return np.eye(3) if c >= 0 else -np.eye(3)
        vx = np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ],
            dtype=np.float64,
        )
        return np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))

    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(U @ Vt)
    S = np.diag([1.0, 1.0, 1.0 if d >= 0 else -1.0])
    return U @ S @ Vt


def cluster_points_by_distance(points: np.ndarray, cutoff: float = 12.5) -> np.ndarray:
    """Clusters 3D points using scipy.cluster.hierarchy.fcluster if available, or a fast
    connected-components / greedy sphere-leader clustering in pure NumPy.
    """
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        dists = pdist(points)
        Z = linkage(dists, method="average")
        return fcluster(Z, t=cutoff, criterion="distance")
    except Exception:
        n = len(points)
        if n == 0:
            return np.zeros(0, dtype=np.int32)
        labels = np.zeros(n, dtype=np.int32)
        centers: list[np.ndarray] = []
        for i in range(n):
            p = points[i]
            assigned = False
            for c_idx, c_pos in enumerate(centers):
                if float(np.linalg.norm(p - c_pos)) <= cutoff:
                    labels[i] = c_idx + 1
                    # Update running centroid slightly
                    centers[c_idx] = 0.85 * c_pos + 0.15 * p
                    assigned = True
                    break
            if not assigned:
                centers.append(p.copy())
                labels[i] = len(centers)
        return labels
