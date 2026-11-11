"""Automated unit, performance, and biosecurity safety tests for LockKey-Explorer."""

from __future__ import annotations

import time
import numpy as np
import pytest

from core.fetcher import BiosecurityBlockError, validate_safe_id
from core.pipeline import analyze_classroom_complex
from core.viewer_3d import build_interactive_3d_html
from core.what_if_explorer import run_what_if_experiment
from mcp_server import (
    find_surface_pockets,
    load_classroom_protein,
    match_partner_key,
    simulate_what_if_swap,
)
from api.index import build_vercel_analysis_payload


def test_biosecurity_allowlist_blocks_unapproved_proteins() -> None:
    """Ensures pathogens, toxins, or arbitrary PDB IDs outside the classroom allowlist are blocked."""
    for blocked_id in ["1ABC", "6VXX", "7BZ5", "TOXIN_1", "UNKNOWN_PDB"]:
        with pytest.raises(BiosecurityBlockError):
            validate_safe_id(blocked_id)

    # Verify all 4 primary classroom proteins pass validation
    for safe_id in ["3OGO", "1MEL", "6EQE", "1A3N", "AF-A0A0K8P6T7"]:
        cid, meta = validate_safe_id(safe_id)
        assert cid in ("3OGO", "1MEL", "6EQE", "1A3N", "AF-A0A0K8P6T7")
        assert "story" in meta


def test_surface_and_fingerprint_mathematical_bounds_and_speed() -> None:
    """Verifies Fibonacci SAS surface & chemical fingerprint complete quickly and obey [-1, +1] bounds."""
    t0 = time.perf_counter()
    bundle = analyze_classroom_complex("3OGO", min_plddt=65.0, selected_pocket_id=1)
    elapsed = time.perf_counter() - t0

    # Must run on a laptop CPU in under 3.5 seconds (including PDB parse + surface + ICP)
    assert elapsed < 3.5, f"Pipeline took {elapsed:.2f}s (expected < 3.5s)"

    fp = bundle.lock_fingerprint
    assert len(fp.surface.points) > 100
    assert np.all(fp.surface.curvature >= -1.0) and np.all(fp.surface.curvature <= 1.0)
    assert np.all(fp.charge >= -1.0) and np.all(fp.charge <= 1.0)
    assert np.all(fp.hydropathy >= -1.0) and np.all(fp.hydropathy <= 1.0)


def test_ghost_mold_inversion_rules() -> None:
    """Verifies the Three Lock-and-Key Rules on the Ghost Mold (2.8 Å projection)."""
    bundle = analyze_classroom_complex("1MEL", min_plddt=65.0, selected_pocket_id=1)
    idx = bundle.selected_pocket.point_indices
    lock_fp = bundle.lock_fingerprint
    mold = bundle.ghost_mold

    # P_ideal = P_lock + 2.8 * N_lock
    expected_pts = lock_fp.surface.points[idx] + 2.8 * lock_fp.surface.normals[idx]
    np.testing.assert_allclose(mold.points, expected_pts, atol=1e-5)
    # Invert Shape & Charge, Match Greasy Hydropathy
    np.testing.assert_allclose(mold.ideal_curvature, -lock_fp.surface.curvature[idx], atol=1e-5)
    np.testing.assert_allclose(mold.ideal_charge, -lock_fp.charge[idx], atol=1e-5)
    np.testing.assert_allclose(mold.ideal_hydropathy, +lock_fp.hydropathy[idx], atol=1e-5)


def test_what_if_salt_bridge_charge_repulsion_3ogo() -> None:
    """Verifies that swapping Key Arg35 (+) to Asp35 (-) in GFP+Enhancer (3OGO) causes a dramatic score drop."""
    bundle = analyze_classroom_complex("3OGO", min_plddt=65.0, selected_pocket_id=1)
    res = run_what_if_experiment(
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
    assert res.original_resname == "ARG"
    assert res.new_resname == "ASP"
    assert res.score_delta < -15.0
    assert res.badge_status == "degraded"
    assert "repulsion" in res.badge_text.lower()


@pytest.mark.parametrize("preset_id", ["3OGO", "1MEL", "6EQE", "1A3N"])
def test_all_four_classroom_examples_end_to_end(preset_id: str) -> None:
    """Runs all 4 classroom presets through the FastMCP tools and HTML 3D exporter."""
    info = load_classroom_protein(preset_id, min_plddt=65.0)
    assert info["biosecurity_status"] == "VERIFIED_SAFE_CLASSROOM_PROTEIN"

    pockets = find_surface_pockets(preset_id, top_k=3)
    assert len(pockets) >= 1
    assert "center_xyz" in pockets[0]

    match_info = match_partner_key(preset_id, pocket_id=1)
    sc = match_info["scorecard"]
    assert 40.0 <= sc["total_puzzle_fit_score"] <= 100.0
    assert sc["contact_surface_area_angstrom2"] > 50.0

    bundle = analyze_classroom_complex(preset_id, min_plddt=65.0, selected_pocket_id=1)
    html = build_interactive_3d_html(bundle, paint_mode="Puzzle Shape Mode")
    assert "$3Dmol.createViewer" in html
    assert bundle.metadata["short_name"] in html


def test_mcp_simulate_what_if_swap_tool() -> None:
    out = simulate_what_if_swap("3OGO", chain_id="B", residue_num=35, new_amino_acid="ASP")
    assert out["original_amino_acid"] == "ARG"
    assert out["new_amino_acid"] == "ASP"
    assert out["new_score"] < out["baseline_score"]


def test_vercel_serverless_api_payload() -> None:
    payload = build_vercel_analysis_payload(
        preset_id="3OGO",
        min_plddt=65.0,
        pocket_id=1,
        paint_mode="Magnet Charge Mode",
        swap_choice="Negative (-) Aspartate [ASP]",
    )
    assert payload["preset_id"] == "3OGO"
    assert payload["what_if"]["badge_status"] == "degraded"
    assert "$3Dmol.createViewer" in payload["viewer_html"]
    assert len(payload["heatmap"]["z_matrix"]) > 0

