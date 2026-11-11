"""Vercel Serverless Python Function entrypoint (api/index.py) for LockKey-Explorer."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse
import numpy as np

# Ensure repository root is on sys.path in Vercel serverless runtime
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.fingerprint import pocket_to_dict
from core.pipeline import analyze_classroom_complex
from core.scorer import scorecard_to_dict
from core.viewer_3d import build_interactive_3d_html
from core.what_if_explorer import EDUCATIONAL_SWAP_OPTIONS, run_what_if_experiment, what_if_to_dict


def build_vercel_analysis_payload(
    preset_id: str = "3OGO",
    min_plddt: float = 65.0,
    pocket_id: int = 1,
    paint_mode: str = "Puzzle Shape Mode",
    show_ghost: bool = True,
    show_partner: bool = True,
    target_chain: str | None = None,
    target_resnum: int | None = None,
    swap_choice: str = "Original (Natural Amino Acid)",
) -> dict:
    """Computes the complete biophysical analysis, What-If simulation, 3D viewer HTML,
    and 2D Touch-Map matrix for the Vercel web application frontend.
    """
    bundle = analyze_classroom_complex(
        preset_or_id=preset_id,
        min_plddt=float(min_plddt),
        selected_pocket_id=int(pocket_id),
    )

    feat = bundle.metadata.get("featured_contact", {})
    default_chain = str(feat.get("key_chain", bundle.metadata["key_chain"]))
    default_resnum = int(feat.get("key_resnum", 35))

    use_chain = target_chain if target_chain else default_chain
    use_resnum = int(target_resnum) if target_resnum is not None else default_resnum

    # Build list of selectable interface contacts for the What-If dropdown
    contact_choices = [
        {
            "label": f"Chain {default_chain}: {feat.get('key_resname', 'ARG')}{default_resnum} (Featured: {feat.get('description', '')})",
            "chain": default_chain,
            "resnum": default_resnum,
        }
    ]
    seen_cr = {(default_chain, default_resnum)}
    for c in bundle.scorecard.contacts[:10]:
        cr = (c.key_chain, int(c.key_resnum))
        if cr not in seen_cr:
            seen_cr.add(cr)
            contact_choices.append(
                {
                    "label": f"Chain {c.key_chain}: {c.key_resname}{c.key_resnum} ↔ Lock {c.lock_resname}{c.lock_resnum} ({c.interaction_type})",
                    "chain": c.key_chain,
                    "resnum": int(c.key_resnum),
                }
            )

    if (use_chain, use_resnum) not in seen_cr:
        use_chain, use_resnum = default_chain, default_resnum

    what_if_res = run_what_if_experiment(
        lock_atoms=bundle.lock_atoms,
        key_atoms=bundle.key_atoms,
        lock_fingerprint=bundle.lock_fingerprint,
        key_fingerprint=bundle.key_fingerprint,
        pocket=bundle.selected_pocket,
        ghost_mold=bundle.ghost_mold,
        chain_id=use_chain,
        residue_num=use_resnum,
        new_amino_acid=swap_choice,
    )

    viewer_html = build_interactive_3d_html(
        bundle=bundle,
        paint_mode=paint_mode,
        show_ghost_mold=show_ghost,
        show_partner_key=show_partner,
        highlight_residue=(use_chain, use_resnum),
        height=510,
    )

    # Build 2D Touch-Map Heatmap arrays
    active_sc = what_if_res.swapped_scorecard
    top_contacts = active_sc.contacts[:14]
    lock_labels: list[str] = []
    key_labels: list[str] = []
    for c in top_contacts:
        ll = f"Lock {c.lock_resname}{c.lock_resnum}"
        kl = f"Key {c.key_resname}{c.key_resnum}"
        if ll not in lock_labels:
            lock_labels.append(ll)
        if kl not in key_labels:
            key_labels.append(kl)

    z_mat = np.zeros((len(key_labels), len(lock_labels)), dtype=float)
    hover_mat = [
        ["No direct contact (< 4.8 Å)" for _ in range(len(lock_labels))]
        for _ in range(len(key_labels))
    ]
    for c in top_contacts:
        li = lock_labels.index(f"Lock {c.lock_resname}{c.lock_resnum}")
        ki = key_labels.index(f"Key {c.key_resname}{c.key_resnum}")
        z_mat[ki, li] = c.strength_score
        hover_mat[ki][li] = f"<b>{c.interaction_type}</b><br>{c.tooltip}"

    return {
        "preset_id": bundle.preset_id,
        "metadata": bundle.metadata,
        "trim_stats": bundle.trim_stats,
        "pockets": [pocket_to_dict(p) for p in bundle.pockets],
        "selected_pocket": pocket_to_dict(bundle.selected_pocket),
        "baseline_scorecard": scorecard_to_dict(bundle.scorecard),
        "active_scorecard": scorecard_to_dict(active_sc),
        "what_if": {
            **what_if_to_dict(what_if_res),
            "badge_status": what_if_res.badge_status,
            "selected_chain": use_chain,
            "selected_resnum": use_resnum,
        },
        "contact_choices": contact_choices,
        "swap_options": list(EDUCATIONAL_SWAP_OPTIONS.keys()),
        "heatmap": {
            "x_lock_labels": lock_labels,
            "y_key_labels": key_labels,
            "z_matrix": z_mat.tolist(),
            "hover_text": hover_mat,
        },
        "viewer_html": viewer_html,
        "trimmed_pdb_str": bundle.trimmed_pdb_str,
    }


class handler(BaseHTTPRequestHandler):
    """Standard Vercel Python Function HTTP handler."""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        preset_id = qs.get("preset", ["3OGO"])[0]
        min_plddt = float(qs.get("min_plddt", ["65"])[0])
        pocket_id = int(qs.get("pocket_id", ["1"])[0])
        paint_mode = qs.get("paint_mode", ["Puzzle Shape Mode"])[0]
        show_ghost = qs.get("show_ghost", ["true"])[0].lower() == "true"
        show_partner = qs.get("show_partner", ["true"])[0].lower() == "true"
        target_chain = qs.get("chain", [None])[0]
        target_res = qs.get("resnum", [None])[0]
        swap_choice = qs.get("swap", ["Original (Natural Amino Acid)"])[0]

        try:
            payload = build_vercel_analysis_payload(
                preset_id=preset_id,
                min_plddt=min_plddt,
                pocket_id=pocket_id,
                paint_mode=paint_mode,
                show_ghost=show_ghost,
                show_partner=show_partner,
                target_chain=target_chain,
                target_resnum=int(target_res) if target_res else None,
                swap_choice=swap_choice,
            )
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "public, max-age=30")
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            err = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(err)
