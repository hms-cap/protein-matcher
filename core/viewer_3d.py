"""3D WebGL Viewer Generator (py3Dmol) and Standalone HTML Exporter for LockKey-Explorer."""

from __future__ import annotations

import json
import numpy as np

from core.pipeline import LockKeyAnalysisBundle


def _rgb_hex(r: float, g: float, b: float) -> str:
    ri = int(np.clip(round(r * 255), 0, 255))
    gi = int(np.clip(round(g * 255), 0, 255))
    bi = int(np.clip(round(b * 255), 0, 255))
    return f"0x{ri:02x}{gi:02x}{bi:02x}"


def map_value_to_paint_hex(value: float, paint_mode: str) -> str:
    """Maps a normalized property value to the requested classroom color palette."""
    if paint_mode == "Puzzle Shape Mode":
        # Deep Purple (#5b21b6) = Concave Keyhole (-1) <-> Gold (#f59e0b) = Convex Bump (+1)
        t = float(np.clip((value + 0.6) / 1.2, 0.0, 1.0))
        r = (1 - t) * 0.36 + t * 0.96
        g = (1 - t) * 0.13 + t * 0.62
        b = (1 - t) * 0.71 + t * 0.04
        return _rgb_hex(r, g, b)

    if paint_mode == "Magnet Charge Mode":
        # Blue = Positive (+1) <-> White = Neutral (0) <-> Red = Negative (-1)
        v = float(np.clip(value, -1.0, 1.0))
        if v >= 0.0:
            # White to Royal Blue (#2563eb)
            t = min(1.0, v * 1.4)
            return _rgb_hex(1.0 - 0.85 * t, 1.0 - 0.61 * t, 1.0 - 0.08 * t)
        else:
            # White to Crimson Red (#dc2626)
            t = min(1.0, abs(v) * 1.4)
            return _rgb_hex(1.0 - 0.14 * t, 1.0 - 0.85 * t, 1.0 - 0.85 * t)

    if paint_mode == "Oil-vs-Water Mode":
        # Orange (#f97316) = Greasy Hydrophobic (+1) <-> Teal (#0d9488) = Water-Loving (-1)
        t = float(np.clip((value + 0.85) / 1.7, 0.0, 1.0))
        r = (1 - t) * 0.05 + t * 0.98
        g = (1 - t) * 0.58 + t * 0.45
        b = (1 - t) * 0.53 + t * 0.09
        return _rgb_hex(r, g, b)

    # Default: AlphaFold Confidence Mode (Official pLDDT colors)
    # >90: Dark Blue (#0053d6), 70-90: Light Blue (#65cbf3), 50-70: Yellow (#ffdb13), <50: Orange (#ff7d45)
    plddt = float(value)
    if plddt >= 90.0:
        return "0x0053d6"
    if plddt >= 70.0:
        return "0x65cbf3"
    if plddt >= 50.0:
        return "0xffdb13"
    return "0xff7d45"


def build_interactive_3d_html(
    bundle: LockKeyAnalysisBundle,
    paint_mode: str = "Puzzle Shape Mode",
    show_ghost_mold: bool = True,
    show_partner_key: bool = True,
    highlight_residue: tuple[str, int] | None = None,
    width: int = 860,
    height: int = 560,
) -> str:
    """Generates a self-contained HTML snippet (or full standalone page) rendering the 3D Lock-and-Key
    molecular puzzle using 3Dmol.js with surface dots, Ghost Mold spheres, and Salt Bridge cylinders.
    """
    pdb_json = json.dumps(bundle.trimmed_pdb_str)
    lock_chain = bundle.metadata["lock_chain"]
    key_chain = bundle.metadata["key_chain"]

    # Prepare Lock surface skin dots colored by the active Paint Mode (subsampled to ~550 dots for 60fps WebGL)
    fp = bundle.lock_fingerprint
    pts = fp.surface.points
    n_pts = len(pts)
    stride = max(1, n_pts // 550)

    skin_spheres = []
    for i in range(0, n_pts, stride):
        p = pts[i]
        if paint_mode == "Puzzle Shape Mode":
            val = float(fp.surface.curvature[i])
        elif paint_mode == "Magnet Charge Mode":
            val = float(fp.charge[i])
        elif paint_mode == "Oil-vs-Water Mode":
            val = float(fp.hydropathy[i])
        else:
            val = float(fp.plddt[i])
        col_hex = map_value_to_paint_hex(val, paint_mode)
        in_pocket = i in set(bundle.selected_pocket.point_indices.tolist())
        rad = 0.72 if in_pocket else 0.45
        alpha = 0.88 if in_pocket else 0.45
        skin_spheres.append(
            {
                "x": round(float(p[0]), 2),
                "y": round(float(p[1]), 2),
                "z": round(float(p[2]), 2),
                "color": col_hex,
                "radius": rad,
                "alpha": alpha,
            }
        )

    # Prepare Ghost Key Mold glowing green translucent dots
    ghost_spheres = []
    if show_ghost_mold and len(bundle.ghost_mold.points) > 0:
        g_pts = bundle.ghost_mold.points
        g_stride = max(1, len(g_pts) // 140)
        for i in range(0, len(g_pts), g_stride):
            gp = g_pts[i]
            ghost_spheres.append(
                {
                    "x": round(float(gp[0]), 2),
                    "y": round(float(gp[1]), 2),
                    "z": round(float(gp[2]), 2),
                }
            )

    # Prepare yellow dashed interaction cylinders for Salt Bridges & H-bonds
    bridge_lines = []
    if show_partner_key:
        for c in bundle.scorecard.contacts[:12]:
            if c.interaction_type in ("Salt Bridge", "Hydrogen Bond / Polar", "Charge Repulsion"):
                l_mask = (bundle.lock_atoms.res_nums == c.lock_resnum) & (
                    np.array(bundle.lock_atoms.chain_ids) == c.lock_chain
                )
                k_mask = (bundle.key_atoms.res_nums == c.key_resnum) & (
                    np.array(bundle.key_atoms.chain_ids) == c.key_chain
                )
                if np.any(l_mask) and np.any(k_mask):
                    p1 = np.mean(bundle.lock_atoms.coords[l_mask], axis=0)
                    p2 = np.mean(bundle.key_atoms.coords[k_mask], axis=0)
                    color = (
                        "0xfacc15"
                        if c.interaction_type == "Salt Bridge"
                        else "0xef4444"
                        if c.interaction_type == "Charge Repulsion"
                        else "0x38bdf8"
                    )
                    bridge_lines.append(
                        {
                            "x1": round(float(p1[0]), 2),
                            "y1": round(float(p1[1]), 2),
                            "z1": round(float(p1[2]), 2),
                            "x2": round(float(p2[0]), 2),
                            "y2": round(float(p2[1]), 2),
                            "z2": round(float(p2[2]), 2),
                            "color": color,
                            "label": f"{c.lock_resname}{c.lock_resnum} ↔ {c.key_resname}{c.key_resnum}",
                        }
                    )

    pocket_center = [round(float(x), 2) for x in bundle.selected_pocket.center]
    skin_json = json.dumps(skin_spheres)
    ghost_json = json.dumps(ghost_spheres)
    bridges_json = json.dumps(bridge_lines)
    hl_json = json.dumps(
        {"chain": highlight_residue[0], "resi": int(highlight_residue[1])}
        if highlight_residue
        else None
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>LockKey-Explorer 3D Viewer — {bundle.metadata['title']}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.1.0/3Dmol-min.js"></script>
<style>
  body {{ margin: 0; padding: 0; background: #0f172a; font-family: system-ui, -apple-system, sans-serif; color: #f8fafc; }}
  #viewer-container {{ position: relative; width: 100%; height: {height}px; border-radius: 12px; overflow: hidden; border: 1px solid #334155; }}
  .hud-overlay {{
    position: absolute; top: 12px; left: 12px; z-index: 10;
    background: rgba(15, 23, 42, 0.82); backdrop-filter: blur(6px);
    padding: 8px 14px; border-radius: 8px; border: 1px solid #475569;
    font-size: 13px; line-height: 1.4; pointer-events: none;
  }}
  .hud-legend {{
    position: absolute; bottom: 12px; left: 12px; z-index: 10;
    background: rgba(15, 23, 42, 0.85); padding: 6px 12px; border-radius: 8px;
    border: 1px solid #475569; font-size: 12px; display: flex; gap: 14px; align-items: center;
  }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }}
</style>
</head>
<body>
<div id="viewer-container">
  <div class="hud-overlay">
    <strong>{bundle.metadata['short_name']}</strong><br/>
    <span>Active Keyhole: <b>{bundle.selected_pocket.name}</b> | Paint: <b>{paint_mode}</b></span>
  </div>
  <div class="hud-legend">
    <span><span class="dot" style="background:#a855f7;"></span>Lock Protein ({lock_chain})</span>
    {"<span><span class='dot' style='background:#f59e0b;'></span>Partner Key (" + key_chain + ")</span>" if show_partner_key else ""}
    {"<span><span class='dot' style='background:#22c55e;'></span>Ghost Key Mold (2.8 Å)</span>" if show_ghost_mold else ""}
    <span><span class="dot" style="background:#facc15;"></span>Salt Bridges</span>
  </div>
  <div id="mol-gl" style="width: 100%; height: 100%;"></div>
</div>
<script>
(function() {{
  const pdbData = {pdb_json};
  const skinDots = {skin_json};
  const ghostDots = {ghost_json};
  const bridges = {bridges_json};
  const highlightRes = {hl_json};
  const lockChain = "{lock_chain}";
  const keyChain = "{key_chain}";
  const showKey = {"true" if show_partner_key else "false"};
  const pocketCenter = {json.dumps(pocket_center)};

  const element = document.getElementById("mol-gl");
  const viewer = $3Dmol.createViewer(element, {{ backgroundColor: "#0f172a" }});
  viewer.addModel(pdbData, "pdb");

  if (lockChain !== keyChain) {{
    viewer.setStyle({{ chain: lockChain }}, {{ cartoon: {{ color: "#94a3b8", opacity: 0.85 }} }});
    if (showKey) {{
      viewer.setStyle({{ chain: keyChain }}, {{
        cartoon: {{ color: "#f59e0b", opacity: 0.92 }},
        stick: {{ radius: 0.14, colorscheme: "orangeCarbon" }}
      }});
    }} else {{
      viewer.setStyle({{ chain: keyChain }}, {{}});
    }}
  }} else {{
    // Single-chain enzyme (PETase): highlight residues 178-212 as the clamping lid
    viewer.setStyle({{ chain: lockChain }}, {{ cartoon: {{ color: "#94a3b8", opacity: 0.85 }} }});
    if (showKey) {{
      viewer.setStyle({{ chain: lockChain, resi: ["178-212"] }}, {{
        cartoon: {{ color: "#f59e0b", opacity: 0.95 }},
        stick: {{ radius: 0.16, colorscheme: "orangeCarbon" }}
      }});
    }}
  }}

  // Render painted 3D molecular skin dots
  for (let i = 0; i < skinDots.length; i++) {{
    const s = skinDots[i];
    viewer.addSphere({{
      center: {{ x: s.x, y: s.y, z: s.z }},
      radius: s.radius,
      color: parseInt(s.color, 16),
      alpha: s.alpha
    }});
  }}

  // Render glowing green Ghost Key Mold dots floating inside the pocket
  for (let i = 0; i < ghostDots.length; i++) {{
    const g = ghostDots[i];
    viewer.addSphere({{
      center: {{ x: g.x, y: g.y, z: g.z }},
      radius: 0.65,
      color: 0x22c55e,
      alpha: 0.65
    }});
  }}

  // Render yellow dashed cylinders for Salt Bridges & H-bonds
  for (let i = 0; i < bridges.length; i++) {{
    const b = bridges[i];
    viewer.addCylinder({{
      start: {{ x: b.x1, y: b.y1, z: b.z1 }},
      end: {{ x: b.x2, y: b.y2, z: b.z2 }},
      radius: 0.22,
      color: parseInt(b.color, 16),
      dashed: true,
      fromCap: 1,
      toCap: 1
    }});
    viewer.addLabel(b.label, {{
      position: {{ x: (b.x1 + b.x2) / 2.0, y: (b.y1 + b.y2) / 2.0, z: (b.z1 + b.z2) / 2.0 }},
      backgroundColor: "rgba(15, 23, 42, 0.85)",
      fontColor: "#fde047",
      fontSize: 11,
      borderRadius: 4
    }});
  }}

  // Highlight active What-If amino acid if selected
  if (highlightRes) {{
    viewer.addStyle(
      {{ chain: highlightRes.chain, resi: highlightRes.resi }},
      {{ stick: {{ radius: 0.35, color: "#ec4899" }}, sphere: {{ scale: 0.45, color: "#ec4899" }} }}
    );
  }}

  viewer.addSphere({{
    center: {{ x: pocketCenter[0], y: pocketCenter[1], z: pocketCenter[2] }},
    radius: 1.3,
    color: 0x38bdf8,
    alpha: 0.9
  }});

  viewer.zoomTo();
  viewer.render();
}})();
</script>
</body>
</html>"""
    return html
