# 🧩 LockKey-Explorer: 3D Protein "Lock-and-Key" Puzzle Workbench

**High School Science Fair Edition** — An interactive 3D molecular puzzle web app (Installable PWA + Python package) and **Model Context Protocol (`FastMCP`)** server for exploring how everyday biology proteins recognize and bind one another.

---

## 🌟 Classroom Mission & Biosecurity Guardrails

`LockKey-Explorer` lets students and science fair judges explore real 3D structures from the [Protein Data Bank (PDB)](https://www.rcsb.org/) and [EBI AlphaFold Database](https://alphafold.ebi.ac.uk/) using three intuitive rules:
1. **3D Shape Lock ($\kappa_i \in [-1, +1]$)**: Deep purple concave keyholes ("Valleys") match golden convex protein loops ("Hills").
2. **Magnetic Charge Attraction ($q_i \in [-1, +1]$)**: Positive ($+$) amino acids (`ARG`, `LYS`) attract Negative ($-$) amino acids (`ASP`, `GLU`) to form **Salt Bridges**.
3. **Oil-vs-Water Stickiness ($h_i \in [-1, +1]$)**: Greasy hydrophobic patches (`ILE`, `VAL`, `LEU`, `PHE`, `TRP`) stick together to hide from water.

### 🛡️ 100% Classroom-Safe Allowlist
- Strictly limited to **4 safe everyday biology protein puzzles**:
  1. **The Jellyfish Glow Switch (`PDB: 3OGO`)** — Green Fluorescent Protein (GFP) + Enhancer helper protein.
  2. **Egg-White Defense (`PDB: 1MEL` / `1BVK` / `1FDL`)** — Hen Egg-White Lysozyme + Camelid VHH partner.
  3. **Recycling Plastic Bottles (`PDB: 6EQE` / `AF-A0A0K8P6T7`)** — *Ideonella sakaiensis* PETase plastic-eating enzyme.
  4. **Carrying Oxygen in Blood (`PDB: 1A3N`)** — Human Hemoglobin $\alpha$ and $\beta$ subunits.
- **Zero Pathogens, Toxins, or De Novo Sequence Design**: Unverified IDs are blocked automatically, and the educational **"What-If" Explorer** only toggles a single existing interface residue to demonstrate attraction vs. repulsion.
- **100% Offline & Laptop CPU Ready**: Runs in $<1.5\text{ seconds}$ on any school laptop (Windows, Mac, Linux, Chromebook) with pre-cached structures.

---

## 🚀 Quick Start (Installable Web App)

### 1. Install & Pre-Fetch Offline Classroom Proteins
```bash
# Using uv (recommended) or pip
uv sync
uv run lockkey-download

# Or with standard pip:
pip install -e .
lockkey-download
```

### 2. Launch the Interactive 3D Web App
```bash
uv run lockkey-explorer
# Or directly via Streamlit:
uv run streamlit run app.py
```
Open **`http://localhost:8501`** in Chrome, Edge, or Safari. Because `LockKey-Explorer` includes a Web App Manifest (`static/manifest.json`), you can click **"Install LockKey-Explorer"** in your browser's address bar to install it as a standalone desktop/tablet Science Fair app!

### 3. Run the 3 Classroom Demo Scripts & Generate Standalone 3D HTML Models
```bash
uv run python examples/demo_1_gfp_glow_switch.py
uv run python examples/demo_2_egg_lysozyme.py
uv run python examples/demo_3_petase_plastic.py
```
Standalone interactive 3D WebGL `.html` files are saved to `examples/output/`.

### 4. Run the Automated Test Suite
```bash
uv run pytest -v
```

---

## 🤖 FastMCP Server (`mcp_server.py`)

Expose the 4 educational tools (`load_classroom_protein`, `find_surface_pockets`, `match_partner_key`, `simulate_what_if_swap`) to Claude Code or Google Antigravity:

```bash
uv run lockkey-mcp
# Or:
uv run python mcp_server.py
```
