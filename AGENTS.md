# AGENTS.md — High School Science Fair Edition: 3D Protein "Lock-and-Key" Puzzle Explorer (`LockKey-Explorer`)

## 1. Project Mission & Biosecurity-Safe Educational Scope
Build **`LockKey-Explorer`**, an interactive, high-school-friendly 3D molecular puzzle application and Model Context Protocol (MCP) server. Given a safe, everyday protein structure ("The Lock") from the [EBI AlphaFold Database](https://alphafold.ebi.ac.uk/) or [Protein Data Bank (PDB)](https://www.rcsb.org/), the software discovers its surface binding pockets and **searches** a curated library of natural partner proteins ("The Keys") to see how their 3D shapes, electrical charges, and water-loving/water-hating patches fit together.

### Strict Biosecurity & Educational Safety Guardrails (Mandatory)
1. **100% Safe Classroom Proteins Only**: All examples, presets, and tests are strictly limited to safe, standard high-school biology proteins:
   - **Egg-White Lysozyme (`PDB: 1BVK` / `1FDL` / `1MEL`)** — the classic tear and egg-white enzyme studied in AP/IB Biology.
   - **Glowing Green Fluorescent Protein (`GFP`, `PDB: 3OGO`)** — the Nobel Prize-winning jellyfish glowing protein coupled with its enhancer helper protein.
   - **Plastic-Recycling Enzyme (`PETase`, `PDB: 6EQE` / `AF-A0A0K8P6T7`)** — an environmental bacteria enzyme that breaks down recyclable plastic bottles.
   - **Human Hemoglobin (`PDB: 1A3N`)** — the oxygen-carrying protein in red blood cells ($\alpha$ and $\beta$ subunit coupling).
2. **Zero Pathogens, Toxins, or De Novo Sequence Generation**: Do **not** include any viral proteins, bacterial toxins, or clinical pathogen targets. Do **not** generate or mutate new amino acid sequences or call external sequence-design models. Instead of *de novo* sequence invention, include an educational **"What-If Single Amino Acid Explorer"** that lets students toggle a single existing interface amino acid (e.g., swapping a `+` charge to a `-` charge) to visually learn how opposite charges attract and like charges repel.
3. **100% Laptop CPU & Offline-Ready**: Runs on any school or home laptop (Windows, Mac, Chromebook Linux, or Raspberry Pi) using standard Python libraries (`numpy`, `scipy`, `biopython`, `streamlit`, `py3Dmol`, `plotly`) with zero GPU or paid API keys required.
