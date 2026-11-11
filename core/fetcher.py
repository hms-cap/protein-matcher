"""Module 1: Safe Structure Ingestion & AlphaFold Confidence Filter (core/fetcher.py)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np

from core import (
    CHARGED_ATOM_MAP,
    KYTE_DOOLITTLE_NORM,
    STANDARD_AMINO_ACIDS,
    VDW_RADII,
)
from data.download_examples import (
    STRUCTURES_DIR,
    download_alphafold,
    download_pdb,
    load_allowlist,
)


class BiosecurityBlockError(PermissionError):
    """Raised when a user or agent requests a protein outside the safe classroom allowlist."""


@dataclass
class SimpleAtom:
    """Lightweight PDB atom record (100% pure Python/NumPy, compatible with Bio.PDB API)."""

    name: str
    element: str
    coord: np.ndarray
    bfactor: float
    occupancy: float
    serial_number: int
    raw_line: str

    def get_name(self) -> str:
        return self.name

    def get_bfactor(self) -> float:
        return self.bfactor

    def set_bfactor(self, value: float) -> None:
        self.bfactor = float(value)


@dataclass
class SimpleResidue:
    """Lightweight PDB residue record."""

    id: tuple[str, int, str]
    resname: str
    atoms: list[SimpleAtom] = field(default_factory=list)

    def get_resname(self) -> str:
        return self.resname

    def get_atoms(self) -> list[SimpleAtom]:
        return self.atoms

    def __iter__(self):
        return iter(self.atoms)


@dataclass
class SimpleChain:
    """Lightweight PDB chain record."""

    id: str
    residues: list[SimpleResidue] = field(default_factory=list)

    def detach_child(self, res_id: tuple[str, int, str]) -> None:
        self.residues = [r for r in self.residues if r.id != res_id]

    def __iter__(self):
        return iter(self.residues)


@dataclass
class SimpleModel:
    """Lightweight PDB model record."""

    id: int = 0
    chains: list[SimpleChain] = field(default_factory=list)

    def __iter__(self):
        return iter(self.chains)


@dataclass
class SimpleStructure:
    """Lightweight PDB structure record."""

    id: str
    models: list[SimpleModel] = field(default_factory=list)

    def get_atoms(self):
        for m in self.models:
            for c in m.chains:
                for r in c.residues:
                    for a in r.atoms:
                        yield a

    def __iter__(self):
        return iter(self.models)


@dataclass
class AtomArrayData:
    """Vectorized atomic representation for fast laptop CPU geometry & electrostatics."""

    coords: np.ndarray          # (M, 3) float64
    radii: np.ndarray           # (M,) float64
    charges: np.ndarray         # (M,) float64
    hydropathy: np.ndarray      # (M,) float64
    bfactors: np.ndarray        # (M,) float64 (stores pLDDT or crystallographic B-factor)
    res_nums: np.ndarray        # (M,) int32
    res_names: list[str]        # len M
    atom_names: list[str]       # len M
    chain_ids: list[str]        # len M


def validate_safe_id(preset_or_id: str) -> tuple[str, dict[str, Any]]:
    """Checks that the requested protein is on the strict high-school classroom allowlist."""
    allowlist = load_allowlist()
    presets = allowlist["classroom_presets"]
    clean = preset_or_id.strip().upper()

    if clean.endswith(".PDB"):
        stem = Path(preset_or_id).stem.upper()
        if stem in presets:
            return stem, presets[stem]

    if clean == "A0A0K8P6T7":
        clean = "AF-A0A0K8P6T7"
    elif clean == "PETASE":
        clean = "6EQE"
    elif clean == "GFP":
        clean = "3OGO"
    elif clean == "LYSOZYME":
        clean = "1MEL"
    elif clean == "HEMOGLOBIN":
        clean = "1A3N"

    if clean not in presets:
        allowed_str = ", ".join(presets.keys())
        raise BiosecurityBlockError(
            f"Safety Guardrail Blocked '{preset_or_id}': LockKey-Explorer only permits verified "
            f"classroom-safe proteins ({allowed_str}). Pathogens, toxins, and unverified IDs are prohibited."
        )
    return clean, presets[clean]


def fetch_structure_file(preset_or_id: str) -> tuple[Path, dict[str, Any]]:
    """Returns a local Path to the approved PDB file, downloading if needed."""
    canonical_id, meta = validate_safe_id(preset_or_id)
    STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)
    pdb_path = STRUCTURES_DIR / f"{canonical_id}.pdb"

    if not pdb_path.exists() or pdb_path.stat().st_size < 500:
        if meta["source"] == "rcsb":
            ok = download_pdb(canonical_id, pdb_path)
        else:
            ok = download_alphafold(
                meta.get("uniprot_id", "A0A0K8P6T7"),
                pdb_path,
                fallback_pdb=meta.get("fallback_pdb", "6EQE"),
            )
        if not ok:
            raise RuntimeError(f"Could not fetch classroom structure {canonical_id}.")

    return pdb_path, meta


def load_biopython_structure(pdb_path: str | Path, structure_id: str = "protein") -> SimpleStructure:
    """Parses the first MODEL of a PDB coordinate file into a clean structure object."""
    text = Path(pdb_path).read_text(encoding="utf-8", errors="ignore")
    chains_map: dict[str, SimpleChain] = {}
    res_map: dict[tuple[str, tuple[str, int, str]], SimpleResidue] = {}
    seen_atom_in_res: set[tuple[str, tuple[str, int, str], str]] = set()

    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM  "):
            continue
        altloc = line[16:17]
        if altloc not in (" ", "A", "1"):
            continue

        atom_name = line[12:16].strip()
        resname = line[17:20].strip().upper()
        chain_id = line[21:22].strip() or "A"
        try:
            resseq = int(line[22:26].strip())
            icode = line[26:27].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
        except ValueError:
            continue

        try:
            occ = float(line[54:60].strip())
        except ValueError:
            occ = 1.0
        try:
            bfac = float(line[60:66].strip())
        except ValueError:
            bfac = 85.0
        try:
            serial = int(line[6:11].strip())
        except ValueError:
            serial = 1

        elem = line[76:78].strip().upper() if len(line) >= 78 else ""
        if not elem:
            elem = "".join(ch for ch in atom_name if ch.isalpha())[:1].upper() or "C"

        # Normalize 3OGO biological assembly (GFP Chain B + Enhancer Chain G -> Chain A + Chain B,
        # and shift Enhancer Arg36 author numbering by -1 so Key Arg35 matches the classroom specification)
        if structure_id.upper() == "3OGO":
            if chain_id == "B":
                chain_id = "A"
            elif chain_id == "G":
                chain_id = "B"
                resseq = resseq - 1
            else:
                continue

        res_id = (" ", resseq, icode or " ")
        dedup_key = (chain_id, res_id, atom_name)
        if dedup_key in seen_atom_in_res:
            continue
        seen_atom_in_res.add(dedup_key)

        if chain_id not in chains_map:
            chains_map[chain_id] = SimpleChain(id=chain_id)

        r_key = (chain_id, res_id)
        if r_key not in res_map:
            res_obj = SimpleResidue(id=res_id, resname=resname)
            res_map[r_key] = res_obj
            chains_map[chain_id].residues.append(res_obj)

        res_map[r_key].atoms.append(
            SimpleAtom(
                name=atom_name,
                element=elem,
                coord=np.array([x, y, z], dtype=np.float64),
                bfactor=bfac,
                occupancy=occ,
                serial_number=serial,
                raw_line=line,
            )
        )

    model = SimpleModel(id=0, chains=list(chains_map.values()))
    return SimpleStructure(id=structure_id, models=[model])


def normalize_plddt_values(structure: SimpleStructure) -> bool:
    """Detects whether B-factors represent AlphaFold pLDDT (0-100) or X-ray B-factors.
    For X-ray structures where flexible loops have high B-factors (>40), synthesizes an
    educational pseudo-pLDDT confidence score so the classroom pLDDT slider works seamlessly on all presets!
    """
    bfactors = [atom.get_bfactor() for atom in structure.get_atoms()]
    if not bfactors:
        return False
    arr = np.array(bfactors, dtype=float)
    is_alphafold = bool(np.median(arr) > 68.0 and np.max(arr) <= 100.0)

    if not is_alphafold:
        for model in structure:
            for chain in model:
                residues = [r for r in chain if r.id[0] == " "]
                n_res = len(residues)
                for idx, res in enumerate(residues):
                    res_b = np.mean([a.get_bfactor() for a in res.get_atoms()])
                    is_terminal = idx < 4 or idx >= n_res - 4
                    score = float(np.clip(102.0 - 0.85 * res_b, 35.0, 98.0))
                    if is_terminal:
                        score = min(score, 56.0)
                    for atom in res.get_atoms():
                        atom.set_bfactor(score)
    return is_alphafold


def trim_low_confidence_tails(
    structure: SimpleStructure,
    min_plddt: float = 65.0,
) -> tuple[SimpleStructure, dict[str, Any]]:
    """Trims residues whose mean confidence (pLDDT stored in B-factor) is below min_plddt.
    Returns the trimmed structure and statistics about removed residues.
    """
    normalize_plddt_values(structure)
    total_residues = 0
    kept_residues = 0
    trimmed_list: list[str] = []

    for model in structure:
        for chain in model:
            to_detach = []
            for res in list(chain):
                if res.id[0] != " ":
                    to_detach.append(res.id)
                    continue
                resname = res.get_resname().strip()
                if resname not in STANDARD_AMINO_ACIDS:
                    to_detach.append(res.id)
                    continue
                total_residues += 1
                atoms = list(res.get_atoms())
                mean_plddt = float(np.mean([a.get_bfactor() for a in atoms])) if atoms else 0.0
                if mean_plddt < min_plddt:
                    to_detach.append(res.id)
                    trimmed_list.append(f"{chain.id}:{resname}{res.id[1]} ({mean_plddt:.1f})")
                else:
                    kept_residues += 1
            for res_id in to_detach:
                chain.detach_child(res_id)

    stats = {
        "min_plddt": float(min_plddt),
        "total_residues_before": total_residues,
        "kept_residues": kept_residues,
        "trimmed_residues_count": total_residues - kept_residues,
        "trimmed_examples": trimmed_list[:10],
    }
    return structure, stats


def extract_atom_data(
    structure: SimpleStructure,
    chain_id: str | None = None,
    residue_filter: tuple[int, int] | None = None,
) -> AtomArrayData:
    """Extracts vectorized atomic arrays for a specific chain (or virtual sub-domain)."""
    coords = []
    radii = []
    charges = []
    hydropathy = []
    bfactors = []
    res_nums = []
    res_names = []
    atom_names = []
    chain_ids = []

    model = next(iter(structure))
    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if res.id[0] != " ":
                continue
            resname = res.get_resname().strip().upper()
            if resname not in STANDARD_AMINO_ACIDS:
                continue
            resnum = int(res.id[1])
            if residue_filter is not None and not (residue_filter[0] <= resnum <= residue_filter[1]):
                continue

            kd_val = KYTE_DOOLITTLE_NORM.get(resname, 0.0)
            res_charge_map = CHARGED_ATOM_MAP.get(resname, {})

            for atom in res:
                elem = (atom.element or atom.get_name()[0]).strip().upper()
                if elem == "H":
                    continue
                aname = atom.get_name().strip().upper()
                coords.append(atom.coord)
                radii.append(VDW_RADII.get(elem, 1.70))
                charges.append(res_charge_map.get(aname, 0.0))
                hydropathy.append(kd_val)
                bfactors.append(float(atom.get_bfactor()))
                res_nums.append(resnum)
                res_names.append(resname)
                atom_names.append(aname)
                chain_ids.append(chain.id)

    if not coords:
        return AtomArrayData(
            coords=np.zeros((0, 3), dtype=np.float64),
            radii=np.zeros(0, dtype=np.float64),
            charges=np.zeros(0, dtype=np.float64),
            hydropathy=np.zeros(0, dtype=np.float64),
            bfactors=np.zeros(0, dtype=np.float64),
            res_nums=np.zeros(0, dtype=np.int32),
            res_names=[],
            atom_names=[],
            chain_ids=[],
        )

    return AtomArrayData(
        coords=np.asarray(coords, dtype=np.float64),
        radii=np.asarray(radii, dtype=np.float64),
        charges=np.asarray(charges, dtype=np.float64),
        hydropathy=np.asarray(hydropathy, dtype=np.float64),
        bfactors=np.asarray(bfactors, dtype=np.float64),
        res_nums=np.asarray(res_nums, dtype=np.int32),
        res_names=res_names,
        atom_names=atom_names,
        chain_ids=chain_ids,
    )


def structure_to_pdb_string(structure: SimpleStructure, chains: list[str] | None = None) -> str:
    """Serializes a structure (or selected chains) back to standard PDB format."""
    lines: list[str] = []
    serial = 1
    for model in structure:
        for chain in model:
            if chains is not None and chain.id not in chains:
                continue
            for res in chain:
                if res.id[0] != " ":
                    continue
                resseq = res.id[1]
                resname = res.resname
                for atom in res:
                    x, y, z = atom.coord
                    aname = f" {atom.name:<3s}" if len(atom.name) < 4 else atom.name[:4]
                    line = (
                        f"ATOM  {serial:5d} {aname:4s} {resname:>3s} {chain.id:1s}"
                        f"{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                        f"{atom.occupancy:6.2f}{atom.bfactor:6.2f}          {atom.element:>2s}"
                    )
                    lines.append(line)
                    serial += 1
            lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"
