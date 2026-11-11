"""Core biophysical engine for LockKey-Explorer."""

from __future__ import annotations

# Standard Van der Waals atomic radii in Angstroms (Å)
VDW_RADII: dict[str, float] = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "FE": 1.40,
}

# Kyte-Doolittle Hydropathy Scale normalized to [-1.0, +1.0] (divided by 4.5)
# Positive (+) = Greasy / Hydrophobic ("Oil drops stick together")
# Negative (-) = Water-Loving / Polar
KYTE_DOOLITTLE_NORM: dict[str, float] = {
    "ILE": 4.5 / 4.5,    # +1.00
    "VAL": 4.2 / 4.5,    # +0.93
    "LEU": 3.8 / 4.5,    # +0.84
    "PHE": 2.8 / 4.5,    # +0.62
    "CYS": 2.5 / 4.5,    # +0.56
    "MET": 1.9 / 4.5,    # +0.42
    "ALA": 1.8 / 4.5,    # +0.40
    "GLY": -0.4 / 4.5,   # -0.09
    "THR": -0.7 / 4.5,   # -0.16
    "SER": -0.8 / 4.5,   # -0.18
    "TRP": -0.9 / 4.5,   # -0.20
    "TYR": -1.3 / 4.5,   # -0.29
    "PRO": -1.6 / 4.5,   # -0.36
    "HIS": -3.2 / 4.5,   # -0.71
    "GLU": -3.5 / 4.5,   # -0.78
    "GLN": -3.5 / 4.5,   # -0.78
    "ASP": -3.5 / 4.5,   # -0.78
    "ASN": -3.5 / 4.5,   # -0.78
    "LYS": -3.9 / 4.5,   # -0.87
    "ARG": -4.5 / 4.5,   # -1.00
}

# Amino acid net charge at neutral pH 7.4
RESIDUE_CHARGE_PH74: dict[str, float] = {
    "ARG": +1.0,
    "LYS": +1.0,
    "HIS": +0.2,
    "ASP": -1.0,
    "GLU": -1.0,
}

# Charged side-chain atom names where formal charge is localized
CHARGED_ATOM_MAP: dict[str, dict[str, float]] = {
    "ARG": {"NH1": 0.5, "NH2": 0.5, "CZ": 0.2},
    "LYS": {"NZ": 1.0},
    "HIS": {"ND1": 0.1, "NE2": 0.1},
    "ASP": {"OD1": -0.5, "OD2": -0.5, "CG": -0.2},
    "GLU": {"OE1": -0.5, "OE2": -0.5, "CD": -0.2},
}

STANDARD_AMINO_ACIDS = set(KYTE_DOOLITTLE_NORM.keys())
