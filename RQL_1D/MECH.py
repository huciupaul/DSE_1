# src/MECH.py
from pathlib import Path

# Path of *this* file → parent → sibling folder 'mechanisms'
_MECH_DIR = (Path(__file__).resolve().parent / "mechanisms").resolve()

# Absolute path string; Cantera likes native str not Path
MECH = str(_MECH_DIR / "CapursoMechanism.yaml")