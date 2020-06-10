""" Defines which columns go into which models """
from typing import Dict, Any
import os
from views.utils import io
from . import solver

_THIS_DIR = os.path.dirname(__file__)

cm: Dict[Any, Any] = solver.solve_formulas(
    io.load_yaml(os.path.join(_THIS_DIR, "cm.yaml"))
)
pgm: Dict[Any, Any] = solver.solve_formulas(
    io.load_yaml(os.path.join(_THIS_DIR, "pgm.yaml"))
)

__all__ = ["cm", "pgm"]
