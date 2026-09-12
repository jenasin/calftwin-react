"""CalfTwin-ReAct: a state-coupled agent testbed for dairy-calf digital twins.

All data produced by this package are synthetic. See README.md, "Validation
status", before drawing any conclusion about a living animal.
"""
__version__ = "0.2.0"

from .config import CalfParams, CostWeights, RunConfig          # noqa: F401
from .runner import AGENTS, run_episode, run_grid               # noqa: F401
from .scenarios import SCENARIOS                                # noqa: F401
