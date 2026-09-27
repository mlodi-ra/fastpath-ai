"""Public FastPath API."""

from .contracts import Choice, Noul, Score
from .engine import FastPathEngine
from .errors import ArtifactError, ContractError, FastPathError

__all__ = [
    "ArtifactError",
    "Choice",
    "ContractError",
    "FastPathEngine",
    "FastPathError",
    "Noul",
    "Score",
]
__version__ = "0.1.0a2"
