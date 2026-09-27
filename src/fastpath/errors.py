class FastPathError(Exception):
    """Base error for the package."""


class ContractError(FastPathError):
    """Raised when a decision schema cannot be compiled."""


class ArtifactError(FastPathError):
    """Raised when a model artifact is invalid or incompatible."""
