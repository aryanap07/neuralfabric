"""Library-specific exception types."""


class LevelGraphsError(Exception):
    """Base class for all NeuralFabric exceptions."""


class NotFittedError(LevelGraphsError):
    """Raised when a method needs a fitted estimator that hasn't been fit yet."""


class ShapeMismatchError(LevelGraphsError):
    """Raised when input shapes are incompatible with the requested operation."""
