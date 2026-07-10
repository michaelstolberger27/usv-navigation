from . import _compat  # noqa: F401  (normalizes the dependency's CPA convention)
from .automaton import ColavAutomaton
from .controllers import HeadingControlProvider
from .sync_runtime import SyncColavRuntime, StepResult

__version__ = "0.1.0"

__all__ = [
    "ColavAutomaton",
    "HeadingControlProvider",
    "SyncColavRuntime",
    "StepResult",
]
