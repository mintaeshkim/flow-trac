import sys
from types import ModuleType

from trac.agent import TRACAgent, TRACConfig

__all__ = ["TRACAgent", "TRACConfig"]


def _register_legacy_checkpoint_modules() -> None:
    """Keep checkpoints made before the package was flattened loadable."""
    legacy_agents = ModuleType("trac.agents")
    legacy_agents.__path__ = []
    legacy_trac = ModuleType("trac.agents.trac")
    legacy_trac.__path__ = []

    legacy_agents.trac = legacy_trac
    legacy_trac.agent = sys.modules["trac.agent"]
    sys.modules.setdefault("trac.agents", legacy_agents)
    sys.modules.setdefault("trac.agents.trac", legacy_trac)
    sys.modules.setdefault("trac.agents.trac.agent", sys.modules["trac.agent"])


_register_legacy_checkpoint_modules()
