"""Cross-cutting utilities: seeding, config loading, (later) timing & logging."""

from .config import load_config
from .seeds import set_seed

__all__ = ["load_config", "set_seed"]
