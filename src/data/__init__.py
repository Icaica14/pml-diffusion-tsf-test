"""Data layer: loaders, temporal splitting, train-only scaling, windowing, manifest.

The single entry point is the *data contract* (`ForecastDataset`): one loader
returns the same object to every model so the comparison is fair by construction
(plan §4.5).
"""

from .contract import ForecastDataset, temporal_split, make_windows
from .scaling import Scaler

__all__ = ["ForecastDataset", "temporal_split", "make_windows", "Scaler"]
