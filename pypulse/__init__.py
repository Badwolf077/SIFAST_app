"""PyPulse - A Python package for spatiotemporal pulse characterization."""

from . import io
from .config.settings import ProcessingConfig
from .fiber.registry import register_fiber_array
from .processing.sifast import SIFAST
from .processing.spatial_scan import merge_spatial_scans
from .processing.srsi import SRSI

__all__ = [
    "SRSI",
    "SIFAST",
    "ProcessingConfig",
    "register_fiber_array",
    "io",
    "merge_spatial_scans",
]
__version__ = "0.1.2"
__author__ = "Xu Yilin"
__email__ = "xuyilin@siom.ac.cn"
