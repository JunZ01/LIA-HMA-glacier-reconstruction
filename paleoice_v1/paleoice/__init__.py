"""
PaleoIce: LIA冰川冰厚与ELA重建工具包
=====================================

基于OGGM框架，用于批量重建古冰川(LIA)的:
- 冰厚分布
- ELA (平衡线高度)
- 冰川体积

作者: Jun (Yunnan University)
"""

__version__ = "0.1.0"

from .prepare import prepare_lia_for_oggm, validate_inputs
from .flowline import init_glacier_dirs, compute_flowlines
# from .thickness import calculate_thickness  # Step 3
# from .ela import calculate_ela  # Step 4
