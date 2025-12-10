"""
DeepSeek-Style Fused MoE Kernel Implementation

This package implements a fused Grouped GEMM kernel that combines
Routed Expert and Shared Expert computations into a single kernel launch,
reducing memory bandwidth by loading input activations only once.

Key Components:
- utils.py: Routing metadata computation and grid configuration
- kernels.py: Triton fused forward kernel with virtual gathering
- moe_layer.py: PyTorch module and autograd integration
- benchmark.py: Correctness verification and performance testing
"""

from .moe_layer import FusedDeepSeekMoE, DeepSeekMoELayer
from .utils import compute_routing_metadata, get_grid_config

__all__ = [
    'FusedDeepSeekMoE',
    'DeepSeekMoELayer', 
    'compute_routing_metadata',
    'get_grid_config',
]
