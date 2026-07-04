# -*- coding: utf-8 -*-
"""Population-synthesis primitives."""

from jr_optlib.population.curvature import (
    entropic_projection,
    margin_information_matrix,
)
from jr_optlib.population.hard_ipf import ConstraintSpec, HardIPF
from jr_optlib.population.integerize import (
    GHOST_X_EPS,
    SWAP_MAX_MOVES_PER_SLICE,
    SWAP_MAX_PASSES,
    compute_zone_residuals,
    optimize_repair_zone,
    pps_without_replacement,
    swap_repair_zone,
    step1_split,
    step2_anchor_pps,
)

__all__ = [
    "ConstraintSpec",
    "GHOST_X_EPS",
    "HardIPF",
    "entropic_projection",
    "margin_information_matrix",
    "SWAP_MAX_MOVES_PER_SLICE",
    "SWAP_MAX_PASSES",
    "compute_zone_residuals",
    "optimize_repair_zone",
    "pps_without_replacement",
    "swap_repair_zone",
    "step1_split",
    "step2_anchor_pps",
]
