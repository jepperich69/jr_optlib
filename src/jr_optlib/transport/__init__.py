"""Transport / population-synthesis primitives."""

from jr_optlib.transport.ipf import ipf_2d, make_contingency2d, Contingency2D
from jr_optlib.transport.sinkhorn import (
    sinkhorn_balanced,
    sinkhorn_balanced_uv,
    make_transport,
    TransportInstance,
)

__all__ = [
    "ipf_2d", "make_contingency2d", "Contingency2D",
    "sinkhorn_balanced", "sinkhorn_balanced_uv",
    "make_transport", "TransportInstance",
]
