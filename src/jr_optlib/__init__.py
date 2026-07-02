"""jr_optlib -- vetted, oracle-tested optimization primitives for JR research.

One library, one source of truth. If a second paper could plausibly use a
function, it lives here (vetted + tested + documented + indexed); paper-specific
glue stays in the paper repo.

Reproducibility rule: papers import live during development, then pin the
commit hash and vendor a snapshot into the reproduction package at submission.
See README.md and registry/SCHEMA.md.
"""

__version__ = "0.1.0"

from jr_optlib.transport.ipf import ipf_2d, make_contingency2d
from jr_optlib.transport.sinkhorn import sinkhorn_balanced, make_transport
from jr_optlib.setcover.entropy import solve_entropy_setcover

__all__ = ["ipf_2d", "make_contingency2d",
           "sinkhorn_balanced", "make_transport",
           "solve_entropy_setcover", "__version__"]
