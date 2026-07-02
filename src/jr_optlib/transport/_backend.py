# -*- coding: utf-8 -*-
"""LP backend detection, ported from the paper code.

Prefers OR-Tools (GLOP), falls back to PuLP (CBC). Same precedence as
Pub_MIPEntropy_MPC so migrated functions behave identically.
"""

_BACKEND = None
_HAS_GRAPH = False
try:
    from ortools.linear_solver import pywraplp  # noqa: F401
    _BACKEND = "ortools"
    try:
        from ortools.graph import pywrapgraph  # noqa: F401
        _HAS_GRAPH = True
    except Exception:
        _HAS_GRAPH = False
except Exception:
    try:
        import pulp  # noqa: F401
        _BACKEND = "pulp"
    except Exception:
        _BACKEND = None


def backend_name():
    return _BACKEND
