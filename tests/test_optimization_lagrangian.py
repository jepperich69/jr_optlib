# -*- coding: utf-8 -*-
import sys
import numpy as np
from pathlib import Path

JR = Path(r"C:\Users\rich\OneDrive - Danmarks Tekniske Universitet\JR")
OLD_CODE_DIR = JR / "Publikationer" / "Pub_SAA_PMIP_MC" / "code"
sys.path.insert(0, str(OLD_CODE_DIR))

from logsum_toy import dual_ascent_trace_2x2 as old_dual
from jr_optlib.optimization.lagrangian import subgradient_dual_ascent

def test_dual_ascent_matches_2x2_trace():
    # Run the original trace
    old_records = old_dual.run_trace()
    old_final = old_records[-1]
    
    # Run the new generic driver
    def inner_solve(lam):
        chat = old_dual.loaded_cost(lam)
        return old_dual.sinkhorn(chat)
        
    def gap_fn(x):
        _, _, ev = old_dual.row_expected_violations(x)
        return ev - old_dual.KAPPA
        
    init_lam = np.zeros(2)
    final_lam, final_x, final_gap, feasible, iters = subgradient_dual_ascent(
        init_lam=init_lam,
        inner_solve_fn=inner_solve,
        gap_fn=gap_fn,
        alpha=old_dual.ALPHA,
        max_iter=old_dual.MAX_ITER,
        tol=old_dual.TOL,
        lam_min=0.0
    )
    
    # Assert exact match at the final step
    assert feasible == old_final["feasible"]
    assert iters == old_final["k"]
    np.testing.assert_allclose(final_lam, old_final["lambda"], atol=1e-12)
    np.testing.assert_allclose(final_x, old_final["x"], atol=1e-12)
    np.testing.assert_allclose(final_gap, old_final["gap"], atol=1e-12)
