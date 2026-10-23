"""Noise estimator and truth-independent discrepancy search checks."""
from pathlib import Path
import sys
import unittest

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from select_lambda import patch_statistics, search_discrepancy
from iterative import solve_l1
from deblur import forward_operator


class LambdaSelectionTests(unittest.TestCase):
    def test_variance_is_mean_invariant_and_target_counts_full_image(self):
        b=np.zeros((6,8)); b[1:3,2:4]=[[1,3],[5,7]]
        stats=patch_statistics(b,[2,4,1,3])
        self.assertAlmostEqual(stats['variance_uT2'],20/3)
        self.assertAlmostEqual(stats['target_squared_residual_uT2'],48*20/3)
        self.assertEqual(stats['patch_pixels'],4)
        shifted=patch_statistics(b+100,[2,4,1,3])
        self.assertAlmostEqual(stats['variance_uT2'],shifted['variance_uT2'])
        for roi in [[-1,3,0,2],[0,9,0,2],[2,1,0,2],[0,1,0,1]]:
            with self.assertRaises(ValueError): patch_statistics(b,roi)

    def test_identity_lasso_selects_known_lambda_using_real_solvers(self):
        b=np.full((7,8),2.); h=np.ones_like(b)
        sigma=.12; target=b.size*sigma**2
        for method in ['ista','fista']:
            def evaluate(lam):
                x,history=solve_l1(b,h,lam,method=method,pad=0,epsilon=1e-6)
                return dict(energy=float(np.sum((b-forward_operator(x,h,0))**2)),
                            converged=history[-1]['converged'])
            result=search_discrepancy(evaluate,target,2.,float(np.sum(b*b)),tolerance=.01)
            self.assertEqual(result['status'],'matched')
            self.assertLess(abs(result['selected']['lambda_value']/sigma-1),.006)

    def test_unattainable_target_does_not_select_lambda(self):
        def floor(lam): return dict(energy=3.+lam,converged=True)
        result=search_discrepancy(floor,1.,2.,5.,max_trials=5)
        self.assertEqual(result['status'],'target_not_matched')
        self.assertIsNone(result['selected'])
        result=search_discrepancy(floor,10.,2.,5.)
        self.assertEqual(result['status'],'target_above_maximum_residual')

    def test_nonconverged_inner_solution_is_rejected(self):
        def unfinished(lam): return dict(energy=1.,converged=False)
        result=search_discrepancy(unfinished,1.,2.,5.)
        self.assertEqual(result['status'],'inner_solver_not_converged')
        self.assertIsNone(result['selected'])

    def test_search_moves_down_from_initial_lambda(self):
        result=search_discrepancy(lambda lam:dict(energy=lam**2,converged=True),
                                 1e-12,1.,1.,tolerance=.01)
        self.assertEqual(result['status'],'matched')
        self.assertAlmostEqual(result['selected']['lambda_value'],1e-6)


if __name__=='__main__': unittest.main()
