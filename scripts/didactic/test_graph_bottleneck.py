"""Checks for the nine-state graph diagnostic."""
import unittest
import numpy as np
import graph_bottleneck_td as gb


class GraphTests(unittest.TestCase):
    def test_kernel_rows_are_distributions_at_every_leak(self):
        for leak in (0, .05, .4, 1.0):
            p = gb.build_graph(leak)
            self.assertEqual(p.shape, (9, 2, 9))
            np.testing.assert_allclose(p.sum(-1), 1.0, atol=1e-12)
            self.assertTrue((p >= 0).all())

    def test_state_four_is_the_only_crossing_when_there_is_no_leak(self):
        p = gb.build_graph(0.0)
        reach = {s: set(np.flatnonzero(p[s].sum(0))) for s in range(9)}
        crossers = [s for s in gb.GROUP_A if reach[s] & set(gb.GROUP_B + [gb.CUT])]
        self.assertEqual(crossers, [3])            # only state 3 reaches the cut
        self.assertTrue(reach[gb.CUT] & set(gb.GROUP_B))
        for s in gb.GROUP_B:                        # the second group does not go back
            self.assertFalse(reach[s] & set(gb.GROUP_A + [gb.CUT]))

    def test_leak_opens_direct_crossings_that_bypass_state_four(self):
        p = gb.build_graph(0.2)
        direct = [s for s in gb.GROUP_A if s != 3 and
                  set(np.flatnonzero(p[s].sum(0))) & set(gb.GROUP_B)]
        self.assertTrue(direct, "leak did not create a bypass")
        back = [s for s in gb.GROUP_B if set(np.flatnonzero(p[s].sum(0))) & set(gb.GROUP_A)]
        self.assertTrue(back, "leak did not let the second group return")

    def test_the_graph_is_not_a_chain(self):
        """Every state has two distinct successors and some state is reached twice."""
        p = gb.build_graph(0.0)
        succ = p.argmax(-1)
        for s in range(9):
            self.assertNotEqual(succ[s, 0], succ[s, 1], f"state {s} has one successor")
        counts = np.bincount(succ.ravel(), minlength=9)
        self.assertTrue((counts >= 2).sum() >= 5, "graph is too path-like")

    def test_the_decision_at_state_seven_is_real_and_favours_the_goal(self):
        p = gb.build_graph(0.0)
        P = bf_exact = __import__("bottleneck_family_td").exact(
            p.reshape(-1, 9), np.full((9, 2), 0.5), 0.8)
        gap = P[2 * gb.DECISION, gb.GOAL] - P[2 * gb.DECISION + 1, gb.GOAL]
        self.assertGreater(gap, 0.1, "action 0 at state 7 should reach the goal sooner")


if __name__ == "__main__":
    unittest.main()
