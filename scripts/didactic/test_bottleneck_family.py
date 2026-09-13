"""Independent checks for the bottleneck-family diagnostic."""
import unittest

import numpy as np
import jax.numpy as jnp

import bottleneck_family_td as bf


class FamilyTests(unittest.TestCase):
    def test_build_matches_the_hand_written_five_state_graph_off_the_goal(self):
        """build(2, 2) is deterministic_td.py's bottleneck5 except at the goal,
        where this family uses stay/retreat rather than retreat/stay. No
        diagnostic reads a decision at the goal, so the two agree everywhere the
        measurement looks."""
        p, n, b, goal, near = bf.build(2, 2)
        self.assertEqual((n, b, goal, near), (5, 2, 4, 3))
        np.testing.assert_array_equal(
            p.argmax(-1)[:4], [[1, 0], [2, 0], [3, 1], [4, 3]])
        np.testing.assert_array_equal(p.argmax(-1)[goal], [4, 3])

    def test_room_is_closed_and_crossing_is_the_only_way_in(self):
        for L, R in [(1, 2), (2, 3), (4, 5)]:
            p, n, b, goal, near = bf.build(L, R)
            nx = p.argmax(-1)
            room = set(range(L + 1, n))
            for s in room:
                self.assertTrue(set(nx[s]) <= room, f"room leaks at {s}")
            entries = {s for s in range(n) if s not in room and set(nx[s]) & room}
            self.assertEqual(entries, {b})
            self.assertEqual(near, goal - 1)

    def test_exact_successor_is_normalized_and_solves_its_bellman_equation(self):
        for L, R, gamma in [(2, 2, 0.8), (1, 4, 0.5), (3, 3, 0.99)]:
            p, n, *_ = bf.build(L, R)
            K = p.reshape(-1, n)
            pi = np.full((n, 2), 0.5)
            P = bf.exact(K, pi, gamma)
            np.testing.assert_allclose(P.sum(1), 1.0, atol=1e-12)
            V = np.sum(pi[:, :, None] * P.reshape(n, 2, n), axis=1)
            np.testing.assert_allclose(P, (1 - gamma) * K + gamma * K @ V, atol=1e-12)

    def test_each_loss_has_its_own_exact_fixed_point(self):
        """The target each critic regresses reproduces the exact measure."""
        L, R, gamma = 2, 2, 0.8
        p, n, *_ = bf.build(L, R)
        K = p.reshape(-1, n)
        pi = np.full((n, 2), 0.5)
        P = bf.exact(K, pi, gamma)
        V = np.sum(pi[:, :, None] * P.reshape(n, 2, n), axis=1)
        # normsq, softsq and density share this target
        np.testing.assert_allclose((1 - gamma) * K + gamma * K @ V, P, atol=1e-12)
        # fb predicts the density ratio n*M and adds the immediate term as a
        # linear term, so its fixed point is n/(1-gamma) times the same measure
        m = n / (1 - gamma) * P
        Vm = np.sum(pi[:, :, None] * m.reshape(n, 2, n), axis=1)
        np.testing.assert_allclose(n * K + gamma * K @ Vm, m, atol=1e-9)

    def test_readout_puts_every_method_on_the_same_scale(self):
        L, R, gamma = 2, 2, 0.8
        p, n, *_ = bf.build(L, R)
        P = bf.exact(p.reshape(-1, n), np.full((n, 2), 0.5), gamma)
        np.testing.assert_allclose(
            bf.readout(n / (1 - gamma) * P, "fb", gamma, n), P, atol=1e-12)
        for m in ("normsq", "softsq", "expsq", "density"):
            np.testing.assert_allclose(bf.readout(P, m, gamma, n), P, atol=1e-12)

    def test_metrics_are_exact_at_the_exact_measure(self):
        p, n, b, goal, near = bf.build(2, 3)
        P = bf.exact(p.reshape(-1, n), np.full((n, 2), 0.5), 0.8)
        for method, pred in [("fb", n / (1 - 0.8) * P), ("density", P),
                             ("normsq", P), ("softsq", P), ("expsq", P)]:
            m = bf.metrics(pred, method, 0.8, P, goal, b, near)
            self.assertAlmostEqual(m["gap_ratio_near"], 1.0, places=9)
            self.assertAlmostEqual(m["gap_ratio_b"], 1.0, places=9)
            self.assertAlmostEqual(m["rel_err_goal"], 0.0, places=9)
            self.assertEqual(m["wrong_states"], 0)

    def test_spherical_backward_rows_have_the_declared_norm(self):
        rng = np.random.default_rng(0)
        params = {"f": jnp.array(rng.normal(size=(8, 3))),
                  "b": jnp.array(rng.normal(size=(4, 3)))}
        rows = np.asarray(bf.backward(params, True))
        np.testing.assert_allclose(np.linalg.norm(rows, axis=-1), np.sqrt(3), atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(bf.backward(params, False)), np.asarray(params["b"]))

    def test_full_rank_training_recovers_the_decision_for_every_method(self):
        for method in ("fb", "normsq", "softsq", "expsq", "density"):
            rows = bf.run_group(2, 2, 5, method, 0.8, [0], [0.01], 20000, 0.005, False)
            self.assertAlmostEqual(rows[0]["gap_ratio_near"], 1.0, delta=0.05,
                                   msg=f"{method} fails at full rank")


if __name__ == "__main__":
    unittest.main()
