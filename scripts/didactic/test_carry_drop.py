"""Independent mathematical checks for the carry/drop diagnostic."""
import unittest
import itertools
import numpy as np
import carry_drop as cd


class CarryDropTests(unittest.TestCase):
    def test_one_step_transport_and_identical_reset_states(self):
        p = cd.build_mdp(1, 1)
        np.testing.assert_allclose(p[0, 0], [0, .9, .1])
        np.testing.assert_allclose(p[0, 1], [0, 0, 1])
        np.testing.assert_allclose(p[1:], [[[1, 0, 0]] * 2] * 2)
        for gamma in (0, .5, .98):
            m = cd.exact_successor(p, gamma)
            self.assertAlmostEqual(m[0, 0, 1] - m[0, 1, 1], .9, places=10)

    def test_exact_representations_and_bellman_equation(self):
        for depth in (1, 3, 6):
            for k in (1, 10, 50):
                p = cd.build_mdp(depth, k)
                m = cd.exact_successor(p, .98)
                np.testing.assert_allclose(m, p + .98 * np.einsum('san,nu->sau', p, m.mean(1)), atol=1e-11)
                np.testing.assert_allclose(m.sum(-1), 50, atol=1e-11)
                self.assertTrue(np.all(m > 0))
                for method in cd.METHODS:
                    params = cd.exact_parameters(m, method, .98)
                    pred = cd.as_probability(cd.predict(params, method), method, .98)
                    np.testing.assert_allclose(pred.reshape(m.shape), .02*m, atol=1e-11)

    def test_exact_solutions_have_zero_training_gradient(self):
        import jax
        for method in cd.METHODS:
            p = cd.build_mdp(3, 50)
            m = cd.exact_successor(p, .98)
            params = cd.exact_parameters(m, method, .98)
            grad = jax.grad(cd.loss_fn)(params, params, p, method, .98)
            self.assertLess(max(float(np.max(np.abs(x))) for x in jax.tree.leaves(grad)), 1e-9, method)

    def test_metrics_detect_reversed_action_values(self):
        p = cd.build_mdp(3, 50)
        m = cd.exact_successor(p, .98)
        good = cd.metrics(.02*m, p, m, .98)
        self.assertEqual(good['wrong_actions'], 0)
        self.assertAlmostEqual(good['start_gap_ratio'], 1)
        self.assertAlmostEqual(good['greedy_return_ratio'], 1)
        bad = cd.metrics(.02*m[:, ::-1], p, m, .98)
        self.assertEqual(bad['wrong_actions'], 3)
        self.assertAlmostEqual(bad['start_gap_ratio'], -1)
        self.assertAlmostEqual(bad['greedy_return_ratio'], 0)

    def test_always_advance_reference_is_optimal_over_all_policies(self):
        # Independent state-value solve and exhaustive deterministic policies
        # protect the claimed control return against a wrong reference policy.
        for depth in (1, 3, 6):
            for k in (1, 10, 50):
                p = cd.build_mdp(depth, k)
                n = depth+2
                returns = []
                for actions in itertools.product((0, 1), repeat=depth):
                    transition = p[np.arange(n), list(actions)+[0, 0]]
                    reward = transition[:, depth]
                    value = np.linalg.solve(np.eye(n)-.98*transition, reward)
                    returns.append(value[0])
                self.assertAlmostEqual(returns[0], max(returns), places=10)


if __name__ == '__main__':
    unittest.main()
