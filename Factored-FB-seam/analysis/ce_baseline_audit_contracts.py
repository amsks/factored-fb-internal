"""Tiny shared-dataset checks for the categorical-baseline audit; no training."""
from pathlib import Path
import json
import numpy as np
from types import SimpleNamespace
from impls.utils.datasets import FBDataset, WindowedDataset, HILPFutureDataset
from impls.critics.rldp import RLDPCritic

base = FBDataset.create(observations=np.arange(10, dtype=np.float32)[:, None],
                        actions=np.arange(10, dtype=np.float32)[:, None],
                        terminals=np.array([0] * 9 + [1], dtype=np.float32),
                        valids=np.array([1] * 9 + [0], dtype=np.float32))
idxs = np.array([8])
regular = WindowedDataset(base, {'horizon': 5}).sample(1, idxs)
official = WindowedDataset(base, {'horizon': 5, 'rldp_no_clamp_windows': True}).sample(1, idxs)
hilp = HILPFutureDataset(base, {'hilp_future': .99}).sample(1, idxs)
sp_actions, sp_targets = RLDPCritic._sp_pairs(SimpleNamespace(horizon=5), official)
report = {
    'scope': 'Actual shared sampler on ten monotonic states; reproduces boundary alignment without neural training.',
    'ordinary_window': {k: regular[k].tolist() for k in
                        ('observations', 'next_observations', 'future_observations', 'future_actions', 'terminals')},
    'no_clamp_window': {k: official[k].tolist() for k in
                        ('observations', 'next_observations', 'future_observations', 'future_actions', 'terminals')},
    'hilp_emitted_terminals': hilp['terminals'].tolist(),
    'base_emitted_terminals_even_if_terminal_index_requested': base.sample(1, np.array([9]))['terminals'].tolist(),
    'actual_rldp_sp_rollout': {
        'initial_state': official['observations'][0, 0].item(),
        'actions': [float(a[0, 0]) for a in sp_actions],
        'target_states': [float(s[0, 0]) for s in sp_targets],
        'contract': 'The rollout starts at B(observations), then recursively applies all these actions; '
                    'it does not restart at the shifted window start.',
    },
    'terminal_semantics': {
        'gamma': .8,
        'single_terminal_arrival_reward': 1.0,
        'episodic_successor_value': 1.0,
        'episodic_value_times_one_minus_gamma': .2,
        'categorical_terminal_target_integral': 1.0,
        'categorical_integral_divided_by_one_minus_gamma': 5.0,
        'two_action_ranking_example': {
            'episodic_terminal_reward_one_value': 1.0,
            'continuing_reward_point_three_value': 1.5,
            'categorical_terminal_value': 1.0,
            'categorical_continuing_value': .3,
            'interpretation': 'A one-time terminal reward1 is worse than reward.3 forever '
                              'under episodic discounted returns, but the terminal one-hot '
                              'categorical readout reverses that action ranking.',
        },
        'interpretation': 'The one-hot terminal branch corresponds to an absorbing outcome '
                          'when converted by 1/(1-gamma), not a single episodic terminal reward. '
                          'Current FBDataset/WindowedDataset/HILPFutureDataset emit zero terminals.',
    },
}
assert official['observations'][0, 0] == 8
assert official['next_observations'][0, 0] == 9
assert official['future_observations'][0, :, 0].tolist() == [6, 7, 8, 9]
assert official['future_actions'][0, :, 0].tolist() == [5, 6, 7, 8]
assert [float(s[0, 0]) for s in sp_targets] == [9., 6., 7., 8., 9.]
assert np.all(hilp['terminals'] == 0)
output = Path(__file__).with_suffix('.json')
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
