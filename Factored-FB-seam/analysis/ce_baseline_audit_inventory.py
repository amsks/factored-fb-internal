"""Read-only local run provenance inventory for HILP/RLDP categorical audit."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('batch_size', 'z_dim', 'L_dim', 'discount', 'dataset_class', 'horizon',
          'backward_input', 'goal_kernel', 'measure_hit_spec', 'temp', 'actor_temp',
          'measure_density_temp', 'measure_density_actor_temp', 'actor_q_center',
          'measure_density_hard', 'hilp_q_loss', 'hilp_head_reduce', 'hilp_actor_head_reduce',
          'hilp_sf_loss_scale', 'hilp_sf_archi', 'hilp_independent_target_init',
          'hilp_mix_ratio', 'hilp_p_randomgoal', 'group_lr', 'stages_from_pre_step_params',
          'train_goal_ratio', 'rldp_two_phase', 'rldp_fb_target_online_b', 'rldp_ortho_inert',
          'rldp_actor_pessimism', 'rldp_no_clamp_windows', 'rldp_sp_port_alignment',
          'eval_relabel_size', 'eval_relabel_shift', 'eval_relabel_episodes')


def eval_summary(path):
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    per_step = defaultdict(list)
    inferred_flags = set()
    for row in rows:
        if row.get('inferred_cond'):
            inferred_flags.add(row['inferred_cond'])
        try:
            step, success = int(float(row['step'])), float(row['success'])
            per_step[step].append(success)
        except (KeyError, ValueError, TypeError):
            pass
    steps = sorted(per_step)
    if not steps:
        return {'rows': len(rows), 'success_summary': 'no success/step columns'}
    means = [sum(per_step[s]) / len(per_step[s]) for s in steps]
    return {'rows': len(rows), 'steps': steps, 'rows_per_step': {s: len(per_step[s]) for s in steps},
            'last_success': means[-1], 'last_three_step_success': sum(means[-3:]) / min(3, len(means)),
            'inferred_cond_flags': sorted(inferred_flags)}


records = []
for path in sorted((ROOT / 'exp').rglob('config.json')):
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        continue
    critic = raw.get('critic', '')
    if not critic.startswith(('hilp', 'rldp')):
        continue
    config = raw.get('config', raw)
    run = path.parent
    checkpoints = sorted(int(p.stem.split('_')[-1]) for p in run.glob('params_*.pkl')
                         if p.stem.split('_')[-1].isdigit())
    records.append({'path': str(path.relative_to(ROOT)), 'critic': critic, 'seed': raw.get('seed'),
                    'domain': raw.get('domain'), 'env_name': config.get('env_name'),
                    'discarded': '_discarded' in path.parts,
                    'config_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'source_revision_saved': any(k in raw for k in ('commit', 'git_commit', 'git_sha', 'source_sha256')),
                    'config': {key: config.get(key) for key in FIELDS},
                    'architectures': {key: config.get(key) for key in
                                      ('phi', 'sf', 'backward', 'forward', 'hilp_left_encoder', 'actor')},
                    'checkpoint_steps': checkpoints,
                    'goal_eval': eval_summary(run / 'eval.csv'),
                    'reward_inference_eval': eval_summary(run / 'eval_relabel.csv')})
report = {'scope': 'Local configs and saved CSVs only. No fresh training/evaluation; no inference of run source commit from current checkout.',
          'run_count': len(records), 'critic_counts': dict(Counter(r['critic'] for r in records)),
          'source_revision_saved_count': sum(r['source_revision_saved'] for r in records),
          'records': records}
output = ROOT / 'analysis' / 'ce_baseline_audit_inventory.json'
output.write_text(json.dumps(report, indent=2) + '\n')
print('Run configs:', len(records), 'with explicit source revision:', report['source_revision_saved_count'])
for field in ('rldp_two_phase', 'rldp_fb_target_online_b', 'rldp_ortho_inert',
              'rldp_actor_pessimism', 'rldp_no_clamp_windows', 'measure_density_temp'):
    enabled = [r for r in records if r['config'].get(field)]
    print(field, 'enabled:', len(enabled), dict(Counter(r['critic'] for r in enabled)))
for critic in ('hilp_tdjc', 'hilp_density_tdjc', 'rldp', 'rldp_density',
               'hilp_density_gk', 'rldp_density_gk', 'hilp_density_gk_oc', 'rldp_density_gk_oc'):
    rows = [r for r in records if r['critic'] == critic and r['env_name'] == 'cube-single-play-v0']
    print(critic, [{'seed': r['seed'], 'checkpoints': r['checkpoint_steps'],
                    'A_final': (r['goal_eval'] or {}).get('last_success'),
                    'B_final': (r['reward_inference_eval'] or {}).get('last_success'),
                    'batch_size': r['config']['batch_size'], 'backward_input': r['config']['backward_input'],
                    'goal_kernel': r['config']['goal_kernel']} for r in rows])
print(output)
