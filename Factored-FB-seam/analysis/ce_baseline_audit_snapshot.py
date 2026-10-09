"""Snapshot current audited source and scan local baseline training logs; read only."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    'impls/critics/rldp.py', 'impls/critics/rldp_density.py',
    'impls/critics/hilp.py', 'impls/critics/hilp_density.py',
    'impls/critics/density_fb.py', 'impls/actors/flowbc.py',
    'impls/agents/__init__.py', 'impls/utils/datasets.py',
    'impls/utils/eval_cond.py', 'impls/main.py', 'impls/utils/flax_utils.py',
    'env/config.py', 'env/ogbench.py', 'env/unified.py',
    'impls/configs/critic/hilp_density_tdjc.yaml',
    'impls/configs/critic/hilp_density_gk.yaml',
    'impls/configs/critic/hilp_density_gk_oc.yaml',
    'impls/configs/critic/rldp_density.yaml',
    'impls/configs/critic/rldp_density_official.yaml',
)
source = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
          for name in FILES if (ROOT / name).exists()}
inventory = json.loads((ROOT / 'analysis/ce_baseline_audit_inventory.json').read_text())
scanned = []
nonfinite = []
for run in inventory['records']:
    path = ROOT / run['path']
    train = path.with_name('train.csv')
    if not train.exists():
        continue
    count = 0
    with train.open() as f:
        for row_number, row in enumerate(csv.DictReader(f), 2):
            count += 1
            for key, value in row.items():
                if key is None or value in (None, ''):
                    continue
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(v):
                    nonfinite.append({'path': str(train.relative_to(ROOT)),
                                      'line': row_number, 'step': row.get('step'),
                                      'metric': key, 'value': value})
    scanned.append({'path': str(train.relative_to(ROOT)), 'rows': count,
                    'sha256': hashlib.sha256(train.read_bytes()).hexdigest()})
report = {
    'scope': 'Current source snapshot is not a historical run source manifest. '
             'Logged metrics are periodic samples and cannot rule out unlogged transient failures.',
    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'source_sha256': source,
    'train_csv_count': len(scanned), 'train_row_count': sum(r['rows'] for r in scanned),
    'nonfinite_metric_count': len(nonfinite), 'nonfinite_metrics': nonfinite,
    'train_csvs': scanned,
}
output = Path(__file__).with_suffix('.json')
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ('git_head', 'train_csv_count', 'train_row_count',
                                       'nonfinite_metric_count')}, indent=2))
print('Nonfinite by metric:', dict(Counter(r['metric'] for r in nonfinite)))
print(output)
