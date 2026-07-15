"""Per-wandb-group experiment inventory: what it was, where the checkpoints live, and how
their correctness was verified. Joins analysis/atlas/MANIFEST.csv with results_map.py.

Correctness levels (from the 2026-07-15 pre-delete audit):
  loaded    the 6 atlas critics + FB: loaded end-to-end through the real atlas loaders.
  checksum  root moved off /dev/shm this session -> rsync -c byte-parity vs volatile = 0,
            + exact file-count parity, + sampled loadability.
  sampled   persistent-only root (no volatile counterpart) -> loadability sampled only
            (12 jax + 12 torch across roots all OK); not every file individually loaded.
"""
import csv
import os

REPO = '/home/adityamohan/git/Factored-FB'
CHECKSUM_DONE = {'gciql_outputs', 'factored-fb'}   # rsync -c confirmed 0 differ (2026-07-15)
CHECKSUM_PENDING = set()
# The 5 jax critics loaded end-to-end. The atlas FB critic is ONE specific dir
# (factored-fb_outputs/2026-05-14_21-34-52__...__fb_flowbc__s0 = [nogroup:fb_flowbc]),
# not the fb_state_plain group -- so FB is handled by path, not by group here.
ATLAS_CRITIC_GROUPS = {
    'GCIQL-State-Flow-Bilinear-d50', 'GCIQL-State-Flow-Bilinear-d50-ddpgbc',
    'GCIQL-State-Flow', 'factored-fb-gciql', 'GCIQL-State-Flow-SimpleSF-Arch'}
ATLAS_FB_DIR = '2026-05-14_21-34-52__cube-single-play-v0__fb_flowbc__s0'

import sys
sys.path.insert(0, os.path.join(REPO, 'scripts'))
from results_map import MAP  # noqa: E402


def describe(group, rows):
    m = MAP.get(group, {})
    if m.get('graft'):
        return m['graft']
    if m.get('variant'):
        return f"{m.get('agent','')}/{m.get('actor','')} {m['variant']}".strip()
    # torch fallback: the variant hint carried in the manifest
    return rows[0].get('variant', '')[:70]


def correctness(group, roots, has_fb_dir):
    if group in ATLAS_CRITIC_GROUPS or has_fb_dir:
        return 'loaded (end-to-end)'
    if roots & CHECKSUM_DONE:
        return 'checksum-OK + count + sampled-load'
    if roots & CHECKSUM_PENDING:
        return 'checksum-PENDING + count + sampled-load'
    return 'sampled-load only (persistent-only root)'


def main():
    rows = [r for r in csv.DictReader(open(f'{REPO}/analysis/atlas/MANIFEST.csv'))]
    groups = {}
    for r in rows:
        g = r['wandb_group'] or f"[nogroup:{r['agent']}]"
        groups.setdefault(g, []).append(r)

    out = []
    for g, rs in sorted(groups.items()):
        real = [r for r in rs if r['is_smoke'] == 'no']
        if not real:
            continue
        roots = {r['root'] for r in real}
        has_fb_dir = any(ATLAS_FB_DIR in r['path'] for r in real)
        steps = sorted({r['max_step'] for r in real})
        out.append(dict(
            group=g, n_seeds=len(real), desc=describe(g, real),
            location='; '.join(sorted(roots)),
            example_path=os.path.dirname(real[0]['path']),
            steps=steps[-1], size_gb=round(sum(float(r['size_mb']) for r in real) / 1000, 1),
            verified=correctness(g, roots, has_fb_dir),
            in_paper='yes' if real[0]['mapped'] == 'yes' else 'no'))

    out.sort(key=lambda r: (-r['size_gb']))
    csvp = f'{REPO}/analysis/atlas/EXPERIMENT_INVENTORY.csv'
    with open(csvp, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['group', 'desc', 'location', 'example_path',
                                           'n_seeds', 'steps', 'size_gb', 'in_paper', 'verified'])
        w.writeheader()
        for r in out:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f'{csvp}: {len(out)} experiment groups\n')
    # console summary by verification level
    import collections
    lvl = collections.Counter(r['verified'] for r in out)
    for k, v in lvl.items():
        print(f'  {v:3} groups: {k}')
    print(f'\n  {sum(1 for r in out if r["in_paper"]=="yes")} groups map to a paper cell; '
          f'{sum(r["n_seeds"] for r in out)} total runs; '
          f'{sum(r["size_gb"] for r in out):.0f} GB')


if __name__ == '__main__':
    main()
