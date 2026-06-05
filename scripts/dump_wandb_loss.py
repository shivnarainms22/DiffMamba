"""Print loss curves from an OFFLINE wandb run's local .wandb file.

Offline wandb stores history in a binary transaction log (run-*.wandb); there
is no CSV/JSONL. This reads that log directly (no server, no sync) and prints
the understand vs generate (vs text) loss over steps.

Usage:
    python scripts/dump_wandb_loss.py                 # auto-discover
    python scripts/dump_wandb_loss.py /path/run-xyz.wandb
    python scripts/dump_wandb_loss.py /path/to/wandb  # a dir to search
"""
import glob
import json
import os
import sys

SEARCH_ROOTS = [
    os.path.expanduser('~/DiffMamba/wandb'),
    '/scratch/sarin.s/DiffMamba/runs/uni_stage3/wandb',
    '/scratch/sarin.s/DiffMamba/runs/uni_stage3',
    os.path.join(os.path.dirname(__file__), '..', 'wandb'),
]
KEYS = ['train/understand_loss', 'train/generate_loss', 'train/text_loss']


def find_wandb_files(arg):
    if arg and arg.endswith('.wandb') and os.path.isfile(arg):
        return [arg]
    roots = [arg] if arg else SEARCH_ROOTS
    found = []
    for root in roots:
        if root and os.path.isdir(root):
            found += glob.glob(os.path.join(root, '**', '*.wandb'),
                               recursive=True)
    return sorted(set(found))


def iter_history(path):
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto import wandb_internal_pb2 as pb
    ds = DataStore()
    ds.open_for_scan(path)
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof('record_type') != 'history':
            continue
        row = {}
        for item in rec.history.item:
            key = item.key or '.'.join(item.nested_key)
            try:
                row[key] = json.loads(item.value_json)
            except Exception:
                row[key] = item.value_json
        yield row


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    files = find_wandb_files(arg)
    if not files:
        print('No .wandb files found. Pass the path explicitly, e.g.:')
        print('  python scripts/dump_wandb_loss.py /scratch/.../wandb')
        return

    for path in files:
        rows = []
        try:
            for row in iter_history(path):
                if any(k in row for k in KEYS):
                    rows.append(row)
        except Exception as e:  # noqa: BLE001 - report and keep going
            print(f'\n=== {path} ===\n  could not parse: {e}')
            continue
        if not rows:
            print(f'\n=== {path} === (no loss history)')
            continue
        rows.sort(key=lambda r: r.get('_step', 0))
        print(f'\n=== {path}  ({len(rows)} logged steps) ===')
        print(f'{"step":>8}  {"understand":>11}  {"generate":>11}  {"text":>11}')
        for r in rows:
            def g(k):
                v = r.get(k)
                return f'{v:11.4f}' if isinstance(v, (int, float)) else f'{"-":>11}'
            print(f'{int(r.get("_step", -1)):>8}  '
                  f'{g("train/understand_loss")}  '
                  f'{g("train/generate_loss")}  {g("train/text_loss")}')


if __name__ == '__main__':
    main()
