"""Print the understand vs generate (vs text) loss curves from a unified run.

Reads the CSVLogger metrics.csv files and shows, per version dir, the loss
terms over training steps so we can see whether understanding-loss climbed
while generation-loss fell (loss imbalance) or trained down normally.

Usage:
    python scripts/plot_uni_loss.py
    python scripts/plot_uni_loss.py /path/to/runs/uni_stage3/logs
"""
import glob
import os
import sys

import pandas as pd

DEFAULT_LOGS = '/scratch/sarin.s/DiffMamba/runs/uni_stage3/logs'
WANT = ['train/understand_loss', 'train/generate_loss', 'train/text_loss']


def main():
    logs_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOGS
    pattern = os.path.join(logs_dir, 'version_*', 'metrics.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f'No metrics.csv under {pattern}')
        return

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 200)

    for f in files:
        df = pd.read_csv(f)
        cols = [c for c in WANT if c in df.columns]
        if not cols:
            print(f'\n=== {f} === (no train loss columns)')
            continue
        out = (df[['step'] + cols]
               .dropna(subset=cols, how='all')
               .groupby('step').last())
        if out.empty:
            print(f'\n=== {f} === (empty)')
            continue
        print(f'\n=== {f}  ({len(out)} logged steps, '
              f'{out.index.min()}-{out.index.max()}) ===')
        print(out.to_string())


if __name__ == '__main__':
    main()
