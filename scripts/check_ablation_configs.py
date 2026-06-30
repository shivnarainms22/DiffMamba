"""Pre-launch verification for the hybrid attention ablation.

Run this ON THE NODE (env active) BEFORE submitting any ablation job:

    python scripts/check_ablation_configs.py

It composes every ablation experiment with Hydra and prints, per config, the
RESOLVED attention-layer schedule plus the controlled hyperparameters. It then
asserts that everything except the attention layout is identical across the six
configs, so a typo (wrong lr / max_steps / batch) is caught here in seconds
rather than after a multi-day run produces an invalid ablation point.

No CUDA / torch / mamba needed: it reads config values only (resolve=False), it
does not build the model.
"""
import os
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hybrid_schedule import attention_layer_indices  # noqa: E402

CONFIGS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'configs'))

# (experiment, expected attention layer indices) — the second value is the
# independent oracle: if the composed config resolves to anything else, the
# config is wrong. A0 (baseline hybrid_130m, [3,7,11]) is included as a control.
EXPECTED = {
    'hybrid_130m': [3, 7, 11],   # A0 baseline (already trained -> PPL 69.60)
    'hyb_e3':      [2, 5, 8, 11],
    'hyb_e6':      [5, 11],
    'hyb_e12':     [11],
    'hyb_early':   [0, 1, 2],
    'hyb_mid':     [4, 5, 6],
    'hyb_late':    [9, 10, 11],
}

# Hyperparameters that MUST be identical across every config (only the attention
# layout is allowed to vary). Values read raw (unresolved) to avoid needing the
# runtime custom resolvers.
CONTROLLED = ['backbone', 'parameterization', 'mode']


def _get(d, path):
    cur = d
    for k in path.split('.'):
        cur = cur[k]
    return cur


def main():
    rows = []
    controls = {}
    ok = True
    with initialize_config_dir(version_base=None, config_dir=CONFIGS_DIR):
        for exp, expected_idx in EXPECTED.items():
            cfg = compose(config_name='config',
                          overrides=[f'+experiment={exp}', 'seed=1'])
            raw = OmegaConf.to_container(cfg, resolve=False)
            m = raw['model']
            every = int(m.get('hybrid_attention_every', 4))
            offset = int(m.get('hybrid_attention_offset', every - 1))
            explicit = m.get('hybrid_attention_layers', None)
            explicit = list(explicit) if explicit is not None else None
            n_blocks = int(m['n_blocks'])
            idx = attention_layer_indices(n_blocks, every, offset,
                                          explicit=explicit)

            max_steps = _get(raw, 'trainer.max_steps')
            lr = _get(raw, 'optim.lr')
            gbs = _get(raw, 'loader.global_batch_size')
            wname = _get(raw, 'wandb.name')

            match = '  OK' if idx == expected_idx else '  *** MISMATCH ***'
            if idx != expected_idx:
                ok = False
            rows.append((exp, idx, len(idx), max_steps, lr, gbs, wname, match))

            controls[exp] = {k: _get(raw, k) for k in CONTROLLED}
            controls[exp]['max_steps'] = max_steps
            controls[exp]['lr'] = lr
            controls[exp]['global_batch_size'] = gbs

    print(f'{"experiment":<14}{"attn_layers":<18}{"#":<3}'
          f'{"max_steps":<11}{"lr":<8}{"gbs":<5}{"wandb":<12}status')
    print('-' * 84)
    for exp, idx, nidx, ms, lr, gbs, wn, match in rows:
        print(f'{exp:<14}{str(idx):<18}{nidx:<3}{str(ms):<11}'
              f'{str(lr):<8}{str(gbs):<5}{wn:<12}{match}')

    # All ablation configs must share the controlled hyperparameters (compare
    # the variants against the hybrid_130m baseline; only attention may differ).
    base = controls['hybrid_130m']
    print('\nControlled-hyperparameter check (must match hybrid_130m):')
    for exp, c in controls.items():
        if exp == 'hybrid_130m':
            continue
        diffs = {k: (base[k], c[k]) for k in base if base[k] != c[k]}
        if diffs:
            ok = False
            print(f'  {exp}: *** DIFFERS ***  {diffs}')
        else:
            print(f'  {exp}: OK')

    print('\n' + ('ALL CHECKS PASSED — safe to submit.' if ok
                  else '*** CHECKS FAILED — DO NOT SUBMIT, fix configs first. ***'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
