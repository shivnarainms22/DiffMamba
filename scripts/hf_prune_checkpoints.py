#!/usr/bin/env python3
"""Prune redundant checkpoints from the diffmamba-checkpoints HF repo.

Dry-run by default: prints the KEEP/DELETE plan with sizes and deletes NOTHING.
Pass --confirm to apply the deletion (a single commit).

Policy
------
* Legacy flat directories (the old RunPod-era upload layout) are each duplicated
  by a runs/<name>/ tree that keeps best+last, so they are deleted wholesale.
* Under runs/<run>/checkpoints/: keep best.ckpt and last.ckpt; delete every
  numbered step checkpoint (step=*.ckpt / step_*.ckpt).
* Safety guard: a numbered checkpoint is NOT deleted if doing so would leave its
  run with zero checkpoints (e.g. a run that only ever uploaded a step= file).
* --drop-dir <prefix> deletes a run directory ENTIRELY (best+last included), for
  runs you no longer want at all (repeatable).

Usage
-----
  python scripts/hf_prune_checkpoints.py                       # preview
  python scripts/hf_prune_checkpoints.py --drop-dir runs/s50_lr2e3   # + nuke a run
  python scripts/hf_prune_checkpoints.py --confirm             # apply
"""
import argparse
import re
from collections import defaultdict

from huggingface_hub import HfApi, CommitOperationDelete

DEFAULT_REPO = "Shiv-22/diffmamba-checkpoints"

# Old flat-layout dirs. Each is reproduced under runs/<name>/checkpoints/ with
# best+last kept, so nothing unique is lost by removing them entirely.
LEGACY_PREFIXES = [
    "runB_transformer_130m/",
    "runD_130m_seed1/",
    "runD_130m_seed2/",
    "runD_130m_lr1e3_seed1/",
    "scaling_50m/",
]
KEEP_NAMES = {"best.ckpt", "last.ckpt"}
STEP_RE = re.compile(r"step[=_][^/]*\.ckpt$")


def human(n):
    return f"{n / 1e9:.2f}GB"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--drop-dir", action="append", default=[],
                    help="run dir prefix to delete ENTIRELY (repeatable)")
    ap.add_argument("--confirm", action="store_true",
                    help="actually delete (default is dry-run)")
    args = ap.parse_args()

    api = HfApi()
    tree = [t for t in api.list_repo_tree(args.repo, recursive=True)
            if hasattr(t, "size")]
    size = {t.path: t.size for t in tree}
    files = list(size)
    if not files:
        print(f"{args.repo}: no files found."); return

    drop_prefixes = LEGACY_PREFIXES + [d.rstrip("/") + "/" for d in args.drop_dir]

    by_dir = defaultdict(list)
    for f in files:
        by_dir[f.rsplit("/", 1)[0] if "/" in f else ""].append(f)

    delete, keep = [], []
    for f in files:
        name = f.rsplit("/", 1)[-1]
        d = f.rsplit("/", 1)[0] if "/" in f else ""
        if any(f.startswith(p) for p in drop_prefixes):
            delete.append(f)
        elif name in KEEP_NAMES:
            keep.append(f)
        elif STEP_RE.search(name):
            # Only delete a numbered step if a kept checkpoint will survive in the
            # same dir — never leave a run with nothing.
            survivors = [g for g in by_dir[d]
                         if g.rsplit("/", 1)[-1] in KEEP_NAMES
                         and not any(g.startswith(p) for p in drop_prefixes)]
            (delete if survivors else keep).append(f)
        else:
            keep.append(f)

    keep_sz = sum(size[f] for f in keep)
    del_sz = sum(size[f] for f in delete)
    print(f"repo: {args.repo}")
    print(f"\n=== KEEP ({len(keep)} files, {human(keep_sz)}) ===")
    for f in sorted(keep):
        print(f"  keep   {human(size[f]):>9}  {f}")
    print(f"\n=== DELETE ({len(delete)} files, {human(del_sz)}) ===")
    for f in sorted(delete):
        print(f"  DELETE {human(size[f]):>9}  {f}")
    print(f"\nreclaim {human(del_sz)};  {len(files)} -> {len(keep)} files "
          f"({human(keep_sz)} remaining)")

    if not args.confirm:
        print("\nDRY RUN — nothing deleted. Re-run with --confirm to apply.")
        return
    if not delete:
        print("\nnothing to delete."); return

    ops = [CommitOperationDelete(path_in_repo=f) for f in delete]
    api.create_commit(repo_id=args.repo, operations=ops,
                      commit_message=f"prune {len(delete)} redundant checkpoints")
    print(f"\nDeleted {len(delete)} files ({human(del_sz)}) in one commit.")


if __name__ == "__main__":
    main()
