"""Throwaway probe: which candidate VLM datasets load under datasets 4.x?

Streams one record from each candidate and prints OK + column schema, or FAIL
+ the reason. Used to pick the align (captioning) and SFT (VQA) sources for the
Stage-1 training run. Run on a node with internet:  python scripts/probe_datasets.py
"""
from datasets import load_dataset

CANDIDATES = [
    ("pixparse/cc3m-wds", "train"),
    ("lmms-lab/COCO-Caption2017", "val"),
    ("nielsr/coco-karpathy", "train"),
    ("lmms-lab/VQAv2", "validation"),
    ("merve/vqav2-small", "validation"),
    ("HuggingFaceM4/the_cauldron", "train"),
]


def main():
    for name, split in CANDIDATES:
        try:
            ds = load_dataset(name, split=split, streaming=True)
            rec = next(iter(ds))
            schema = {k: type(v).__name__ for k, v in rec.items()}
            print("OK  ", name, "|", split, "|", schema)
        except Exception as e:  # noqa: BLE001 - probe, want every failure reason
            print("FAIL", name, "|", repr(e)[:140])


if __name__ == "__main__":
    main()
