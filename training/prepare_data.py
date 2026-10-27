"""python -m training.prepare_data --input artifacts/verified.jsonl --output artifacts/sft"""
import argparse
import hashlib
import json
from pathlib import Path
from training.data import prepare_records, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = prepare_records(read_jsonl(args.input))
    if not groups["train"] or not groups["validation"]:
        raise ValueError("Need nonempty train and validation splits; reserve independent families")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "splits": {}}
    for split, rows in groups.items():
        target = args.output / f"{split}.jsonl"
        target.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        manifest["splits"][split] = {"records": len(rows),
                "families": sorted({r["family"] for r in rows}),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
