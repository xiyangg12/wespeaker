#!/usr/bin/env python3
"""Create a portable Kaldi wav.scp by recursively scanning a dataset."""

import argparse
import re
from pathlib import Path


def utterance_id(relative_path: Path) -> str:
    """Derive a whitespace-free utterance ID from a relative audio path."""
    raw_id = "__".join(relative_path.with_suffix("").parts)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan WAV files and write '<utterance-id> <relative-path>'."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_scp", type=Path)
    parser.add_argument(
        "subdirectories",
        nargs="*",
        help="Optional subdirectories to scan; the whole dataset is scanned by default.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        parser.error(f"dataset root does not exist: {dataset_root}")

    scan_roots = (
        [dataset_root / subdirectory for subdirectory in args.subdirectories]
        if args.subdirectories
        else [dataset_root]
    )
    for scan_root in scan_roots:
        if not scan_root.is_dir():
            parser.error(f"subdirectory does not exist: {scan_root}")

    wav_paths = sorted(
        path
        for scan_root in scan_roots
        for path in scan_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )
    if not wav_paths:
        parser.error("no .wav files found")

    entries = []
    seen_ids = set()
    for wav_path in wav_paths:
        relative_path = wav_path.relative_to(dataset_root)
        if any(character.isspace() for character in relative_path.as_posix()):
            parser.error(
                "audio paths containing whitespace are unsupported by this "
                f"WeSpeaker loader: {relative_path}"
            )
        utt_id = utterance_id(relative_path)
        if utt_id in seen_ids:
            parser.error(f"duplicate generated utterance ID: {utt_id}")
        seen_ids.add(utt_id)
        entries.append(f"{utt_id} {relative_path.as_posix()}\n")

    output_scp = args.output_scp.expanduser()
    output_scp.parent.mkdir(parents=True, exist_ok=True)
    output_scp.write_text("".join(entries), encoding="utf-8")
    print(f"Wrote {len(entries)} entries to {output_scp}")


if __name__ == "__main__":
    main()
