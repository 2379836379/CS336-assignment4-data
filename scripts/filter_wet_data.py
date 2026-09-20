"""Filter WET records into plain-text language-model training data.

The script is intentionally model-agnostic: downloaded fastText models are used
when available, while all structural filters remain usable offline.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from warcio.archiveiterator import ArchiveIterator

from cs336_data.processing import (
    classify_nsfw,
    classify_toxic_speech,
    exact_line_deduplication,
    gopher_quality_filter,
    identify_language,
    mask_emails,
    mask_ips,
    mask_phone_numbers,
)


def process_file(path: Path, output: Path, min_language_score: float = 0.7) -> Counter:
    stats = Counter(input=0, kept=0, language=0, quality=0, harmful=0)
    with gzip.open(path, "rb") as stream, output.open("w", encoding="utf-8") as out:
        for record in ArchiveIterator(stream):
            if record.rec_type != "conversion":
                continue
            stats["input"] += 1
            text = record.content_stream().read().decode("utf-8", errors="replace").strip()
            language, score = identify_language(text)
            if language != "en" or score < min_language_score:
                stats["language"] += 1
                continue
            text, _ = mask_emails(text)
            text, _ = mask_phone_numbers(text)
            text, _ = mask_ips(text)
            nsfw, nsfw_score = classify_nsfw(text)
            toxic, toxic_score = classify_toxic_speech(text)
            if (nsfw == "nsfw" and nsfw_score >= 0.8) or (toxic == "toxic" and toxic_score >= 0.8):
                stats["harmful"] += 1
                continue
            if not gopher_quality_filter(text):
                stats["quality"] += 1
                continue
            out.write(text + "\n")
            stats["kept"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="WET .warc.wet.gz file or directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--stats", type=Path, default=None)
    args = parser.parse_args()
    files = [args.input] if args.input.is_file() else sorted(args.input.glob("*.gz"))
    args.output.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    for path in files:
        stats = process_file(path, args.output / (path.stem + ".txt"))
        totals.update(stats)
        print(json.dumps({"file": str(path), **stats}, sort_keys=True))
    if args.stats:
        args.stats.write_text(json.dumps(dict(totals), indent=2), encoding="utf-8")
    print(json.dumps(dict(totals), sort_keys=True))


if __name__ == "__main__":
    main()
