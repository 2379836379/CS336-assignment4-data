"""Serialize newline-delimited documents with the GPT-2 tokenizer."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def tokenize(input_path: Path, output_path: Path, tokenizer_name: str = "gpt2") -> int:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ids: list[int] = []
    with input_path.open(encoding="utf-8") as stream:
        for line in stream:
            ids.extend(tokenizer.encode(line.rstrip("\n")))
            ids.append(tokenizer.eos_token_id)
    np.asarray(ids, dtype=np.uint16).tofile(output_path)
    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", default="gpt2")
    args = parser.parse_args()
    count = tokenize(args.input, args.output, args.tokenizer)
    print(f"wrote {count} tokens to {args.output}")


if __name__ == "__main__":
    main()
