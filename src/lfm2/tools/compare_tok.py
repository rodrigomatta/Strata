#!/usr/bin/env python3
"""Compare a local LFM2 runner against an independently supplied tokenizer.json.

Requires gguf and tokenizers. No downloads and no GPU calls are made here.
"""
import argparse
import random
import subprocess

import gguf
from tokenizers import Tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", required=True)
    p.add_argument("--gguf", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--fuzz", type=int, default=0, help="extra deterministic mixed-Unicode cases")
    args = p.parse_args()
    reader = gguf.GGUFReader(args.gguf, "r")

    def field(name, default=None):
        value = reader.get_field(name)
        return value.contents() if value is not None else default

    ref = Tokenizer.from_file(args.tokenizer)
    # Check identity, not just a few examples from an unrelated vocabulary.
    tokens = field("tokenizer.ggml.tokens")
    types = field("tokenizer.ggml.token_type")
    vocab = ref.get_vocab(with_added_tokens=True)
    unused = 0
    for i, text in enumerate(tokens):
        if types[i] == 5 and text not in vocab:  # GGUF UNUSED padding, not encodable text
            unused += 1
            continue
        if vocab.get(text) != i:
            raise AssertionError(f"vocabulary mismatch at {i}: {text!r}")
    for text, i in vocab.items():
        if i >= len(tokens) or tokens[i] != text:
            raise AssertionError(f"reference token absent from GGUF: {i}: {text!r}")
    pre = field("tokenizer.ggml.pre")
    if pre not in ("lfm2", "llama3"):
        raise AssertionError(f"unsupported pre-tokenizer {pre!r}")
    bos = field("tokenizer.ggml.bos_token_id", -1)
    add_bos = bool(field("tokenizer.ggml.add_bos_token", bos >= 0))
    print(f"vocabulary: {len(vocab)} matching IDs; {unused} GGUF UNUSED padding entries; "
          f"pre={pre}, bos={bos}, add_bos={add_bos}")
    cases = [
        ("english", "Hello, world! The quick brown fox jumps over the lazy dog."),
        ("portuguese", "Olá! A ação, o coração e o café não são difíceis. Você está bem?"),
        ("numbers", "12345 3.14 2026 000000001 -42 1,234.56"),
        ("contraction", "it's It's IT'S we're I'LL they've can't"),
        ("chat", "<|im_start|>user\noi<|im_end|>\n<|im_start|>assistant\n"),
    ]
    rng = random.Random(42)
    atoms = ["a", "HELLO", "é", "ação", "\u0301", "中", "日", "🙂", "🚀", "123456",
             "it's", "'RE", " ", "  ", "\t", "\n", "\r\n", "\u00a0", "\u2003",
             ".", "?!", "²", "٣", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    cases += [(f"fuzz-{i}", "".join(rng.choices(atoms, k=rng.randrange(1, 25))))
              for i in range(args.fuzz)]
    failed = 0
    for label, text in cases:
        # tokenizer.json's ByteLevel postprocessor does not add BOS. Test the
        # raw encoding independently, then test the GGUF insertion policy.
        raw = ref.encode(text, add_special_tokens=False).ids
        for with_bos in (False, True):
            command = [args.binary, "--tok", args.gguf, text]
            if not with_bos:
                command.append("--no-bos")
            run = subprocess.run(command, check=True, capture_output=True, text=True)
            got = [int(x) for x in run.stdout.split()]
            want = ([bos] if with_bos and add_bos else []) + raw
            if got != want:
                failed += 1
                print(f"FAIL {label} bos={with_bos}: {text!r}\n expected {want}\n got      {got}")
        if not label.startswith("fuzz"):
            print(f"checked {label}")
    print(f"{len(cases) * 2} comparisons, {failed} failures")
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
