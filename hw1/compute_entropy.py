"""Compute single-character probabilities and entropy for Chinese text corpora."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

CJK_RANGE_START = 0x4E00
CJK_RANGE_END = 0x9FFF


def iter_cjk_chars(text: str) -> Iterable[str]:
	for char in text:
		code_point = ord(char)
		if CJK_RANGE_START <= code_point <= CJK_RANGE_END:
			yield char


def analyze_characters(text: str) -> tuple[Counter[str], int, float]:
	counts = Counter(iter_cjk_chars(text))
	total = sum(counts.values())
	if not total:
		raise ValueError("input corpus does not contain any CJK Unified Ideographs")
	probabilities = (count / total for count in counts.values())
	entropy = -sum(p * math.log2(p) for p in probabilities)
	return counts, total, entropy


def format_top_characters(counts: Counter[str], total: int, top_k: int) -> str:
	lines = []
	header = "rank char count prob information(bits) contribution"
	lines.append(header)
	for rank, (char, count) in enumerate(counts.most_common(top_k), 1):
		probability = count / total
		information_bits = -math.log2(probability)
		contribution = probability * information_bits
		lines.append(
			f"{rank:>4}  {char}   {count:>8}  {probability:>0.6f}"
			f"      {information_bits:>10.6f}       {contribution:>10.6f}"
		)
	return "\n".join(lines)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Compute Chinese character probabilities and Shannon entropy."
	)
	parser.add_argument(
		"corpus",
		type=Path,
		help="Path to the UTF-8 encoded corpus file to analyze.",
	)
	parser.add_argument(
		"--top",
		type=int,
		default=20,
		help="Show the top-K most frequent characters (default: 20).",
	)
	args = parser.parse_args()

	text = args.corpus.read_text(encoding="utf-8", errors="ignore")
	counts, total, entropy = analyze_characters(text)
	unique = len(counts)

	print(f"Corpus: {args.corpus}")
	print(f"Total Chinese characters: {total}")
	print(f"Unique Chinese characters: {unique}")
	print(f"Shannon entropy: {entropy:.6f} bits")

	if args.top > 0:
		print("\nTop characters:")
		print(format_top_characters(counts, total, args.top))


if __name__ == "__main__":
	main()
