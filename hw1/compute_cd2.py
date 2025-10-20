"""Compute English word probabilities and entropy for text corpora."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

WORD_PATTERN = re.compile(r"[A-Za-z]+")


def iter_english_words(text: str, case_sensitive: bool) -> Iterable[str]:
	"""Yield ASCII alphabetic word tokens, normalizing case unless requested."""
	for match in WORD_PATTERN.finditer(text):
		word = match.group(0)
		start = match.start()
		if start > 0 and text[start - 1] in {"'", "’"} and word.lower() == "s":
			continue
		yield word if case_sensitive else word.lower()


def analyze_words(text: str, case_sensitive: bool) -> tuple[Counter[str], int, float]:
	counts = Counter(iter_english_words(text, case_sensitive))
	total = sum(counts.values())
	if not total:
		raise ValueError("input corpus does not contain any ASCII alphabetic words")
	probabilities = (count / total for count in counts.values())
	entropy = -sum(p * math.log2(p) for p in probabilities)
	return counts, total, entropy


def format_top_words(counts: Counter[str], total: int, top_k: int) -> str:
	lines = []
	header = "rank word                count prob information(bits) contribution"
	lines.append(header)
	for rank, (word, count) in enumerate(counts.most_common(top_k), 1):
		probability = count / total
		information_bits = -math.log2(probability)
		contribution = probability * information_bits
		lines.append(
			f"{rank:>4}  {word:<18} {count:>8}  {probability:>0.6f}"
			f"      {information_bits:>10.6f}       {contribution:>10.6f}"
		)
	return "\n".join(lines)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Compute English word probabilities and Shannon entropy."
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
		help="Show the top-K most frequent words (default: 20).",
	)
	parser.add_argument(
		"--case-sensitive",
		action="store_true",
		help="Treat uppercase and lowercase words as distinct tokens.",
	)
	args = parser.parse_args()

	text = args.corpus.read_text(encoding="utf-8", errors="ignore")
	counts, total, entropy = analyze_words(text, args.case_sensitive)
	unique = len(counts)

	print(f"Corpus: {args.corpus}")
	print(f"Total words: {total}")
	print(f"Unique words: {unique}")
	print(f"Shannon entropy: {entropy:.6f} bits")

	if args.top > 0:
		print("\nTop words:")
		print(format_top_words(counts, total, args.top))


if __name__ == "__main__":
	main()
